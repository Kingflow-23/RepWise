"""HTTP routes for the RepWise demo."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import sqlite3

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.config import AppConfig
from ..db.repository import StudentRepository
from ..observability import configure_logging, request_id_context
from .auth import create_access_token, hash_password, read_access_token, verify_password
from ..generation.pipeline import PlanPipeline
from ..llm.ollama import OllamaGenerator, PlanGenerator
from ..retrieval.base import MissingRetriever, Retriever
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.semantic import SemanticRetriever, load_chunks
from ..retrieval.sparse import BM25Retriever
from ..schemas.models import (
    AccessToken,
    AccountCredentials,
    PlanOptions,
    PlanRequest,
    PlanResult,
    TrainingPlan,
    UserProfile,
)

logger = logging.getLogger(__name__)


def create_app(
    config: AppConfig | None = None,
    *,
    repository: StudentRepository | None = None,
    retriever: Retriever | None = None,
    generator: PlanGenerator | None = None,
) -> FastAPI:
    """Build the API, persistence layer, retriever, generator, and routes.

    Args:
        config: Application settings, or ``None`` to load them from the environment.
        repository: Optional repository implementation; it is initialized before
            the app is returned.
        retriever: Optional retrieval service. If omitted, the configured corpus
            and retrieval strategy are used.
        generator: Optional plan-generation service; if omitted, a local Ollama
            generator is configured from the settings.

    Returns:
        A configured FastAPI application. If default retriever initialization
        fails, the app is still created with a ``MissingRetriever`` so health and
        documentation routes remain available.
    """
    config = config or AppConfig.from_environment()
    configure_logging("api")
    logger.info(
        "Initializing API database=%s retriever=%s model=%s",
        config.database_path,
        config.retrieval_strategy,
        config.ollama_model,
    )
    repository = repository or StudentRepository(config.database_path)
    repository.initialize()
    logger.info("Database initialized")
    if retriever is None:
        retrieval_started = perf_counter()
        try:
            if not config.chunk_file:
                raise FileNotFoundError(
                    "Set REPWISE_CHUNK_FILE to the prepared JSONL corpus before starting the API."
                )
            chunks = load_chunks(Path(config.chunk_file).parent)
            lexical = BM25Retriever(chunks)
            if config.retrieval_strategy == "bm25":
                retriever = lexical
            elif config.retrieval_strategy == "semantic":
                retriever = SemanticRetriever(
                    chunks, config.embedding_index, config.embedding_model
                )
            elif config.retrieval_strategy == "hybrid":
                semantic = SemanticRetriever(
                    chunks, config.embedding_index, config.embedding_model
                )
                retriever = HybridRetriever(lexical, semantic)
            else:
                raise ValueError("REPWISE_RETRIEVAL must be bm25, semantic or hybrid.")
            logger.info(
                "Retriever initialized strategy=%s chunks=%s elapsed_ms=%.1f",
                config.retrieval_strategy,
                len(chunks),
                (perf_counter() - retrieval_started) * 1000,
            )
        except (OSError, RuntimeError, ValueError) as error:
            # Keep health and API docs available, but preserve the real setup error.
            retriever = MissingRetriever(str(error))
            logger.exception(
                "Retriever initialization failed strategy=%s elapsed_ms=%.1f",
                config.retrieval_strategy,
                (perf_counter() - retrieval_started) * 1000,
            )
    else:
        logger.info("Using injected retriever service")
    generator = generator or OllamaGenerator(config.ollama_model, config.ollama_url)
    if not isinstance(generator, OllamaGenerator):
        logger.info("Using injected plan generator service")
    pipeline = PlanPipeline(retriever, generator, config.top_k)
    logger.info("Generation pipeline initialized top_k=%s", config.top_k)

    app = FastAPI(
        title="RepWise",
        version="0.1.0",
        description="Course demo API for personalised workout plans.",
    )
    bearer = HTTPBearer(auto_error=False)

    @app.middleware("http")
    async def log_http_request(request: Request, call_next):
        """Log an HTTP request and attach a request ID to its response.

        Args:
            request: Incoming Starlette request.
            call_next: Middleware continuation that dispatches the request.

        Returns:
            The downstream response, with an ``X-Request-ID`` header.

        Notes:
            The request-ID logging context is reset even when downstream handling
            raises; such exceptions are logged and re-raised.
        """
        request_id = str(uuid4())
        context_token = request_id_context.set(request_id)
        started = perf_counter()
        try:
            response = await call_next(request)
            elapsed_ms = (perf_counter() - started) * 1000
            response.headers["X-Request-ID"] = request_id
            route = request.scope.get("route")
            route_path = route.path if route else "unmatched"
            logger.info(
                "HTTP request completed method=%s route=%s status=%s elapsed_ms=%.1f",
                request.method,
                route_path,
                response.status_code,
                elapsed_ms,
            )
            return response
        except Exception:
            logger.exception(
                "HTTP request failed method=%s route=%s",
                request.method,
                request.scope.get("route").path
                if request.scope.get("route")
                else "unmatched",
            )
            raise
        finally:
            request_id_context.reset(context_token)

    def require_account(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> str:
        """Authenticate a bearer token and return its account ID.

        Args:
            credentials: Optional HTTP bearer credentials extracted by FastAPI.

        Returns:
            The internal account ID from the verified token.

        Raises:
            HTTPException: With status 401 for missing or invalid credentials, or
                503 when the application authentication secret is not configured.
        """
        if credentials is None:
            logger.warning("Authentication rejected reason=missing_credentials")
            raise HTTPException(
                status_code=401, detail="Sign in to access saved training data"
            )
        if not config.auth_secret:
            logger.error("Authentication unavailable reason=missing_auth_secret")
            raise HTTPException(
                status_code=503,
                detail="Set REPWISE_AUTH_SECRET before using authentication",
            )
        try:
            return read_access_token(credentials.credentials, config.auth_secret)
        except Exception as error:
            logger.warning(
                "Authentication rejected reason=invalid_or_expired_token"
            )
            raise HTTPException(
                status_code=401, detail="Invalid or expired access token"
            ) from error

    def check_owner(student_id: str, user_id: str) -> None:
        """Require a URL account ID to match the authenticated account.

        Args:
            student_id: Account ID supplied in the route path.
            user_id: Account ID decoded from the authenticated bearer token.

        Raises:
            HTTPException: With status 403 if the IDs do not match.
        """
        if student_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="You can only access your own saved training data",
            )

    @app.post("/api/v1/auth/register", response_model=AccessToken)
    def register(credentials: AccountCredentials) -> AccessToken:
        """Register an account and return its signed-in bearer token.

        Args:
            credentials: New account name and plaintext password from the request.

        Returns:
            An access-token response containing the account ID and full name.

        Raises:
            HTTPException: With status 503 if authentication is not configured,
                or 409 if the full name is already registered.
        """
        if not config.auth_secret:
            raise HTTPException(
                status_code=503, detail="Set REPWISE_AUTH_SECRET before registering"
            )
        try:
            user_id = repository.create_user(
                credentials.full_name, hash_password(credentials.password)
            )
        except sqlite3.IntegrityError as error:
            logger.warning("Account registration rejected reason=duplicate_name")
            raise HTTPException(
                status_code=409, detail="That full name is already registered"
            ) from error
        logger.info("Account registered")
        return AccessToken(
            access_token=create_access_token(user_id, config.auth_secret),
            user_id=user_id,
            full_name=credentials.full_name,
        )

    @app.post("/api/v1/auth/login", response_model=AccessToken)
    def login(credentials: AccountCredentials) -> AccessToken:
        """Verify account credentials and issue a bearer token.

        Args:
            credentials: Account name and password supplied for sign-in.

        Returns:
            An access-token response containing the account ID and full name.

        Raises:
            HTTPException: With status 503 if authentication is not configured,
                or 401 if the account name or password is invalid.
        """
        if not config.auth_secret:
            raise HTTPException(
                status_code=503, detail="Set REPWISE_AUTH_SECRET before signing in"
            )
        account = repository.get_account_by_name(credentials.full_name)
        if not account or not verify_password(credentials.password, account[1]):
            logger.warning("Login rejected reason=invalid_credentials")
            raise HTTPException(status_code=401, detail="Invalid name or password")
        user_id, _, full_name = account
        logger.info("Account signed in")
        return AccessToken(
            access_token=create_access_token(user_id, config.auth_secret),
            user_id=user_id,
            full_name=full_name,
        )

    @app.get("/api/v1/health")
    def health() -> dict[str, object]:
        """Report database, retrieval, and model configuration without generation.

        Returns:
            A status mapping describing the configured retrieval strategy and
            model, including the setup error when retrieval is unavailable.
        """
        result = {
            "status": "ok",
            "database": "ready",
            "retrieval": (
                "ready"
                if not isinstance(retriever, MissingRetriever)
                else "not connected"
            ),
            "retrieval_error": (
                retriever.message if isinstance(retriever, MissingRetriever) else None
            ),
            "retriever": config.retrieval_strategy,
            "model": config.ollama_model,
        }
        logger.info(
            "Health checked database=%s retrieval=%s strategy=%s model=%s",
            result["database"],
            result["retrieval"],
            result["retriever"],
            result["model"],
        )
        return result

    @app.put("/api/v1/students/{student_id}/profile", response_model=UserProfile)
    def save_profile(
        student_id: str, profile: UserProfile, user_id: str = Depends(require_account)
    ) -> UserProfile:
        """Save the authenticated student's supplied profile.

        Args:
            student_id: Account ID from the route path.
            profile: Validated profile to store.
            user_id: Authenticated account ID supplied by the bearer dependency.

        Returns:
            The saved profile.

        Raises:
            HTTPException: With status 403 if the route account is not the caller.
        """
        check_owner(student_id, user_id)
        repository.save_profile(student_id, profile)
        logger.info("Student profile saved")
        return profile

    @app.get("/api/v1/students/{student_id}/profile", response_model=UserProfile)
    def get_profile(
        student_id: str, user_id: str = Depends(require_account)
    ) -> UserProfile:
        """Read the authenticated student's saved profile.

        Args:
            student_id: Account ID from the route path.
            user_id: Authenticated account ID supplied by the bearer dependency.

        Returns:
            The saved validated profile.

        Raises:
            HTTPException: With status 403 for an account mismatch or 404 if no
                profile has been saved.
        """
        check_owner(student_id, user_id)
        profile = repository.get_profile(student_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Student profile not found")
        logger.info("Student profile loaded")
        return profile

    @app.post("/api/v1/students/{student_id}/plans")
    def create_plan(
        student_id: str, options: PlanOptions, user_id: str = Depends(require_account)
    ) -> dict[str, object]:
        """Generate and save a plan using the student's saved profile.

        Args:
            student_id: Account ID from the route path.
            options: Focus and generation options for this plan.
            user_id: Authenticated account ID supplied by the bearer dependency.

        Returns:
            A mapping containing the saved plan ID and completed pipeline result.

        Raises:
            HTTPException: With status 403 for an account mismatch, 404 when no
                profile exists, 422 when the request or generation is invalid,
                or 503 when retrieval fails.
        """
        check_owner(student_id, user_id)
        profile = repository.get_profile(student_id)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail="Save the student profile before creating a plan",
            )
        if (
            options.workouts_per_week is not None
            and options.workouts_per_week > len(profile.available_days)
        ):
            raise HTTPException(
                status_code=422,
                detail="workouts_per_week cannot exceed the days in the saved profile",
            )
        request = PlanRequest(profile=profile, **options.model_dump())
        logger.info(
            "Plan request accepted weeks=%s workouts_per_week=%s",
            request.weeks,
            request.target_workouts_per_week,
        )
        result: PlanResult = pipeline.create_plan(request)
        if result.status == "completed":
            plan_id = repository.save_plan(student_id, request, result)
            logger.info("Plan generated and saved")
            return {"plan_id": plan_id, **result.model_dump()}
        logger.warning("Plan request ended status=%s", result.status)
        status_code = 503 if result.status == "retrieval_error" else 422
        raise HTTPException(status_code=status_code, detail=result.model_dump())

    @app.get("/api/v1/students/{student_id}/plans")
    def list_plans(
        student_id: str,
        limit: int = Query(20, ge=1, le=100),
        user_id: str = Depends(require_account),
    ) -> list[dict[str, object]]:
        """List saved plans belonging to the authenticated student.

        Args:
            student_id: Account ID from the route path.
            limit: Maximum number of recent plans to return (1 through 100).
            user_id: Authenticated account ID supplied by the bearer dependency.

        Returns:
            Recent plan records in descending creation-time order.

        Raises:
            HTTPException: With status 403 if the route account is not the caller.
        """
        check_owner(student_id, user_id)
        plans = repository.list_plans(student_id, limit)
        logger.info("Plan history loaded count=%s", len(plans))
        return plans

    @app.put("/api/v1/students/{student_id}/plans/{plan_id}")
    def update_plan(
        student_id: str,
        plan_id: str,
        plan: TrainingPlan,
        user_id: str = Depends(require_account),
    ) -> dict[str, object]:
        """Replace one saved plan belonging to the authenticated student.

        Args:
            student_id: Account ID from the route path.
            plan_id: ID of the saved plan to replace.
            plan: Validated replacement plan.
            user_id: Authenticated account ID supplied by the bearer dependency.

        Returns:
            A mapping containing the plan ID and updated plan data.

        Raises:
            HTTPException: With status 403 for an account mismatch or 404 if the
                plan does not exist for this student.
        """
        check_owner(student_id, user_id)
        if not repository.update_plan(student_id, plan_id, plan):
            raise HTTPException(status_code=404, detail="Saved plan not found")
        logger.info("Saved plan updated")
        return {"id": plan_id, "plan": plan.model_dump()}

    @app.delete("/api/v1/students/{student_id}/plans/{plan_id}", status_code=204)
    def delete_plan(
        student_id: str,
        plan_id: str,
        user_id: str = Depends(require_account),
    ) -> Response:
        """Delete one saved plan belonging to the authenticated student.

        Args:
            student_id: Account ID from the route path.
            plan_id: ID of the saved plan to delete.
            user_id: Authenticated account ID supplied by the bearer dependency.

        Returns:
            An empty HTTP 204 response after deletion.

        Raises:
            HTTPException: With status 403 for an account mismatch or 404 if the
                plan does not exist for this student.
        """
        check_owner(student_id, user_id)
        if not repository.delete_plan(student_id, plan_id):
            raise HTTPException(status_code=404, detail="Saved plan not found")
        logger.info("Saved plan deleted")
        return Response(status_code=204)

    return app


app = create_app()
