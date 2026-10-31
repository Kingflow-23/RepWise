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
    """Create the application and allow fake services to be injected in tests."""
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
        """Authenticate every route that reads or writes student-specific data."""
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
        """Bind the URL identity to the signed-in account, never to client claims."""
        if student_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="You can only access your own saved training data",
            )

    @app.post("/api/v1/auth/register", response_model=AccessToken)
    def register(credentials: AccountCredentials) -> AccessToken:
        """Create an account with an Argon2 password hash and sign it in."""
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
        """Verify a password hash and return a short-lived signed bearer token."""
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
        """Report local component configuration without making an LLM call."""
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
        """Save the student details explicitly provided for future plans."""
        check_owner(student_id, user_id)
        repository.save_profile(student_id, profile)
        logger.info("Student profile saved")
        return profile

    @app.get("/api/v1/students/{student_id}/profile", response_model=UserProfile)
    def get_profile(
        student_id: str, user_id: str = Depends(require_account)
    ) -> UserProfile:
        """Read one saved profile."""
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
        """Create and save a plan from a stored profile and current request."""
        check_owner(student_id, user_id)
        profile = repository.get_profile(student_id)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail="Save the student profile before creating a plan",
            )
        request = PlanRequest(profile=profile, **options.model_dump())
        logger.info("Plan request accepted weeks=%s", request.weeks)
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
        """List plans previously created for one student."""
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
        """Replace one of the signed-in student's saved plans."""
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
        """Delete one of the signed-in student's saved plans."""
        check_owner(student_id, user_id)
        if not repository.delete_plan(student_id, plan_id):
            raise HTTPException(status_code=404, detail="Saved plan not found")
        logger.info("Saved plan deleted")
        return Response(status_code=204)

    return app


app = create_app()
