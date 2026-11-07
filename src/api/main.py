"""HTTP routes for the RepWise demo."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import sqlite3

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Path as APIPath,
    Query,
    Request,
    Response,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from config.config import AppConfig
from ..db.repository import StudentRepository
from ..observability import configure_logging, request_id_context
from .auth import create_access_token, hash_password, read_access_token, verify_password
from ..generation.adaptation import (
    WorkoutGenerator,
    WorkoutAdaptationPipeline,
    feedback_mentions_pain,
    next_unlogged_workout,
)
from ..generation.pipeline import PlanPipeline, validate_plan
from ..llm.ollama import OllamaGenerator, PlanGenerator
from ..retrieval.base import MissingRetriever, Retriever
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.query_expansion import (
    OllamaQueryExpander,
    QueryExpansionRetriever,
    ScopeAnalyzer,
)
from ..retrieval.semantic import SemanticRetriever, load_chunks
from ..retrieval.sparse import BM25Retriever
from ..schemas.models import (
    AccessToken,
    AccountIdentity,
    AccountCredentials,
    PlanOptions,
    PlanRequest,
    TrainingPlan,
    UserProfile,
    WorkoutFeedbackRequest,
)
from ..speech import (
    MAX_AUDIO_BYTES,
    SpeechTranscriber,
    WhisperTranscriber,
    validate_recording,
)
from ..tools.generate_workout_plan import generate_workout_plan

logger = logging.getLogger(__name__)


def create_app(
    config: AppConfig | None = None,
    *,
    repository: StudentRepository | None = None,
    retriever: Retriever | None = None,
    scope_analyzer: ScopeAnalyzer | None = None,
    generator: PlanGenerator | None = None,
    workout_generator: WorkoutGenerator | None = None,
    transcriber: SpeechTranscriber | None = None,
) -> FastAPI:
    """Build the API, persistence layer, retriever, generator, and routes.

    Args:
        config: Application settings, or ``None`` to load them from the environment.
        repository: Optional repository implementation; it is initialized before
            the app is returned.
        retriever: Optional retrieval service. If omitted, the configured corpus
            and retrieval strategy are used.
        scope_analyzer: Optional request-scope analyzer. If omitted, the
            configured local Ollama model is used.
        generator: Optional plan-generation service; if omitted, a local Ollama
            generator is configured from the settings.
        workout_generator: Optional structured workout generator. The default
            Ollama generator is reused when it is also the plan generator.
        transcriber: Optional speech-to-text service; defaults to lazy local Whisper.

    Returns:
        A configured FastAPI application. If default retriever initialization
        fails, the app is still created with a ``MissingRetriever`` so health and
        documentation routes remain available.
    """
    config = config or AppConfig.from_environment()
    transcriber = (
        transcriber
        if transcriber is not None
        else WhisperTranscriber(config.speech_model)
    )
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
    if scope_analyzer is None:
        scope_analyzer = OllamaQueryExpander(
            config.query_expansion_model, config.ollama_url
        )
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
    if not isinstance(retriever, QueryExpansionRetriever):
        retriever = QueryExpansionRetriever(
            retriever, scope_analyzer, rewrite=config.query_expansion
        )
    logger.info(
        "Retriever scope gate enabled query_rewriting=%s model=%s",
        config.query_expansion,
        config.query_expansion_model,
    )
    generator = generator or OllamaGenerator(config.ollama_model, config.ollama_url)
    if not isinstance(generator, OllamaGenerator):
        logger.info("Using injected plan generator service")
    if workout_generator is None and isinstance(generator, OllamaGenerator):
        workout_generator = generator
    workout_adapter = (
        WorkoutAdaptationPipeline(workout_generator)
        if workout_generator is not None
        else None
    )
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
                (
                    request.scope.get("route").path
                    if request.scope.get("route")
                    else "unmatched"
                ),
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
            logger.warning("Authentication rejected reason=invalid_or_expired_token")
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

    @app.post("/api/v1/speech/transcribe")
    async def transcribe_recording(
        request: Request, user_id: str = Depends(require_account)
    ) -> dict[str, str]:
        """Transcribe a signed-in user's WAV recording without saving or generating a plan."""
        media_type = (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        )
        if media_type not in {"audio/wav", "audio/x-wav", "audio/wave"}:
            raise HTTPException(
                status_code=415, detail="Send a WAV microphone recording."
            )
        audio = bytearray()
        async for block in request.stream():
            if len(audio) + len(block) > MAX_AUDIO_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="The recording is too large. Record up to two minutes.",
                )
            audio.extend(block)
        try:
            recording = bytes(audio)
            validate_recording(recording)
            text = await run_in_threadpool(transcriber.transcribe, recording)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "No speech was detected. Please record your request again."
                )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {"text": text.strip()}

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

    @app.get("/api/v1/auth/me", response_model=AccountIdentity)
    def current_account(user_id: str = Depends(require_account)) -> AccountIdentity:
        """Verify the bearer token and return the signed-in account's identity."""
        account = repository.get_account_identity(user_id)
        if account is None:
            raise HTTPException(
                status_code=401, detail="Account is no longer available"
            )
        return AccountIdentity.model_validate(account)

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
        result = generate_workout_plan(
            options,
            student_id=student_id,
            repository=repository,
            pipeline=pipeline,
        )
        if result.status == "missing_profile":
            raise HTTPException(status_code=404, detail=result.message)
        if result.status == "invalid_request":
            raise HTTPException(status_code=422, detail=result.message)
        payload = result.model_dump(exclude={"plan_id"})
        if result.status == "completed":
            logger.info("Plan generated and saved")
            return {"plan_id": result.plan_id, **payload}
        if result.status == "out_of_scope":
            logger.info("Plan request declined by scope gate")
            return payload
        logger.warning("Plan request ended status=%s", result.status)
        status_code = 503 if result.status == "retrieval_error" else 422
        raise HTTPException(status_code=status_code, detail=payload)

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

    @app.post(
        "/api/v1/students/{student_id}/plans/{plan_id}/weeks/{week_number}"
        "/workouts/{workout_index}/feedback"
    )
    def log_workout(
        student_id: str,
        plan_id: str,
        week_number: int = APIPath(ge=1),
        workout_index: int = APIPath(ge=0),
        feedback_request: WorkoutFeedbackRequest = ...,
        user_id: str = Depends(require_account),
    ) -> dict[str, object]:
        """Save workout feedback and adapt only the next unlogged workout."""
        check_owner(student_id, user_id)
        saved = repository.get_plan(student_id, plan_id)
        if saved is None:
            raise HTTPException(status_code=404, detail="Saved plan not found")

        request = PlanRequest.model_validate(saved["request"])
        plan = TrainingPlan.model_validate(saved["plan"])
        week = next((item for item in plan.weeks if item.number == week_number), None)
        if week is None or workout_index >= len(week.workouts):
            raise HTTPException(
                status_code=404, detail="Workout not found in saved plan"
            )
        logged_workout = week.workouts[workout_index]

        existing_feedback = repository.list_workout_feedback(student_id, plan_id)
        prior_log = next(
            (
                item
                for item in existing_feedback
                if item.week_number == week_number
                and item.workout_index == workout_index
            ),
            None,
        )
        feedback = repository.save_workout_feedback(
            student_id,
            plan_id,
            week_number,
            workout_index,
            feedback_request,
        )
        updated_plan = plan
        adapted_workout: dict[str, int] | None = None
        adapted = False
        if prior_log and prior_log.adapted_week_number is not None:
            message = "Feedback updated. Its previously adapted workout was not changed again."
        elif feedback_mentions_pain(feedback_request):
            message = (
                "Feedback saved. No workout was changed because the comment "
                "mentions pain or discomfort; review the plan's safety note."
            )
        elif workout_adapter is None:
            message = (
                "Feedback saved, but workout adaptation is unavailable for the "
                "configured generator."
            )
        else:
            logged_workouts = {
                (item.week_number, item.workout_index) for item in existing_feedback
            }
            logged_workouts.add((week_number, workout_index))
            target = next_unlogged_workout(
                plan,
                after_week_number=week_number,
                after_workout_index=workout_index,
                logged_workouts=logged_workouts,
            )
            if target is None:
                message = "Feedback saved. There is no later unlogged workout in this plan to adapt."
            else:
                target_week_number, target_index, target_workout = target
                try:
                    replacement = workout_adapter.adapt(
                        request=request,
                        plan=plan,
                        logged_workout=logged_workout,
                        target_workout=target_workout,
                        target_week_number=target_week_number,
                        feedback=feedback_request,
                    )
                    if replacement == target_workout:
                        message = (
                            "Feedback saved. The next unlogged workout was reviewed "
                            "and did not need changes."
                        )
                    else:
                        updated_weeks = [
                            (
                                item.model_copy(
                                    update={
                                        "workouts": [
                                            (
                                                replacement
                                                if index == target_index
                                                else workout
                                            )
                                            for index, workout in enumerate(
                                                item.workouts
                                            )
                                        ]
                                    }
                                )
                                if item.number == target_week_number
                                else item
                            )
                            for item in plan.weeks
                        ]
                        updated_plan = plan.model_copy(update={"weeks": updated_weeks})
                        source_ids = {
                            source["source_id"] for source in saved["sources"]
                        }
                        validate_plan(updated_plan, request, source_ids)
                        if not repository.update_plan_after_feedback(
                            student_id,
                            plan_id,
                            updated_plan,
                            week_number,
                            workout_index,
                            target_week_number,
                            target_index,
                        ):
                            raise HTTPException(
                                status_code=404, detail="Saved plan not found"
                            )
                        adapted = True
                        adapted_workout = {
                            "week_number": target_week_number,
                            "workout_index": target_index,
                        }
                        message = (
                            "Feedback saved and the next unlogged workout was adjusted."
                        )
                        logger.info(
                            "Workout adapted plan_id=%s from_week=%s from_index=%s "
                            "to_week=%s to_index=%s",
                            plan_id,
                            week_number,
                            workout_index,
                            target_week_number,
                            target_index,
                        )
                except (OSError, RuntimeError, ValueError) as error:
                    logger.exception(
                        "Workout adaptation failed plan_id=%s week=%s index=%s",
                        plan_id,
                        week_number,
                        workout_index,
                    )
                    updated_plan = plan
                    message = (
                        "Feedback was saved, but the next workout was left "
                        f"unchanged because adaptation failed: {error}"
                    )

        feedback_records = repository.list_workout_feedback(student_id, plan_id)
        feedback = next(
            (
                item
                for item in feedback_records
                if item.week_number == week_number
                and item.workout_index == workout_index
            ),
            feedback,
        )
        return {
            "plan_id": plan_id,
            "feedback": feedback.model_dump(),
            "workout_feedback": [item.model_dump() for item in feedback_records],
            "plan": updated_plan.model_dump(),
            "adapted": adapted,
            "adapted_workout": adapted_workout,
            "message": message,
        }

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
