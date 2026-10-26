"""HTTP routes for the RepWise demo."""

from __future__ import annotations

from pathlib import Path

import sqlite3

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.config import AppConfig
from ..db.repository import StudentRepository
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


def create_app(
    config: AppConfig | None = None,
    *,
    repository: StudentRepository | None = None,
    retriever: Retriever | None = None,
    generator: PlanGenerator | None = None,
) -> FastAPI:
    """Create the application and allow fake services to be injected in tests."""
    config = config or AppConfig.from_environment()
    repository = repository or StudentRepository(config.database_path)
    repository.initialize()
    if retriever is None:
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
        except (OSError, RuntimeError, ValueError) as error:
            # Keep health and API docs available, but preserve the real setup error.
            retriever = MissingRetriever(str(error))
    generator = generator or OllamaGenerator(config.ollama_model, config.ollama_url)
    pipeline = PlanPipeline(retriever, generator, config.top_k)

    app = FastAPI(
        title="RepWise",
        version="0.1.0",
        description="Course demo API for personalised workout plans.",
    )
    bearer = HTTPBearer(auto_error=False)

    def require_account(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> str:
        """Authenticate every route that reads or writes student-specific data."""
        if credentials is None:
            raise HTTPException(
                status_code=401, detail="Sign in to access saved training data"
            )
        if not config.auth_secret:
            raise HTTPException(
                status_code=503,
                detail="Set REPWISE_AUTH_SECRET before using authentication",
            )
        try:
            return read_access_token(credentials.credentials, config.auth_secret)
        except Exception as error:
            raise HTTPException(
                status_code=401, detail="Invalid or expired access token"
            ) from error

    def check_owner(student_id: str, username: str) -> None:
        """Bind the URL identity to the signed-in account, never to client claims."""
        if student_id != username:
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
            repository.create_user(
                credentials.username, hash_password(credentials.password)
            )
        except sqlite3.IntegrityError as error:
            raise HTTPException(
                status_code=409, detail="That username is already registered"
            ) from error
        return AccessToken(
            access_token=create_access_token(credentials.username, config.auth_secret)
        )

    @app.post("/api/v1/auth/login", response_model=AccessToken)
    def login(credentials: AccountCredentials) -> AccessToken:
        """Verify a password hash and return a short-lived signed bearer token."""
        if not config.auth_secret:
            raise HTTPException(
                status_code=503, detail="Set REPWISE_AUTH_SECRET before signing in"
            )
        saved_hash = repository.get_password_hash(credentials.username)
        if not saved_hash or not verify_password(credentials.password, saved_hash):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        return AccessToken(
            access_token=create_access_token(credentials.username, config.auth_secret)
        )

    @app.get("/api/v1/health")
    def health() -> dict[str, object]:
        """Report local component configuration without making an LLM call."""
        return {
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
            "model": config.ollama_model,
        }

    @app.put("/api/v1/students/{student_id}/profile", response_model=UserProfile)
    def save_profile(
        student_id: str, profile: UserProfile, username: str = Depends(require_account)
    ) -> UserProfile:
        """Save the student details explicitly provided for future plans."""
        check_owner(student_id, username)
        repository.save_profile(student_id, profile)
        return profile

    @app.get("/api/v1/students/{student_id}/profile", response_model=UserProfile)
    def get_profile(
        student_id: str, username: str = Depends(require_account)
    ) -> UserProfile:
        """Read one saved profile."""
        check_owner(student_id, username)
        profile = repository.get_profile(student_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Student profile not found")
        return profile

    @app.post("/api/v1/students/{student_id}/plans")
    def create_plan(
        student_id: str, options: PlanOptions, username: str = Depends(require_account)
    ) -> dict[str, object]:
        """Create and save a plan from a stored profile and current request."""
        check_owner(student_id, username)
        profile = repository.get_profile(student_id)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail="Save the student profile before creating a plan",
            )
        request = PlanRequest(profile=profile, **options.model_dump())
        result: PlanResult = pipeline.create_plan(request)
        if result.status == "completed":
            plan_id = repository.save_plan(student_id, options, result)
            return {"plan_id": plan_id, **result.model_dump()}
        status_code = 503 if result.status == "retrieval_error" else 422
        raise HTTPException(status_code=status_code, detail=result.model_dump())

    @app.get("/api/v1/students/{student_id}/plans")
    def list_plans(
        student_id: str,
        limit: int = Query(20, ge=1, le=100),
        username: str = Depends(require_account),
    ) -> list[dict[str, object]]:
        """List plans previously created for one student."""
        check_owner(student_id, username)
        return repository.list_plans(student_id, limit)

    @app.put("/api/v1/students/{student_id}/plans/{plan_id}")
    def update_plan(
        student_id: str,
        plan_id: str,
        plan: TrainingPlan,
        username: str = Depends(require_account),
    ) -> dict[str, object]:
        """Replace one of the signed-in student's saved plans."""
        check_owner(student_id, username)
        if not repository.update_plan(student_id, plan_id, plan):
            raise HTTPException(status_code=404, detail="Saved plan not found")
        return {"id": plan_id, "plan": plan.model_dump()}

    @app.delete("/api/v1/students/{student_id}/plans/{plan_id}", status_code=204)
    def delete_plan(
        student_id: str,
        plan_id: str,
        username: str = Depends(require_account),
    ) -> Response:
        """Delete one of the signed-in student's saved plans."""
        check_owner(student_id, username)
        if not repository.delete_plan(student_id, plan_id):
            raise HTTPException(status_code=404, detail="Saved plan not found")
        return Response(status_code=204)

    return app


app = create_app()
