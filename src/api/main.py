"""HTTP routes for the RepWise demo."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from config.config import AppConfig
from ..db.repository import StudentRepository
from ..generation.pipeline import PlanPipeline
from ..llm.ollama import OllamaGenerator, PlanGenerator
from ..retrieval.base import MissingRetriever, Retriever
from ..retrieval.jsonl import JsonlRetriever
from ..schemas.models import PlanOptions, PlanRequest, PlanResult, UserProfile


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
    retriever = retriever or (
        JsonlRetriever(config.chunk_file) if config.chunk_file else MissingRetriever()
    )
    generator = generator or OllamaGenerator(config.ollama_model, config.ollama_url)
    pipeline = PlanPipeline(retriever, generator, config.top_k)

    app = FastAPI(
        title="RepWise",
        version="0.1.0",
        description="Course demo API for personalised workout plans.",
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
            "model": config.ollama_model,
        }

    @app.put("/api/v1/students/{student_id}/profile", response_model=UserProfile)
    def save_profile(student_id: str, profile: UserProfile) -> UserProfile:
        """Save the student details explicitly provided for future plans."""
        repository.save_profile(student_id, profile)
        return profile

    @app.get("/api/v1/students/{student_id}/profile", response_model=UserProfile)
    def get_profile(student_id: str) -> UserProfile:
        """Read one saved profile."""
        profile = repository.get_profile(student_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Student profile not found")
        return profile

    @app.post("/api/v1/students/{student_id}/plans")
    def create_plan(student_id: str, options: PlanOptions) -> dict[str, object]:
        """Create and save a plan from a stored profile and current request."""
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
        student_id: str, limit: int = Query(20, ge=1, le=100)
    ) -> list[dict[str, object]]:
        """List plans previously created for one student."""
        return repository.list_plans(student_id, limit)

    return app


app = create_app()
