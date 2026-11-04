"""Generate plans through the existing RAG pipeline, with account context."""

from pydantic import ValidationError

from ...db.repository import StudentRepository
from ...generation.pipeline import PlanPipeline
from ...schemas.models import PlanOptions, PlanRequest
from .schemas import GenerateWorkoutPlanResult


def generate_workout_plan(
    options: PlanOptions,
    *,
    student_id: str,
    repository: StudentRepository,
    pipeline: PlanPipeline,
) -> GenerateWorkoutPlanResult:
    """Load the saved profile, generate one week, and save only successful plans.

    ``options`` are the model-supplied tool arguments, validated as PlanOptions.
    The caller must authenticate and authorize ``student_id`` and supply the
    repository and configured pipeline from trusted application context.
    No models, indexes or database connections are initialized by this tool.

    Missing profiles, impossible schedules, scope refusals and pipeline failures
    return structured outcomes. Database failures propagate to the caller;
    they must not be reported as successful saves.
    """
    profile = repository.get_profile(student_id)
    if profile is None:
        return GenerateWorkoutPlanResult(
            status="missing_profile",
            message="Save the student profile before creating a plan",
        )
    try:
        request = PlanRequest(profile=profile, **options.model_dump())
    except ValidationError as error:
        return GenerateWorkoutPlanResult(
            status="invalid_request",
            message="; ".join(item["msg"] for item in error.errors()),
        )

    previous_plan = repository.get_latest_plan(student_id)
    result = pipeline.create_plan(request, previous_plan=previous_plan)
    plan_id = None
    if result.status == "completed":
        plan_id = repository.save_plan(student_id, request, result)
    return GenerateWorkoutPlanResult(**result.model_dump(), plan_id=plan_id)
