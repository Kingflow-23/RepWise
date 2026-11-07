"""Plan generation and activation tools (Workout Planning Agent).

``generate_workout_plan`` reuses the existing RAG pipeline: it retrieves fitness
evidence, asks the model for a one-week plan, validates it against the profile
and the evidence, and saves it. The API's plan route calls the same function.

A plan produced for the agent is saved as a **draft** whenever the student
already has an active plan, because replacing the plan they follow needs their
approval. ``activate_plan`` (always approval-gated) turns the draft into the
active plan. A student's very first plan replaces nothing, so it is saved as
active straight away.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Literal

from pydantic import ValidationError, model_validator

from src.db.repository import StudentRepository
from src.generation.pipeline import PlanPipeline
from src.schemas.models import Model, PlanOptions, PlanRequest, PlanResult, PlanStatus
from src.tools.context import ToolContext
from src.tools.results import (
    PLAN_NOT_FOUND,
    ToolFailure,
    failure,
    report_storage_errors,
)

logger = logging.getLogger(__name__)


class GenerateWorkoutPlanResult(PlanResult):
    """A plan-generation outcome; a saved plan always carries its ID and status.

    Attributes:
        plan_id: ID of the saved plan, only when ``status`` is ``completed``.
        plan_status: ``active`` or ``draft`` for a saved plan.
    """

    plan_id: str | None = None
    plan_status: PlanStatus | None = None

    @model_validator(mode="after")
    def saved_plan_matches_status(self) -> "GenerateWorkoutPlanResult":
        """Reject a completed result without a saved ID, or the reverse.

        Returns:
            The validated result.

        Raises:
            ValueError: If ``status`` and ``plan_id`` disagree.
        """
        if self.status == "completed" and not self.plan_id:
            raise ValueError("A completed generation must include its saved plan ID")
        if self.status != "completed" and self.plan_id is not None:
            raise ValueError("An unsuccessful generation cannot have a saved plan ID")
        return self


class ActivatePlanResult(Model):
    """Output of ``activate_plan``."""

    ok: Literal[True] = True
    plan_id: str
    plan_status: Literal["active"] = "active"


def generate_workout_plan(
    options: PlanOptions,
    *,
    student_id: str,
    repository: StudentRepository,
    pipeline: PlanPipeline,
    status: Literal["active", "draft"] = "active",
) -> GenerateWorkoutPlanResult:
    """Load the saved profile, generate one week, and save only successful plans.

    The caller must authenticate ``student_id`` and supply the repository and
    the configured pipeline from trusted application context. No models,
    indexes or database connections are initialized here.

    Args:
        options: Focus and options, validated as :class:`PlanOptions`.
        student_id: Authenticated account that will own the plan.
        repository: Database access.
        pipeline: The configured RAG generation pipeline.
        status: How to save a completed plan: ``active`` (the API default) or
            ``draft`` (waiting for approval).

    Returns:
        A structured outcome. Missing profiles, impossible schedules, scope
        refusals and pipeline failures are statuses, not exceptions.

    Raises:
        sqlite3.Error: If saving fails; that must never look like a saved plan.
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
    if result.status != "completed":
        return GenerateWorkoutPlanResult(**result.model_dump())
    plan_id = repository.save_plan(student_id, request, result, status=status)
    return GenerateWorkoutPlanResult(
        **result.model_dump(), plan_id=plan_id, plan_status=status
    )


def generate_plan_for_agent(
    context: ToolContext, **arguments: object
) -> GenerateWorkoutPlanResult:
    """Generate a plan for the agent, as a draft when it would replace one.

    Args:
        context: Trusted tool context.
        **arguments: :class:`PlanOptions` fields supplied by the model.

    Returns:
        The generation outcome; ``plan_status`` says whether the new plan is
        already active or waits for approval. An unavailable pipeline or
        database returns ``status="unavailable"``.
    """
    try:
        options = PlanOptions.model_validate(arguments)
        has_active_plan = context.repository.get_current_plan(context.student_id)
        return generate_workout_plan(
            options,
            student_id=context.student_id,
            repository=context.repository,
            pipeline=context.pipeline(),
            status="draft" if has_active_plan else "active",
        )
    except ValidationError as error:
        return GenerateWorkoutPlanResult(
            status="invalid_request",
            message="; ".join(item["msg"] for item in error.errors()),
        )
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        logger.exception("Agent plan generation failed")
        return GenerateWorkoutPlanResult(
            status="unavailable",
            message=f"The plan could not be generated or saved: {error}",
        )


@report_storage_errors
def activate_plan(
    context: ToolContext, plan_id: str
) -> ActivatePlanResult | ToolFailure:
    """Make a draft plan the student's active plan, archiving the previous one.

    This tool always requires the student's approval (see ``config/tools.json``).

    Args:
        context: Trusted tool context.
        plan_id: The draft plan to activate.

    Returns:
        The activated plan ID, or ``plan_not_found``.
    """
    if not context.repository.activate_plan(context.student_id, plan_id):
        return failure(PLAN_NOT_FOUND, "Saved plan not found.")
    return ActivatePlanResult(plan_id=plan_id)
