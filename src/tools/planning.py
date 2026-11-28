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
from datetime import date
from typing import Literal

from pydantic import ValidationError, model_validator

from src.db.repository import StudentRepository
from src.generation.pipeline import PlanPipeline
from src.schemas.models import (
    AgentPlanOptions,
    Model,
    PlanOptions,
    PlanRequest,
    PlanResult,
    PlanStatus,
)
from src.tools import calendar as calendar_tools
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
        schedule_week_start: Monday of the constrained week, when requested.
        schedule_note: Calendar and availability facts used for the schedule.
    """

    plan_id: str | None = None
    plan_status: PlanStatus | None = None
    schedule_week_start: date | None = None
    schedule_note: str | None = None

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
    plan_status: Literal["active", "draft"] = "active"
    activated: bool = True
    calendar_status: Literal[
        "scheduled", "partial", "no_slots", "not_connected", "unavailable", "failed"
    ] = "not_connected"
    calendar_workouts: int = 0
    calendar_message: str | None = None
    calendar_result: calendar_tools.BookingResult | None = None


def generate_workout_plan(
    options: PlanOptions,
    *,
    student_id: str,
    repository: StudentRepository,
    pipeline: PlanPipeline,
    status: Literal["active", "draft"] = "active",
    available_days: list[str] | None = None,
    schedule_week_start: date | None = None,
    schedule_note: str | None = None,
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
        available_days: Trusted date-specific weekdays to constrain this plan to.
        schedule_week_start: Monday of the specific calendar week, when used.
        schedule_note: Availability facts to return to the planning agent.

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
    if options.max_session_minutes is not None:
        profile = profile.model_copy(
            update={
                "session_minutes": min(
                    profile.session_minutes, options.max_session_minutes
                )
            }
        )
    if available_days is not None:
        profile = profile.model_copy(update={"available_days": available_days})
    try:
        request = PlanRequest(
            profile=profile,
            schedule_week_start=schedule_week_start,
            **options.model_dump(),
        )
    except ValidationError as error:
        return GenerateWorkoutPlanResult(
            status="invalid_request",
            message="; ".join(item["msg"] for item in error.errors()),
        )

    previous_plan = repository.get_latest_plan(student_id)
    result = pipeline.create_plan(request, previous_plan=previous_plan)
    if result.status != "completed":
        return GenerateWorkoutPlanResult(
            **result.model_dump(),
            schedule_week_start=schedule_week_start,
            schedule_note=schedule_note,
        )
    plan_id = repository.save_plan(student_id, request, result, status=status)
    return GenerateWorkoutPlanResult(
        **result.model_dump(),
        plan_id=plan_id,
        plan_status=status,
        schedule_week_start=schedule_week_start,
        schedule_note=schedule_note,
    )


def generate_plan_for_agent(
    context: ToolContext, **arguments: object
) -> GenerateWorkoutPlanResult:
    """Generate a plan for the agent, as a draft when it would replace one.

    Args:
        context: Trusted tool context.
        **arguments: :class:`AgentPlanOptions` fields supplied by the model.

    Returns:
        The generation outcome; ``plan_status`` says whether the new plan is
        already active or waits for approval. An unavailable pipeline or
        database returns ``status="unavailable"``.
    """
    try:
        agent_options = AgentPlanOptions.model_validate(arguments)
        options = PlanOptions.model_validate(
            agent_options.model_dump(exclude={"schedule_scope"})
        )
        has_active_plan = context.repository.get_current_plan(context.student_id)
        available_days = None
        schedule_week_start = None
        schedule_note = None
        is_date_scoped = agent_options.schedule_scope == "this_week"
        if is_date_scoped:
            profile = context.repository.get_profile(context.student_id)
            if profile is not None:
                session_minutes = min(
                    profile.session_minutes,
                    options.max_session_minutes or profile.session_minutes,
                )
                availability = calendar_tools.available_days_this_week(
                    context, profile.available_days, session_minutes
                )
                if isinstance(availability, ToolFailure):
                    return GenerateWorkoutPlanResult(
                        status=availability.code,
                        message=(
                            "Calendar availability could not be checked, so a "
                            "date-specific plan was not generated. "
                            f"{availability.error}"
                        ),
                        schedule_note=availability.error,
                    )
                if not availability.available_days:
                    empty_schedule_message = (
                        "No future profile weekday has enough free time in the "
                        "current calendar week."
                    )
                    return GenerateWorkoutPlanResult(
                        status="no_available_days",
                        message=empty_schedule_message,
                        schedule_week_start=availability.week_start,
                        schedule_note=empty_schedule_message,
                    )

                requested_count = options.workouts_per_week or min(
                    3, len(profile.available_days)
                )
                effective_count = min(requested_count, len(availability.available_days))
                calendar_detail = (
                    "Connected calendar was checked; availability is not booked"
                )
                reduction_detail = (
                    f" Requested {requested_count} workouts were reduced to "
                    f"{effective_count}."
                    if effective_count < requested_count
                    else ""
                )
                schedule_note = (
                    f"{calendar_detail} for the week starting "
                    f"{availability.week_start.isoformat()}. Available profile days: "
                    f"{', '.join(availability.available_days)}."
                    f"{reduction_detail}"
                )
                available_days = availability.available_days
                schedule_week_start = availability.week_start
                options = options.model_copy(
                    update={"workouts_per_week": effective_count}
                )
        return generate_workout_plan(
            options,
            student_id=context.student_id,
            repository=context.repository,
            pipeline=context.pipeline(),
            status="draft" if has_active_plan or is_date_scoped else "active",
            available_days=available_days,
            schedule_week_start=schedule_week_start,
            schedule_note=schedule_note,
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
    context: ToolContext,
    plan_id: str,
    bookings: list[calendar_tools.Booking] | None = None,
    slot_choices: list[calendar_tools.SlotChoice] | None = None,
    calendar_status: Literal[
        "scheduled", "no_slots", "not_connected", "unavailable"
    ] = "not_connected",
    calendar_message: str | None = None,
) -> ActivatePlanResult | ToolFailure:
    """Make a draft plan the student's active plan, archiving the previous one.

    This tool always requires the student's approval. Prepared calendar
    bookings are pinned into the approved arguments, then rechecked at execution.

    Args:
        context: Trusted tool context.
        plan_id: The draft plan to activate.
        bookings: Exact times stored in the activation approval.
        calendar_status: Availability outcome prepared before approval.
        calendar_message: Why calendar events cannot be added, if applicable.

    Returns:
        The activated plan and calendar outcome, or a structured failure.
    """
    record = context.repository.get_plan(context.student_id, plan_id)
    if record is None:
        return failure(PLAN_NOT_FOUND, "Saved plan not found.")
    schedule_week_start = record["request"].get("schedule_week_start")
    if schedule_week_start:
        if context.calendar is None or calendar_status != "scheduled" or not bookings:
            return failure(
                "no_workouts_to_book",
                "This week's plan cannot be activated without approved calendar slots.",
            )
        result = calendar_tools.add_workouts_to_calendar(context, bookings, plan_id)
        if isinstance(result, ToolFailure):
            return result
        calendar_workouts = sum(
            outcome.status in {"booked", "already_booked"}
            for outcome in result.outcomes
        )
        if calendar_workouts != len(result.outcomes):
            return ActivatePlanResult(
                plan_id=plan_id,
                plan_status="draft",
                activated=False,
                calendar_status="partial",
                calendar_workouts=calendar_workouts,
                calendar_message=(
                    "One or more approved slots became unavailable. The plan "
                    "remains a draft; see each booking outcome."
                ),
                calendar_result=result,
            )
        if not context.repository.activate_plan(context.student_id, plan_id):
            return failure(PLAN_NOT_FOUND, "Saved plan could not be activated.")
        return ActivatePlanResult(
            plan_id=plan_id,
            calendar_status="scheduled",
            calendar_workouts=calendar_workouts,
            calendar_result=result,
        )

    if not context.repository.activate_plan(context.student_id, plan_id):
        return failure(PLAN_NOT_FOUND, "Saved plan not found.")

    calendar_workouts = 0
    calendar_result = None
    if calendar_status == "scheduled":
        if context.calendar is None:
            calendar_status = "not_connected"
            calendar_message = "No calendar is connected."
        elif not bookings:
            return failure(
                "invalid_request",
                "The approved activation action has no prepared calendar bookings.",
            )
        else:
            result = calendar_tools.add_workouts_to_calendar(context, bookings, plan_id)
            if isinstance(result, ToolFailure):
                calendar_status = "failed"
                calendar_message = result.error
            else:
                calendar_result = result
                calendar_workouts = sum(
                    outcome.status in {"booked", "already_booked"}
                    for outcome in result.outcomes
                )
                calendar_status = (
                    "scheduled"
                    if calendar_workouts == len(result.outcomes)
                    else "partial"
                )
                if calendar_status == "partial":
                    calendar_message = (
                        "Some events could not be added; see each booking outcome."
                    )

    return ActivatePlanResult(
        plan_id=plan_id,
        calendar_status=calendar_status,
        calendar_workouts=calendar_workouts,
        calendar_message=calendar_message,
        calendar_result=calendar_result,
    )
