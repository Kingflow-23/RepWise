"""Read tools for the student's profile and current plan (Orchestrator Agent).

None of these tools write anything. They give the orchestrator, once per request,
the context it shares with the specialists, so specialists do not each re-read it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.schemas.models import (
    Model,
    PlanStatus,
    TrainingPlan,
    UserProfile,
    Workout,
    WorkoutFeedback,
)
from src.tools.context import ToolContext
from src.tools.results import (
    PLAN_NOT_FOUND,
    STUDENT_NOT_FOUND,
    WORKOUT_NOT_FOUND,
    ToolFailure,
    failure,
    report_storage_errors,
)


class ProfileResult(Model):
    """Output of ``get_user_profile``."""

    ok: Literal[True] = True
    profile: UserProfile


class PlanSource(Model):
    """One evidence source a plan cites."""

    source_id: str
    title: str
    url: str


class CurrentPlanResult(Model):
    """Output of ``get_current_plan``."""

    ok: Literal[True] = True
    plan_id: str
    status: PlanStatus
    created_at: str
    plan: TrainingPlan
    sources: list[PlanSource] = Field(default_factory=list)


class CalendarBooking(Model):
    """A calendar event RepWise created for one workout."""

    event_id: str
    start: str
    end: str


class PlannedWorkout(Workout):
    """A scheduled workout, where it sits in the plan, and what is saved about it."""

    plan_id: str
    week_number: int = Field(ge=1)
    workout_index: int = Field(ge=0)
    feedback: WorkoutFeedback | None = None
    booking: CalendarBooking | None = None


class TimetableResult(Model):
    """Output of ``get_timetable``."""

    ok: Literal[True] = True
    available_days: list[str]
    session_minutes: int
    plan_id: str | None = None
    planned_workouts: list[PlannedWorkout]


class PerformanceRecord(Model):
    """What the student actually did for one exercise."""

    exercise_name: str = Field(min_length=1, max_length=100)
    sets_completed: int = Field(ge=0, le=50)
    reps_per_set: str = Field(min_length=1, max_length=100)
    weight_kg: float | None = Field(default=None, ge=0, le=1000)


class WorkoutResult(Model):
    """Output of ``get_workout``."""

    ok: Literal[True] = True
    workout: PlannedWorkout
    performance: list[PerformanceRecord] = Field(default_factory=list)


def find_workout(
    plan: TrainingPlan, week_number: int, workout_index: int
) -> Workout | None:
    """Find one workout in a plan.

    Args:
        plan: The plan to search.
        week_number: Week number, starting at 1.
        workout_index: Zero-based position in the week.

    Returns:
        The workout, or ``None`` if the plan has no such workout.
    """
    week = next((item for item in plan.weeks if item.number == week_number), None)
    if week is None or workout_index not in range(len(week.workouts)):
        return None
    return week.workouts[workout_index]


def resolve_plan(context: ToolContext, plan_id: str | None) -> dict | ToolFailure:
    """Load a plan the student owns, or their current plan when no ID is given.

    Args:
        context: Trusted tool context.
        plan_id: A plan ID, or ``None`` for the current active plan.

    Returns:
        The repository's plan record, or a ``plan_not_found`` failure.
    """
    if plan_id is None:
        record = context.repository.get_current_plan(context.student_id)
        if record is None:
            return failure(PLAN_NOT_FOUND, "There is no active plan yet.")
        return record
    record = context.repository.get_plan(context.student_id, plan_id)
    if record is None:
        return failure(PLAN_NOT_FOUND, "Saved plan not found.")
    return record


def planned_workouts(context: ToolContext, record: dict) -> list[PlannedWorkout]:
    """List every workout of a plan with its feedback and calendar booking.

    Args:
        context: Trusted tool context.
        record: Plan record from the repository.

    Returns:
        The workouts in plan order.
    """
    plan_id = record["id"]
    feedback = {
        (item.week_number, item.workout_index): item
        for item in context.repository.list_workout_feedback(
            context.student_id, plan_id
        )
    }
    bookings = {
        (item["week_number"], item["workout_index"]): CalendarBooking(
            event_id=item["event_id"], start=item["start_at"], end=item["end_at"]
        )
        for item in context.repository.list_calendar_events(context.student_id, plan_id)
    }
    workouts = []
    for week in TrainingPlan.model_validate(record["plan"]).weeks:
        for index, workout in enumerate(week.workouts):
            key = (week.number, index)
            workouts.append(
                PlannedWorkout(
                    **workout.model_dump(),
                    plan_id=plan_id,
                    week_number=week.number,
                    workout_index=index,
                    feedback=feedback.get(key),
                    booking=bookings.get(key),
                )
            )
    return workouts


@report_storage_errors
def get_user_profile(context: ToolContext) -> ProfileResult | ToolFailure:
    """Read the student's saved profile.

    Args:
        context: Trusted tool context.

    Returns:
        The profile, or ``student_not_found`` if none has been saved.
    """
    profile = context.repository.get_profile(context.student_id)
    if profile is None:
        return failure(STUDENT_NOT_FOUND, "Save a profile before asking RepWise.")
    return ProfileResult(profile=profile)


@report_storage_errors
def get_current_plan(context: ToolContext) -> CurrentPlanResult | ToolFailure:
    """Read the student's current active plan and the sources it cites.

    Args:
        context: Trusted tool context.

    Returns:
        The plan, or ``plan_not_found`` when the student has no active plan.
        Drafts waiting for approval are not the current plan.
    """
    record = resolve_plan(context, None)
    if isinstance(record, ToolFailure):
        return record
    return CurrentPlanResult(
        plan_id=record["id"],
        status=record["status"],
        created_at=record["created_at"],
        plan=TrainingPlan.model_validate(record["plan"]),
        sources=[
            PlanSource(
                source_id=source["source_id"], title=source["title"], url=source["url"]
            )
            for source in record["sources"]
        ],
    )


@report_storage_errors
def get_timetable(context: ToolContext) -> TimetableResult | ToolFailure:
    """Read the student's availability and the workouts of their current plan.

    Each workout comes with the ``plan_id``, ``week_number`` and
    ``workout_index`` other tools need, its saved feedback and its calendar
    booking, if any.

    Args:
        context: Trusted tool context.

    Returns:
        The timetable. Without an active plan, ``planned_workouts`` is empty.
        ``student_not_found`` when no profile is saved.
    """
    profile = context.repository.get_profile(context.student_id)
    if profile is None:
        return failure(STUDENT_NOT_FOUND, "Save a profile before asking RepWise.")
    record = context.repository.get_current_plan(context.student_id)
    return TimetableResult(
        available_days=profile.available_days,
        session_minutes=profile.session_minutes,
        plan_id=record["id"] if record else None,
        planned_workouts=planned_workouts(context, record) if record else [],
    )


@report_storage_errors
def get_workout(
    context: ToolContext,
    week_number: int,
    workout_index: int,
    plan_id: str | None = None,
) -> WorkoutResult | ToolFailure:
    """Read one workout with its feedback, booking and logged performance.

    Args:
        context: Trusted tool context.
        week_number: Week of the workout, from ``get_timetable``.
        workout_index: Position of the workout in its week.
        plan_id: The plan, or ``None`` for the current plan.

    Returns:
        The workout, or ``plan_not_found`` / ``workout_not_found``.
    """
    record = resolve_plan(context, plan_id)
    if isinstance(record, ToolFailure):
        return record
    workout = next(
        (
            item
            for item in planned_workouts(context, record)
            if (item.week_number, item.workout_index) == (week_number, workout_index)
        ),
        None,
    )
    if workout is None:
        return failure(WORKOUT_NOT_FOUND, "Workout not found in the plan.")
    performance = [
        PerformanceRecord(
            exercise_name=item["exercise_name"],
            sets_completed=item["sets_completed"],
            reps_per_set=item["reps_per_set"],
            weight_kg=item["weight_kg"],
        )
        for item in context.repository.list_workout_performance(
            context.student_id, record["id"]
        )
        if (item["week_number"], item["workout_index"]) == (week_number, workout_index)
    ]
    return WorkoutResult(workout=workout, performance=performance)
