"""Training history, feedback and adjustment tools.

Owners (see ``config/tools.json``):

- Performance & Adaptation Agent: ``save_workout_feedback``,
  ``log_workout_performance``, ``update_upcoming_workout``;
- Workout Planning Agent: ``get_workout_history`` (also shared with Performance);
- Schedule & Calendar Agent: ``organize_schedule``;
- shared by Planning and Performance: ``save_recommendation``.

History is never overwritten: feedback and performance are stored per workout,
and an adapted plan keeps the link between the feedback and the workout it changed.

Catching up after missed workouts is decided by code, not by the model
(:func:`catch_up_moves`): the missed workouts go first onto the available days
that are left in the week, then the workouts not done yet, in plan order.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Literal

from pydantic import Field, ValidationError

from src.generation.adaptation import (
    WorkoutAdaptationPipeline,
    adapt_after_feedback,
    replace_workout,
)
from src.generation.pipeline import validate_plan
from src.schemas.models import (
    CompletionStatus,
    Model,
    PlanRequest,
    TrainingPlan,
    Workout,
    WorkoutDifficulty,
    WorkoutFeedback,
    WorkoutFeedbackRequest,
    weekday_number,
)
from src.tools.context import ToolContext
from src.tools.profile import (
    PerformanceRecord,
    PlannedWorkout,
    find_workout,
    planned_workouts,
    resolve_plan,
)
from src.tools.results import (
    STUDENT_NOT_FOUND,
    WORKOUT_NOT_FOUND,
    ToolFailure,
    failure,
    report_storage_errors,
)

logger = logging.getLogger(__name__)

# Logged statuses of a workout that was at least partly done: it keeps its day.
DONE_STATUSES = ("completed", "partially_completed")


class FeedbackResult(Model):
    """Output of ``save_workout_feedback``."""

    ok: Literal[True] = True
    saved: Literal[True] = True
    feedback: WorkoutFeedback


class HistoryEntry(Model):
    """One workout the student logged, in one of their plans."""

    plan_id: str
    plan_created_at: str
    week_number: int
    workout_index: int
    day: str
    focus: str
    completion_status: str | None = None
    difficulty: str | None = None
    comment: str | None = None
    health_concern: bool | None = None
    performance: list[PerformanceRecord] = Field(default_factory=list)


class HistorySummary(Model):
    """Counts computed by code, so the model never has to add them up."""

    workouts_logged: int
    completed: int
    partially_completed: int
    not_completed: int
    difficulty_counts: dict[str, int]
    health_concern_reports: int


class Recommendation(Model):
    """A proposed change waiting for review."""

    id: str
    plan_id: str
    summary: str
    reason: str
    source_ids: list[str] = Field(default_factory=list)
    created_at: str


class HistoryResult(Model):
    """Output of ``get_workout_history``, newest plan first."""

    ok: Literal[True] = True
    plans_checked: int
    summary: HistorySummary
    workouts: list[HistoryEntry]
    pending_recommendations: list[Recommendation]


class PerformanceResult(Model):
    """Output of ``log_workout_performance``."""

    ok: Literal[True] = True
    saved: Literal[True] = True
    plan_id: str
    week_number: int
    workout_index: int
    records: list[PerformanceRecord]


class UpdatedWorkoutResult(Model):
    """Output of ``update_upcoming_workout``, with the workout before and after."""

    ok: Literal[True] = True
    changed: bool
    plan_id: str
    week_number: int
    workout_index: int
    before: Workout
    after: Workout


class WorkoutMove(Model):
    """Move one workout to another day of the same week."""

    week_number: int = Field(ge=1)
    workout_index: int = Field(ge=0)
    new_day: str = Field(min_length=1, max_length=20)


class ScheduledWorkout(Model):
    """Where one workout sits after a schedule change."""

    week_number: int
    workout_index: int
    day: str
    focus: str


class ScheduleResult(Model):
    """Output of ``organize_schedule``."""

    ok: Literal[True] = True
    plan_id: str
    workouts: list[ScheduledWorkout]
    calendar_events_now_out_of_date: list[str] = Field(
        default_factory=list,
        description="Calendar event IDs of moved workouts. They were not changed.",
    )


class RecommendationResult(Model):
    """Output of ``save_recommendation``."""

    ok: Literal[True] = True
    saved: Literal[True] = True
    recommendation_id: str
    plan_id: str


@report_storage_errors
def save_workout_feedback(
    context: ToolContext,
    week_number: int,
    workout_index: int,
    completion_status: CompletionStatus,
    difficulty: WorkoutDifficulty,
    comment: str | None = None,
    plan_id: str | None = None,
) -> FeedbackResult | ToolFailure:
    """Save how one workout went, including a missed workout.

    The comment goes through the health check once, here, and the decision is
    stored with the feedback. Saving again for the same workout updates it.
    This tool never adapts another workout; ``update_upcoming_workout`` does.

    Args:
        context: Trusted tool context.
        week_number: Week of the workout.
        workout_index: Position of the workout in its week.
        completion_status: ``completed``, ``partially_completed`` or
            ``not_completed`` (a missed workout).
        difficulty: How the workout felt.
        comment: Optional free text from the student.
        plan_id: The plan, or ``None`` for the current plan.

    Returns:
        The saved feedback, or a failure (``invalid_request``,
        ``student_not_found``, ``plan_not_found``, ``workout_not_found``).
    """
    try:
        request = WorkoutFeedbackRequest(
            completion_status=completion_status, difficulty=difficulty, comment=comment
        )
    except ValidationError as error:
        return failure("invalid_request", str(error))
    if context.repository.get_profile(context.student_id) is None:
        return failure(STUDENT_NOT_FOUND, "Save a profile before logging workouts.")
    record = resolve_plan(context, plan_id)
    if isinstance(record, ToolFailure):
        return record
    plan = TrainingPlan.model_validate(record["plan"])
    if find_workout(plan, week_number, workout_index) is None:
        return failure(WORKOUT_NOT_FOUND, "Workout not found in the plan.")

    concern = context.health.check(request.comment) if request.comment else None
    feedback = context.repository.save_workout_feedback(
        context.student_id,
        record["id"],
        week_number,
        workout_index,
        request,
        health_concern=None if concern is None else concern.health_concern,
    )
    return FeedbackResult(feedback=feedback)


@report_storage_errors
def get_workout_history(
    context: ToolContext, max_plans: int = 5
) -> HistoryResult | ToolFailure:
    """Read logged workouts, performance and pending recommendations.

    Args:
        context: Trusted tool context.
        max_plans: How many recent plans to include, newest first.

    Returns:
        Every logged workout with summary counts computed by code. A student
        with no history gets empty lists and zero counts, not a failure.
        ``student_not_found`` when no profile is saved.
    """
    repository, student = context.repository, context.student_id
    if repository.get_profile(student) is None:
        return failure(STUDENT_NOT_FOUND, "Save a profile before asking RepWise.")

    records = repository.list_plans(student, limit=max_plans)
    entries: list[HistoryEntry] = []
    for record in records:
        plan = TrainingPlan.model_validate(record["plan"])
        feedback = {
            (item["week_number"], item["workout_index"]): item
            for item in record["workout_feedback"]
        }
        performance: dict[tuple[int, int], list[PerformanceRecord]] = {}
        for item in repository.list_workout_performance(student, record["id"]):
            key = (item["week_number"], item["workout_index"])
            performance.setdefault(key, []).append(
                PerformanceRecord(
                    exercise_name=item["exercise_name"],
                    sets_completed=item["sets_completed"],
                    reps_per_set=item["reps_per_set"],
                    weight_kg=item["weight_kg"],
                )
            )
        for week_number, workout_index in sorted(set(feedback) | set(performance)):
            workout = find_workout(plan, week_number, workout_index)
            if workout is None:
                continue
            logged = feedback.get((week_number, workout_index), {})
            entries.append(
                HistoryEntry(
                    plan_id=record["id"],
                    plan_created_at=record["created_at"],
                    week_number=week_number,
                    workout_index=workout_index,
                    day=workout.day,
                    focus=workout.focus,
                    completion_status=logged.get("completion_status"),
                    difficulty=logged.get("difficulty"),
                    comment=logged.get("comment"),
                    health_concern=logged.get("health_concern"),
                    performance=performance.get((week_number, workout_index), []),
                )
            )

    statuses = Counter(entry.completion_status for entry in entries)
    difficulties = Counter(entry.difficulty for entry in entries if entry.difficulty)
    return HistoryResult(
        plans_checked=len(records),
        summary=HistorySummary(
            workouts_logged=len(entries),
            completed=statuses["completed"],
            partially_completed=statuses["partially_completed"],
            not_completed=statuses["not_completed"],
            difficulty_counts=dict(difficulties),
            health_concern_reports=sum(bool(entry.health_concern) for entry in entries),
        ),
        workouts=entries,
        pending_recommendations=[
            Recommendation(**item) for item in repository.list_recommendations(student)
        ],
    )


@report_storage_errors
def log_workout_performance(
    context: ToolContext,
    week_number: int,
    workout_index: int,
    exercises: list[PerformanceRecord],
    plan_id: str | None = None,
) -> PerformanceResult | ToolFailure:
    """Save the weights, sets and reps the student reports for one workout.

    All exercise names are checked against the workout first, so a wrong name
    saves nothing. Saving the same exercise again replaces its record.

    Args:
        context: Trusted tool context.
        week_number: Week of the workout.
        workout_index: Position of the workout in its week.
        exercises: What was done, per exercise.
        plan_id: The plan, or ``None`` for the current plan.

    Returns:
        The saved records, or a failure (``plan_not_found``,
        ``workout_not_found``, ``exercise_not_found``, ``invalid_request``).
    """
    record = resolve_plan(context, plan_id)
    if isinstance(record, ToolFailure):
        return record
    workout = find_workout(
        TrainingPlan.model_validate(record["plan"]), week_number, workout_index
    )
    if workout is None:
        return failure(WORKOUT_NOT_FOUND, "Workout not found in the plan.")
    if not exercises:
        return failure("invalid_request", "Give at least one exercise.")

    names = {exercise.name.casefold(): exercise.name for exercise in workout.exercises}
    records = []
    for exercise in exercises:
        exercise = PerformanceRecord.model_validate(exercise)
        name = names.get(exercise.exercise_name.strip().casefold())
        if name is None:
            return failure(
                "exercise_not_found",
                f"{exercise.exercise_name!r} is not in this workout. "
                f"It has: {', '.join(names.values())}. Nothing was saved.",
            )
        records.append(exercise.model_copy(update={"exercise_name": name}))

    for item in records:
        context.repository.save_workout_performance(
            context.student_id,
            record["id"],
            week_number,
            workout_index,
            **item.model_dump(),
        )
    return PerformanceResult(
        plan_id=record["id"],
        week_number=week_number,
        workout_index=workout_index,
        records=records,
    )


ADAPTATION_FAILURES = {
    "feedback_not_found": "Save feedback for this workout before adapting the next one.",
    "already_adapted": "This feedback has already adapted a later workout.",
    "health_concern": (
        "No workout was changed because the feedback may describe pain or injury. "
        "RepWise does not give medical advice: the student should check with a "
        "health professional before training load changes."
    ),
    "no_upcoming_workout": "There is no later unlogged workout in this plan to adapt.",
    "unavailable": "Workout adaptation needs the local model, which is not configured.",
}


@report_storage_errors
def update_upcoming_workout(
    context: ToolContext,
    week_number: int,
    workout_index: int,
    plan_id: str | None = None,
) -> UpdatedWorkoutResult | ToolFailure:
    """Adapt the next unlogged workout using the feedback on a logged workout.

    ``week_number`` and ``workout_index`` name the *logged* workout. The rules
    are the API's (see ``src/generation/adaptation.py``): only sets, reps, rest,
    duration and notes may change; a health concern stops the change; one
    logged workout adapts at most one later workout. Approval is always required.

    Args:
        context: Trusted tool context.
        week_number: Week of the logged workout.
        workout_index: Position of the logged workout in its week.
        plan_id: The plan, or ``None`` for the current plan.

    Returns:
        The workout before and after (``changed`` may be ``False``), or a
        failure whose code says why nothing changed.
    """
    record = resolve_plan(context, plan_id)
    if isinstance(record, ToolFailure):
        return record
    try:
        adapter = WorkoutAdaptationPipeline(context.pipeline().generator)
    except (OSError, RuntimeError, ValueError):
        adapter = None
    try:
        outcome = adapt_after_feedback(
            context.repository,
            context.student_id,
            record["id"],
            week_number,
            workout_index,
            adapter,
        )
    except LookupError as error:
        return failure(WORKOUT_NOT_FOUND, str(error))
    if outcome.status in ADAPTATION_FAILURES:
        return failure(outcome.status, ADAPTATION_FAILURES[outcome.status])
    if outcome.status == "failed":
        return failure("adaptation_failed", f"No workout was changed: {outcome.detail}")
    target_week, target_index = outcome.target
    return UpdatedWorkoutResult(
        changed=outcome.status == "adapted",
        plan_id=record["id"],
        week_number=target_week,
        workout_index=target_index,
        before=outcome.before,
        after=outcome.after,
    )


def _week_workouts(
    context: ToolContext, record: dict, week_number: int
) -> list[PlannedWorkout]:
    """List one week's workouts with their feedback, in weekday order.

    Args:
        context: Trusted tool context.
        record: Plan record from the repository.
        week_number: Week to read.

    Returns:
        The week's workouts (empty when the week does not exist).
    """
    week = [
        item
        for item in planned_workouts(context, record)
        if item.week_number == week_number
    ]
    return sorted(week, key=lambda item: weekday_number(item.day) or 0)


def _status(item: PlannedWorkout) -> str | None:
    """Return a workout's logged completion status, or ``None`` if not logged."""
    return item.feedback.completion_status if item.feedback else None


def catch_up_moves(
    context: ToolContext,
    week_number: int,
    from_day: str | None = None,
    plan_id: str | None = None,
) -> list[WorkoutMove] | ToolFailure:
    """Compute the moves that put missed workouts first in the rest of a week.

    The rule, applied by code so it is the same every time:

    1. The rest of the week starts the day after the last logged workout
       (completed, partly completed or missed), or on ``from_day`` if later.
    2. Workouts already done keep their day.
    3. The missed workouts, then the workouts not logged yet, take the
       available days left in the week, earliest first, in plan order.
    4. A workout that no longer fits takes a free available day that has
       already passed, which means it is skipped this week (the approval
       message says so).

    Args:
        context: Trusted tool context.
        week_number: The week with missed workouts.
        from_day: Optional first day that may be used, such as today's weekday.
        plan_id: The plan, or ``None`` for the current plan.

    Returns:
        The moves (only workouts whose day changes), or a failure:
        ``nothing_to_catch_up`` (no missed workout logged in that week),
        ``no_day_left``, ``invalid_request``, ``plan_not_found``,
        ``workout_not_found``, ``student_not_found``.
    """
    record = resolve_plan(context, plan_id)
    if isinstance(record, ToolFailure):
        return record
    profile = context.repository.get_profile(context.student_id)
    if profile is None:
        return failure(STUDENT_NOT_FOUND, "Save a profile before asking RepWise.")
    week = _week_workouts(context, record, week_number)
    if not week:
        return failure(WORKOUT_NOT_FOUND, f"The plan has no week {week_number}.")
    missed = [item for item in week if _status(item) == "not_completed"]
    if not missed:
        return failure(
            "nothing_to_catch_up",
            f"No workout of week {week_number} is logged as missed. Record the "
            "missed workouts with save_workout_feedback first.",
        )

    start = max(weekday_number(item.day) or 0 for item in week if _status(item)) + 1
    if from_day:
        first = weekday_number(from_day)
        if first is None:
            return failure("invalid_request", f"{from_day!r} is not a weekday.")
        start = max(start, first)

    days = sorted(profile.available_days, key=lambda day: weekday_number(day) or 0)
    done_days = {item.day for item in week if _status(item) in DONE_STATUSES}
    remaining = [
        day
        for day in days
        if (weekday_number(day) or 0) >= start and day not in done_days
    ]
    if not remaining:
        return failure(
            "no_day_left",
            f"No available day is left in week {week_number} to catch up on.",
        )

    pending = [item for item in week if _status(item) is None]
    queue = [*missed, *pending]
    new_days = dict(zip((id(item) for item in queue), remaining))
    used = done_days | set(new_days.values())
    for item in queue[len(remaining) :]:
        # It no longer fits: keep its own day if it is free, else a free one.
        free = [item.day] if item.day not in used else []
        free += [day for day in days if day not in used]
        if not free:
            return failure(
                "no_day_left",
                f"Week {week_number} has more workouts than available days.",
            )
        new_days[id(item)] = free[0]
        used.add(free[0])
    return [
        WorkoutMove(
            week_number=week_number,
            workout_index=item.workout_index,
            new_day=new_days[id(item)],
        )
        for item in queue
        if new_days[id(item)] != item.day
    ]


def describe_moves(
    context: ToolContext, moves: list[WorkoutMove], plan_id: str | None = None
) -> str:
    """Describe moves in words, for the approval message.

    Args:
        context: Trusted tool context.
        moves: Moves to describe.
        plan_id: The plan, or ``None`` for the current plan.

    Returns:
        Text such as "Week 1: Lower body (missed Monday) → Friday.", or ``""``
        when the plan cannot be read.
    """
    record = resolve_plan(context, plan_id)
    if isinstance(record, ToolFailure):
        return ""
    lines = []
    for move in moves:
        week = _week_workouts(context, record, move.week_number)
        item = next((w for w in week if w.workout_index == move.workout_index), None)
        if item is None:
            continue
        logged = [weekday_number(w.day) or 0 for w in week if _status(w)]
        missed = " (missed)" if _status(item) == "not_completed" else ""
        passed = (
            ", a day that has passed, so it is skipped this week"
            if logged and (weekday_number(move.new_day) or 0) <= max(logged)
            else ""
        )
        lines.append(
            f"Week {move.week_number}: {item.focus}, {item.day}{missed} → "
            f"{move.new_day}{passed}."
        )
    return " ".join(lines)


@report_storage_errors
def organize_schedule(
    context: ToolContext,
    moves: list[WorkoutMove] | None = None,
    plan_id: str | None = None,
    catch_up_week: int | None = None,
    from_day: str | None = None,
) -> ScheduleResult | ToolFailure:
    """Move workouts to other available days. All the moves apply, or none do.

    Args:
        context: Trusted tool context.
        moves: Workouts to move. Two moves in one call can swap two days.
        plan_id: The plan, or ``None`` for the current plan.
        catch_up_week: Instead of ``moves``: a week with missed workouts;
            :func:`catch_up_moves` decides the moves.
        from_day: With ``catch_up_week``: the first day that may be used.

    Returns:
        The new schedule, plus the IDs of calendar events that now show an old
        time, or a failure (``day_not_available``, ``day_conflict``,
        ``invalid_schedule``, ``workout_not_found``, ``plan_not_found``, or a
        :func:`catch_up_moves` failure).
    """
    if catch_up_week is not None:
        computed = catch_up_moves(context, catch_up_week, from_day, plan_id)
        if isinstance(computed, ToolFailure):
            return computed
        moves = computed
    record = resolve_plan(context, plan_id)
    if isinstance(record, ToolFailure):
        return record
    profile = context.repository.get_profile(context.student_id)
    if profile is None:
        return failure(STUDENT_NOT_FOUND, "Save a profile before asking RepWise.")
    if not moves:
        return failure("invalid_request", "Give at least one move.")

    available = {day.casefold(): day for day in profile.available_days}
    plan = TrainingPlan.model_validate(record["plan"])
    parsed_moves = [WorkoutMove.model_validate(move) for move in moves]
    for move in parsed_moves:
        workout = find_workout(plan, move.week_number, move.workout_index)
        if workout is None:
            return failure(WORKOUT_NOT_FOUND, "Workout not found in the plan.")
        new_day = available.get(move.new_day.strip().casefold())
        if new_day is None:
            return failure(
                "day_not_available",
                f"{move.new_day!r} is not an available day: "
                f"{', '.join(profile.available_days)}.",
            )
        plan = replace_workout(
            plan,
            move.week_number,
            move.workout_index,
            workout.model_copy(update={"day": new_day}),
        )

    # Checked after all the moves, so that two workouts can swap days.
    for week in plan.weeks:
        days = [workout.day for workout in week.workouts]
        if len(days) != len(set(days)):
            return failure(
                "day_conflict",
                f"Week {week.number} would have two workouts on one day: "
                f"{', '.join(days)}. Move the other workout too.",
            )
    source_ids = {source["source_id"] for source in record["sources"]}
    # Validate against today's profile, not the snapshot saved with the plan:
    # a day the student made available since then is a valid destination.
    request = PlanRequest.model_validate(record["request"]).model_copy(
        update={"profile": profile}
    )
    try:
        validate_plan(plan, request, source_ids, require_citations=bool(source_ids))
    except ValueError as error:
        return failure("invalid_schedule", str(error))

    context.repository.update_plan(context.student_id, record["id"], plan)
    moved = {(move.week_number, move.workout_index) for move in parsed_moves}
    return ScheduleResult(
        plan_id=record["id"],
        workouts=[
            ScheduledWorkout(
                week_number=week.number,
                workout_index=index,
                day=workout.day,
                focus=workout.focus,
            )
            for week in plan.weeks
            for index, workout in enumerate(week.workouts)
        ],
        calendar_events_now_out_of_date=[
            event["event_id"]
            for event in context.repository.list_calendar_events(
                context.student_id, record["id"]
            )
            if (event["week_number"], event["workout_index"]) in moved
        ],
    )


@report_storage_errors
def save_recommendation(
    context: ToolContext,
    summary: str,
    reason: str,
    plan_id: str | None = None,
    source_ids: list[str] | None = None,
) -> RecommendationResult | ToolFailure:
    """Store a proposed change for later review. The plan itself is not changed.

    Args:
        context: Trusted tool context.
        summary: The proposed change.
        reason: Why it would help.
        plan_id: The plan it is about, or ``None`` for the current plan.
        source_ids: ``source_id`` values from ``search_documents`` that support it.

    Returns:
        The recommendation ID, or ``plan_not_found``.
    """
    record = resolve_plan(context, plan_id)
    if isinstance(record, ToolFailure):
        return record
    recommendation_id = context.repository.save_recommendation(
        context.student_id, record["id"], summary, reason, source_ids or []
    )
    return RecommendationResult(
        recommendation_id=recommendation_id, plan_id=record["id"]
    )
