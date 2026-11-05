"""Tools for reviewing and changing training: history, adaptation, and evidence.

Every tool is a plain function: pass its inputs, get a pydantic result back, or a
``Failure`` whose ``code`` says what went wrong. To try one on demo data, run
``python -m agentic.call_tool <tool_name> '<json arguments>'``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Literal

from pydantic import Field

from agentic import tools
from agentic.tools import report_storage_errors
from src.db.repository import StudentRepository
from src.generation.adaptation import (
    PAIN_TERMS,
    WorkoutAdaptationPipeline,
    next_unlogged_workout,
)
from src.generation.pipeline import validate_plan
from src.retrieval.query_expansion import ScopeRefusalError
from src.schemas.models import (
    Model,
    PlanRequest,
    TrainingPlan,
    Workout,
    WorkoutFeedbackRequest,
)

logger = logging.getLogger(__name__)

# PAIN_TERMS is English only, and students also write in French.
FRENCH_PAIN_TERMS = re.compile(
    r"\b(?:douleurs?|douloureux|douloureuse|blessée?s?|blessures?|mal\s+(?:au|à|aux))\b",
    re.IGNORECASE,
)
EXCERPT_CHARACTERS = 600


class Failure(Model):
    """A tool could not do what was asked. ``code`` is stable; ``error`` explains."""

    ok: Literal[False] = False
    code: str
    error: str


class PerformanceRecord(Model):
    """What the student actually did for one exercise."""

    exercise_name: str = Field(min_length=1)
    sets_completed: int = Field(ge=0, le=50)
    reps_per_set: str = Field(min_length=1, max_length=100)
    weight_kg: float | None = Field(default=None, ge=0, le=1000)


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
    performance: list[PerformanceRecord] = Field(default_factory=list)


class HistorySummary(Model):
    """Counts computed by code, so the model does not have to add them up."""

    workouts_logged: int
    completed: int
    partially_completed: int
    not_completed: int
    difficulty_counts: dict[str, int]
    mentions_pain: int


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
    """Move one workout to another day."""

    week_number: int = Field(ge=1)
    workout_index: int = Field(ge=0)
    new_day: str = Field(min_length=1)


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


class Document(Model):
    """One passage found in the fitness corpus."""

    source_id: str
    title: str
    excerpt: str
    url: str
    speaker: str | None = None


class SearchResult(Model):
    """Output of ``search_documents``, best match first."""

    ok: Literal[True] = True
    query: str
    documents: list[Document]


def mentions_pain(text: str | None) -> bool:
    """Say whether text mentions pain or injury, in English or French."""
    return bool(text and (PAIN_TERMS.search(text) or FRENCH_PAIN_TERMS.search(text)))


def open_agent_repository() -> StudentRepository:
    """Open the app's database and make sure the agent's tables exist."""
    repository = tools.open_repository()
    repository.initialize()
    return repository


def find_workout(
    plan: TrainingPlan, week_number: int, workout_index: int
) -> Workout | None:
    week = next((item for item in plan.weeks if item.number == week_number), None)
    if week is None or workout_index not in range(len(week.workouts)):
        return None
    return week.workouts[workout_index]


def _replace_workout(
    plan: TrainingPlan, week_number: int, workout_index: int, workout: Workout
) -> TrainingPlan:
    """Return a copy of the plan with one workout replaced."""
    weeks = []
    for week in plan.weeks:
        workouts = list(week.workouts)
        if week.number == week_number:
            workouts[workout_index] = workout
        weeks.append(week.model_copy(update={"workouts": workouts}))
    return plan.model_copy(update={"weeks": weeks})


def _validate_changed_plan(plan: TrainingPlan, saved: dict) -> None:
    """Apply the same checks as a generated plan. Raises ValueError if it fails."""
    source_ids = {source["source_id"] for source in saved["sources"]}
    validate_plan(
        plan,
        PlanRequest.model_validate(saved["request"]),
        source_ids,
        # A plan saved without sources, such as the demo's, has nothing to cite.
        require_citations=bool(source_ids),
    )


@report_storage_errors
def get_workout_history(
    student_id: str, max_plans: int = 5
) -> HistoryResult | tools.ToolFailure:
    """Read logged workouts across a student's recent plans. Nothing is written."""
    repository = open_agent_repository()
    if repository.get_profile(student_id) is None:
        return tools.STUDENT_NOT_FOUND

    records = repository.list_plans(student_id, limit=max_plans)
    entries: list[HistoryEntry] = []
    for record in records:
        plan = TrainingPlan.model_validate(record["plan"])
        feedback = {
            (item["week_number"], item["workout_index"]): item
            for item in record["workout_feedback"]
        }
        performance: dict[tuple[int, int], list[PerformanceRecord]] = {}
        for item in repository.list_workout_performance(student_id, record["id"]):
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
            mentions_pain=sum(mentions_pain(entry.comment) for entry in entries),
        ),
        workouts=entries,
        pending_recommendations=[
            Recommendation(**item)
            for item in repository.list_recommendations(student_id)
        ],
    )


@report_storage_errors
def log_workout_performance(
    student_id: str,
    plan_id: str,
    week_number: int,
    workout_index: int,
    exercises: list[PerformanceRecord],
) -> PerformanceResult | Failure | tools.ToolFailure:
    """Save what the student did in one workout. Saving again replaces an exercise.

    All the exercises are checked first, so a wrong name saves nothing.
    """
    repository = open_agent_repository()
    saved = repository.get_plan(student_id, plan_id)
    if saved is None:
        return tools.PLAN_NOT_FOUND
    workout = find_workout(
        TrainingPlan.model_validate(saved["plan"]), week_number, workout_index
    )
    if workout is None:
        return tools.WORKOUT_NOT_FOUND
    if not exercises:
        return Failure(code="invalid_request", error="Give at least one exercise.")

    names = {exercise.name.casefold(): exercise.name for exercise in workout.exercises}
    records = []
    for exercise in exercises:
        name = names.get(exercise.exercise_name.strip().casefold())
        if name is None:
            return Failure(
                code="exercise_not_found",
                error=f"{exercise.exercise_name!r} is not in this workout. "
                f"It has: {', '.join(names.values())}. Nothing was saved.",
            )
        records.append(exercise.model_copy(update={"exercise_name": name}))

    for record in records:
        repository.save_workout_performance(
            student_id, plan_id, week_number, workout_index, **record.model_dump()
        )
    return PerformanceResult(
        plan_id=plan_id,
        week_number=week_number,
        workout_index=workout_index,
        records=records,
    )


@report_storage_errors
def update_upcoming_workout(
    student_id: str,
    plan_id: str,
    week_number: int,
    workout_index: int,
    generator,
) -> UpdatedWorkoutResult | Failure | tools.ToolFailure:
    """Adapt the next unlogged workout using the feedback on a logged workout.

    ``week_number`` and ``workout_index`` name the logged workout. This follows the
    API's rules: the adapter can only change sets, reps, rest, duration, and notes;
    pain stops the change; a logged workout adapts at most one later workout.
    """
    repository = open_agent_repository()
    saved = repository.get_plan(student_id, plan_id)
    if saved is None:
        return tools.PLAN_NOT_FOUND
    plan = TrainingPlan.model_validate(saved["plan"])
    logged_workout = find_workout(plan, week_number, workout_index)
    if logged_workout is None:
        return tools.WORKOUT_NOT_FOUND

    all_feedback = repository.list_workout_feedback(student_id, plan_id)
    feedback = next(
        (
            item
            for item in all_feedback
            if (item.week_number, item.workout_index) == (week_number, workout_index)
        ),
        None,
    )
    if feedback is None:
        return Failure(
            code="feedback_not_found",
            error="Save feedback for this workout before adapting the next one.",
        )
    if feedback.adapted_week_number is not None:
        return Failure(
            code="already_adapted",
            error="This feedback has already adapted a later workout.",
        )
    if mentions_pain(feedback.comment):
        return Failure(
            code="pain_reported",
            error="No workout was changed because the feedback mentions pain. "
            "The student should seek appropriate advice.",
        )

    target = next_unlogged_workout(
        plan,
        after_week_number=week_number,
        after_workout_index=workout_index,
        logged_workouts={(i.week_number, i.workout_index) for i in all_feedback},
    )
    if target is None:
        return Failure(
            code="no_upcoming_workout",
            error="There is no later unlogged workout in this plan to adapt.",
        )
    target_week, target_index, target_workout = target

    try:
        replacement = WorkoutAdaptationPipeline(generator).adapt(
            request=PlanRequest.model_validate(saved["request"]),
            plan=plan,
            logged_workout=logged_workout,
            target_workout=target_workout,
            target_week_number=target_week,
            feedback=WorkoutFeedbackRequest(
                completion_status=feedback.completion_status,
                difficulty=feedback.difficulty,
                comment=feedback.comment,
            ),
        )
        changed = replacement != target_workout
        if changed:
            updated = _replace_workout(plan, target_week, target_index, replacement)
            _validate_changed_plan(updated, saved)
            repository.update_plan_after_feedback(
                student_id,
                plan_id,
                updated,
                week_number,
                workout_index,
                target_week,
                target_index,
            )
    except (RuntimeError, ValueError) as error:
        logger.exception("Workout adaptation failed plan_id=%s", plan_id)
        return Failure(
            code="adaptation_failed",
            error=f"No workout was changed: {error}",
        )
    return UpdatedWorkoutResult(
        changed=changed,
        plan_id=plan_id,
        week_number=target_week,
        workout_index=target_index,
        before=target_workout,
        after=replacement,
    )


@report_storage_errors
def organize_schedule(
    student_id: str, plan_id: str, moves: list[WorkoutMove]
) -> ScheduleResult | Failure | tools.ToolFailure:
    """Move workouts to other available days. All the moves apply, or none do."""
    repository = open_agent_repository()
    saved = repository.get_plan(student_id, plan_id)
    if saved is None:
        return tools.PLAN_NOT_FOUND
    profile = repository.get_profile(student_id)
    if profile is None:
        return tools.STUDENT_NOT_FOUND
    if not moves:
        return Failure(code="invalid_request", error="Give at least one move.")

    available = {day.casefold(): day for day in profile.available_days}
    plan = TrainingPlan.model_validate(saved["plan"])
    for move in moves:
        workout = find_workout(plan, move.week_number, move.workout_index)
        if workout is None:
            return tools.WORKOUT_NOT_FOUND
        new_day = available.get(move.new_day.strip().casefold())
        if new_day is None:
            return Failure(
                code="day_not_available",
                error=f"{move.new_day!r} is not an available day: "
                f"{', '.join(profile.available_days)}.",
            )
        plan = _replace_workout(
            plan,
            move.week_number,
            move.workout_index,
            workout.model_copy(update={"day": new_day}),
        )

    # Checked after all the moves, so that two workouts can swap days.
    for week in plan.weeks:
        days = [workout.day for workout in week.workouts]
        if len(days) != len(set(days)):
            return Failure(
                code="day_conflict",
                error=f"Week {week.number} would have two workouts on one day: "
                f"{', '.join(days)}. Move the other workout too.",
            )
    try:
        _validate_changed_plan(plan, saved)
    except ValueError as error:
        return Failure(code="invalid_schedule", error=str(error))

    repository.update_plan(student_id, plan_id, plan)
    moved = {(move.week_number, move.workout_index) for move in moves}
    return ScheduleResult(
        plan_id=plan_id,
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
            for event in repository.list_calendar_events(student_id, plan_id)
            if (event["week_number"], event["workout_index"]) in moved
        ],
    )


@report_storage_errors
def save_recommendation(
    student_id: str,
    summary: str,
    reason: str,
    plan_id: str | None = None,
    source_ids: list[str] | None = None,
) -> RecommendationResult | tools.ToolFailure:
    """Store a proposed change for later review. The plan itself is not changed.

    Without a ``plan_id``, the recommendation is attached to the latest plan.
    """
    repository = open_agent_repository()
    if plan_id is None:
        latest = repository.list_plans(student_id, limit=1)
        if not latest:
            return tools.PLAN_NOT_FOUND
        plan_id = latest[0]["id"]
    elif repository.get_plan(student_id, plan_id) is None:
        return tools.PLAN_NOT_FOUND
    return RecommendationResult(
        recommendation_id=repository.save_recommendation(
            student_id, plan_id, summary, reason, source_ids or []
        )
    )


def search_documents(query: str, retriever, top_k: int = 3) -> SearchResult | Failure:
    """Search the fitness corpus for passages. Nothing is generated or saved.

    Pain and injury questions are refused here. The agent passes the corpus search
    without the plan pipeline's scope check, which refuses knowledge questions such
    as "how long should I rest between sets?", because it only expects plan requests.
    """
    if mentions_pain(query):
        return Failure(
            code="health_question",
            error="RepWise does not answer pain or injury questions. "
            "Suggest seeing a health professional.",
        )
    try:
        chunks = retriever.search(query, top_k)
    except ScopeRefusalError as error:
        return Failure(code="out_of_scope", error=error.user_message)
    except (OSError, RuntimeError, ValueError) as error:
        logger.exception("Document search failed")
        return Failure(code="search_unavailable", error=str(error))
    return SearchResult(
        query=query,
        documents=[
            Document(
                source_id=chunk.chunk_id,
                title=chunk.title,
                excerpt=chunk.text[:EXCERPT_CHARACTERS],
                url=chunk.start_url or chunk.video_url,
                speaker=chunk.speaker,
            )
            for chunk in chunks
        ],
    )
