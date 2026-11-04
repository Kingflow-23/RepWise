"""The three RepWise agent tools.

Parameters and records match the main API (see ``src/api/main.py`` and
``src/schemas/models.py``), and all data goes through ``StudentRepository`` in the
database at ``REPWISE_DATABASE_PATH``. Each tool returns a pydantic model, so its
output is validated. Call ``model_dump()`` on a result to get a dictionary.

There is no login check here: the tools trust the ``student_id`` they receive, so
use them only on a trusted local machine.
"""

from __future__ import annotations

import logging
import sqlite3
from functools import wraps
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from config.config import AppConfig
from src.db.repository import StudentRepository
from src.schemas.models import (
    Model,
    TrainingPlan,
    UserProfile,
    Workout,
    WorkoutFeedback,
    WorkoutFeedbackRequest,
)

logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

CompletionStatus = Literal["completed", "partially_completed", "not_completed"]
WorkoutDifficulty = Literal[
    "much_too_easy",
    "slightly_too_easy",
    "just_right",
    "slightly_too_hard",
    "much_too_hard",
]


class ToolFailure(Model):
    """Any tool's output when it could not do what was asked."""

    ok: Literal[False] = False
    code: Literal[
        "invalid_request",
        "student_not_found",
        "plan_not_found",
        "workout_not_found",
        "storage_error",
    ]
    error: str


class ProfileResult(Model):
    """Output of ``get_user_profile``."""

    ok: Literal[True] = True
    student_id: str
    profile: UserProfile


class PlannedWorkout(Workout):
    """A scheduled workout, where it sits in the plan, and any saved feedback."""

    plan_id: str
    week_number: int = Field(ge=1)
    workout_index: int = Field(ge=0)
    feedback: WorkoutFeedback | None = None


class TimetableResult(Model):
    """Output of ``get_timetable``."""

    ok: Literal[True] = True
    student_id: str
    available_days: list[str]
    session_minutes: int
    planned_workouts: list[PlannedWorkout]


class FeedbackResult(Model):
    """Output of ``save_workout_feedback``."""

    ok: Literal[True] = True
    saved: Literal[True] = True
    student_id: str
    feedback: WorkoutFeedback


STUDENT_NOT_FOUND = ToolFailure(
    code="student_not_found", error="Student profile not found"
)
PLAN_NOT_FOUND = ToolFailure(code="plan_not_found", error="Saved plan not found")
WORKOUT_NOT_FOUND = ToolFailure(
    code="workout_not_found", error="Workout not found in saved plan"
)


def open_repository() -> StudentRepository:
    """Open the app's existing database; a missing one is never created."""
    path = Path(AppConfig.from_environment().database_path)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if not path.exists():
        raise sqlite3.OperationalError(f"Database not found: {path}")
    return StudentRepository(path)


def _report_storage_errors(tool):
    """Return a ``ToolFailure`` instead of raising when the database fails."""

    @wraps(tool)
    def wrapper(*args, **kwargs):
        try:
            return tool(*args, **kwargs)
        except (OSError, sqlite3.Error):
            logger.exception("Agent tool storage failure tool=%s", tool.__name__)
            return ToolFailure(code="storage_error", error="The database failed.")

    return wrapper


@_report_storage_errors
def get_user_profile(student_id: str) -> ProfileResult | ToolFailure:
    """Read a student's saved profile. Nothing is written."""
    profile = open_repository().get_profile(student_id)
    if profile is None:
        return STUDENT_NOT_FOUND
    return ProfileResult(student_id=student_id, profile=profile)


@_report_storage_errors
def get_timetable(student_id: str) -> TimetableResult | ToolFailure:
    """Read a student's availability and latest plan's workouts. Nothing is written."""
    repository = open_repository()
    profile = repository.get_profile(student_id)
    if profile is None:
        return STUDENT_NOT_FOUND

    planned_workouts = []
    for record in repository.list_plans(student_id, limit=1):
        feedback = {
            (item["week_number"], item["workout_index"]): item
            for item in record["workout_feedback"]
        }
        for week in TrainingPlan.model_validate(record["plan"]).weeks:
            for index, workout in enumerate(week.workouts):
                planned_workouts.append(
                    PlannedWorkout(
                        **workout.model_dump(),
                        plan_id=record["id"],
                        week_number=week.number,
                        workout_index=index,
                        feedback=feedback.get((week.number, index)),
                    )
                )
    return TimetableResult(
        student_id=student_id,
        available_days=profile.available_days,
        session_minutes=profile.session_minutes,
        planned_workouts=planned_workouts,
    )


@_report_storage_errors
def save_workout_feedback(
    student_id: str,
    plan_id: str,
    week_number: int,
    workout_index: int,
    completion_status: CompletionStatus,
    difficulty: WorkoutDifficulty,
    comment: str | None = None,
) -> FeedbackResult | ToolFailure:
    """Save feedback for one workout, like the API's feedback route.

    Saving again for the same workout updates its record. Unlike the route, this
    never adapts the next workout.
    """
    try:
        request = WorkoutFeedbackRequest(
            completion_status=completion_status, difficulty=difficulty, comment=comment
        )
    except ValidationError as error:
        return ToolFailure(code="invalid_request", error=str(error))

    repository = open_repository()
    if repository.get_profile(student_id) is None:
        return STUDENT_NOT_FOUND
    record = repository.get_plan(student_id, plan_id)
    if record is None:
        return PLAN_NOT_FOUND
    plan = TrainingPlan.model_validate(record["plan"])
    week = next((item for item in plan.weeks if item.number == week_number), None)
    if week is None or workout_index not in range(len(week.workouts)):
        return WORKOUT_NOT_FOUND

    feedback = repository.save_workout_feedback(
        student_id, plan_id, week_number, workout_index, request
    )
    return FeedbackResult(student_id=student_id, feedback=feedback)
