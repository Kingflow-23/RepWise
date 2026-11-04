"""The three RepWise agent tools, aligned with the main API's contracts.

Inputs and outputs reuse ``src/schemas/models.py``, and feedback is written by
``StudentRepository``, so a record has the same shape as one saved by the API.

Vocabulary shared with the API:

================  ==========================================================
Tool              Main API
================  ==========================================================
``user_id``       ``AccessToken.user_id`` and the ``student_id`` route/column
``workout_id``    ``plan_id`` + ``week_number`` + ``workout_index``
``difficulty``    ``WorkoutFeedbackRequest.difficulty`` (see DIFFICULTY_TO_API)
``notes``         ``comment``; empty notes are stored as ``None``
(timestamp)       ``created_at``
(completed)       ``completion_status="completed"``
================  ==========================================================

Profiles and plans are synthetic and read from ``data/demo_data.json``. Feedback
goes to ``agentic.sqlite3`` in ``REPWISE_DATA_DIR`` (default ``data/agentic``),
which is never the application's own database.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from src.db.repository import StudentRepository, now
from src.schemas.models import (
    Model,
    PlanOptions,
    PlanRequest,
    TrainingPlan,
    UserProfile,
    WorkoutFeedback,
    WorkoutFeedbackRequest,
)

logger = logging.getLogger(__name__)

Difficulty = Literal["easy", "moderate", "hard"]
ToolResult = dict[str, object]

DIFFICULTY_TO_API: dict[str, str] = {
    "easy": "slightly_too_easy",
    "moderate": "just_right",
    "hard": "slightly_too_hard",
}
MAX_NOTES_LENGTH = 500

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEMO_DATA_FILE = Path(__file__).with_name("data") / "demo_data.json"
_DATABASE_NAME = "agentic.sqlite3"
_NO_LOGIN_PASSWORD_HASH = "!synthetic-account-without-login"


class PlannedWorkoutRef(Model):
    """Give one workout of a saved plan a stable, unique ID."""

    workout_id: str = Field(min_length=1)
    week_number: int = Field(ge=1)
    workout_index: int = Field(ge=0)


class DemoUser(Model):
    """One synthetic user, with the same field names as the main schemas."""

    user_id: str = Field(min_length=1)
    full_name: str = Field(min_length=3, max_length=100)
    profile: UserProfile
    plan_id: str = Field(min_length=1)
    plan_options: PlanOptions
    plan: TrainingPlan
    workouts: list[PlannedWorkoutRef]

    @model_validator(mode="after")
    def workouts_identify_scheduled_positions(self) -> "DemoUser":
        """Require unique workout IDs that each point at a real plan position."""
        ids = [item.workout_id for item in self.workouts]
        if len(ids) != len(set(ids)):
            raise ValueError("workout_id values must be unique")
        positions = {
            (week.number, index)
            for week in self.plan.weeks
            for index in range(len(week.workouts))
        }
        references = [(item.week_number, item.workout_index) for item in self.workouts]
        if len(references) != len(positions) or set(references) != positions:
            raise ValueError("workouts must cover each plan position exactly once")
        return self


class DemoData(Model):
    """The synthetic data file."""

    notice: str
    users: list[DemoUser]


def _data_dir() -> Path:
    """Return the feedback directory, honoring ``REPWISE_DATA_DIR``."""
    override = os.getenv("REPWISE_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return _REPOSITORY_ROOT / "data" / "agentic"


def _find_user(user_id: object) -> DemoUser | None:
    """Return the user with exactly this ID, without creating anything."""
    data = DemoData.model_validate_json(_DEMO_DATA_FILE.read_text(encoding="utf-8"))
    return next((user for user in data.users if user.user_id == user_id), None)


def _error(code: str, message: str, **fields: object) -> ToolResult:
    """Build the failure shape shared by all three tools."""
    return {"ok": False, "code": code, "error": message, **fields}


def _unknown_user(user_id: object, **fields: object) -> ToolResult:
    """Build the failure returned for a user ID that does not exist."""
    return _error("unknown_user", f"Unknown user_id: {user_id!r}.", **fields)


def _feedback_record(
    user: DemoUser, reference: PlannedWorkoutRef, feedback: WorkoutFeedback
) -> ToolResult:
    """Return the API's feedback record plus the tool's two identifiers."""
    return {
        "user_id": user.user_id,
        "workout_id": reference.workout_id,
        **feedback.model_dump(),
    }


def _load_feedback(user: DemoUser) -> dict[str, ToolResult]:
    """Read saved feedback by workout ID, without creating the database."""
    path = _data_dir() / _DATABASE_NAME
    if not path.exists():
        return {}
    saved = StudentRepository(path).list_workout_feedback(user.user_id, user.plan_id)
    return {
        reference.workout_id: _feedback_record(user, reference, feedback)
        for reference in user.workouts
        for feedback in saved
        if (feedback.week_number, feedback.workout_index)
        == (reference.week_number, reference.workout_index)
    }


def _writable_repository(user: DemoUser) -> StudentRepository:
    """Open the feedback database and add the rows its foreign keys require."""
    path = _data_dir() / _DATABASE_NAME
    repository = StudentRepository(path)
    repository.initialize()
    request = PlanRequest(
        profile=user.profile,
        weeks=len(user.plan.weeks),
        **user.plan_options.model_dump(),
    )
    timestamp = now()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "INSERT OR IGNORE INTO users(full_name, user_id, password_hash, created_at) "
            "VALUES (?, ?, ?, ?)",
            (user.full_name, user.user_id, _NO_LOGIN_PASSWORD_HASH, timestamp),
        )
        connection.execute(
            "INSERT OR IGNORE INTO student_profiles(student_id, profile_json, updated_at) "
            "VALUES (?, ?, ?)",
            (user.user_id, user.profile.model_dump_json(), timestamp),
        )
        connection.execute(
            "INSERT OR IGNORE INTO training_plans("
            "id, student_id, request_json, plan_json, sources_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                user.plan_id,
                user.user_id,
                request.model_dump_json(),
                user.plan.model_dump_json(),
                "[]",
                timestamp,
            ),
        )
    return repository


def get_saved_feedback(user_id: str, workout_id: str) -> ToolResult | None:
    """Reload one saved feedback record from storage; not exposed as a tool."""
    user = _find_user(user_id)
    return _load_feedback(user).get(workout_id) if user else None


def get_user_profile(user_id: str) -> ToolResult:
    """Return a user's saved profile, using the ``UserProfile`` fields.

    Args:
        user_id: Account ID, the same value as the API's ``student_id``.

    Returns:
        ``{"ok": True, "user_id": ..., "profile": {...}}``, or a failure with
        ``ok``, ``code``, and ``error``. Nothing is written.
    """
    user = _find_user(user_id)
    if user is None:
        return _unknown_user(user_id)
    return {"ok": True, "user_id": user.user_id, "profile": user.profile.model_dump()}


def get_timetable(user_id: str) -> ToolResult:
    """Return a user's weekly availability and planned workouts.

    Args:
        user_id: Account ID, the same value as the API's ``student_id``.

    Returns:
        ``availability`` holds the profile's ``available_days`` and
        ``session_minutes``. Each ``planned_workouts`` item holds the ``Workout``
        fields, its ``workout_id`` and plan position, and its saved ``feedback``
        or ``None``. Failures have ``ok``, ``code``, and ``error``. Nothing is
        written.
    """
    user = _find_user(user_id)
    if user is None:
        return _unknown_user(user_id)
    try:
        feedback = _load_feedback(user)
    except sqlite3.Error:
        logger.exception("Agentic feedback read failed")
        return _error("storage_error", "Saved feedback could not be read.")
    weeks = {week.number: week for week in user.plan.weeks}
    return {
        "ok": True,
        "user_id": user.user_id,
        "availability": {
            "available_days": user.profile.available_days,
            "session_minutes": user.profile.session_minutes,
        },
        "planned_workouts": [
            {
                "workout_id": reference.workout_id,
                "plan_id": user.plan_id,
                "week_number": reference.week_number,
                "workout_index": reference.workout_index,
                **weeks[reference.week_number]
                .workouts[reference.workout_index]
                .model_dump(),
                "feedback": feedback.get(reference.workout_id),
            }
            for reference in user.workouts
        ],
    }


def save_workout_feedback(
    user_id: str,
    workout_id: str,
    difficulty: Difficulty,
    notes: str = "",
) -> ToolResult:
    """Save feedback for a completed workout in the API's feedback format.

    Unknown users and workouts are rejected, never created. Saving again for the
    same workout updates its record, as the API route does. Nothing is adapted.

    Args:
        user_id: Account ID, the same value as the API's ``student_id``.
        workout_id: A ``workout_id`` returned by :func:`get_timetable`.
        difficulty: ``easy``, ``moderate``, or ``hard``; stored as the API value
            given by :data:`DIFFICULTY_TO_API`.
        notes: Optional text of up to 500 characters, stored as ``comment``.

    Returns:
        ``{"ok": True, "saved": True, "feedback": {...}}``, or a failure with
        ``ok``, ``saved`` set to ``False``, ``code``, and ``error``.
    """
    user = _find_user(user_id)
    if user is None:
        return _unknown_user(user_id, saved=False)
    reference = next(
        (item for item in user.workouts if item.workout_id == workout_id), None
    )
    if reference is None:
        return _error(
            "unknown_workout",
            f"Unknown workout_id for this user: {workout_id!r}.",
            saved=False,
        )
    if not isinstance(difficulty, str) or difficulty not in DIFFICULTY_TO_API:
        return _error(
            "invalid_difficulty",
            "difficulty must be easy, moderate, or hard.",
            saved=False,
        )
    if not isinstance(notes, str):
        return _error("invalid_notes", "notes must be a string.", saved=False)
    try:
        request = WorkoutFeedbackRequest(
            completion_status="completed",
            difficulty=DIFFICULTY_TO_API[difficulty],
            comment=notes.strip() or None,
        )
    except ValidationError:
        return _error(
            "invalid_notes",
            f"notes must be at most {MAX_NOTES_LENGTH} characters.",
            saved=False,
        )
    try:
        repository = _writable_repository(user)
        feedback = repository.save_workout_feedback(
            user.user_id,
            user.plan_id,
            reference.week_number,
            reference.workout_index,
            request,
        )
    except (OSError, sqlite3.Error):
        logger.exception("Agentic feedback save failed")
        return _error("storage_error", "Feedback could not be saved.", saved=False)
    logger.info(
        "Agentic feedback saved workout_id=%s difficulty=%s",
        reference.workout_id,
        feedback.difficulty,
    )
    return {
        "ok": True,
        "saved": True,
        "feedback": _feedback_record(user, reference, feedback),
    }
