"""SQLite storage for student profiles and completed plans."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from ..schemas.models import (
    PlanRequest,
    PlanResult,
    TrainingPlan,
    UserProfile,
    WorkoutFeedback,
    WorkoutFeedbackRequest,
)

logger = logging.getLogger(__name__)


def now() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        An aware UTC timestamp suitable for the repository's text columns.
    """
    return datetime.now(UTC).isoformat()


class StudentRepository:
    """Persist account, profile, and plan records in one configured SQLite file."""

    def __init__(self, database_path: str | Path):
        """Configure the SQLite database location.

        Args:
            database_path: File path for the database; parent directories are
                created when :meth:`initialize` is called.
        """
        self.database_path = str(database_path)

    def initialize(self) -> None:
        """Create the repository tables for a fresh or current-schema database.

        Raises:
            RuntimeError: If an existing database uses an unsupported schema.
            sqlite3.Error: If SQLite cannot create the schema.
        """
        logger.info("Initializing SQLite repository")
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    full_name TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            user_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(users)")
            }
            required_user_columns = {
                "full_name",
                "user_id",
                "password_hash",
                "created_at",
            }
            if not required_user_columns <= user_columns:
                raise RuntimeError(
                    "The existing database uses an unsupported schema. "
                    "Back it up and configure a fresh database path; automatic "
                    "schema migration is not supported."
                )
            connection.execute("""
                CREATE TABLE IF NOT EXISTS student_profiles (
                    student_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(student_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)
            profile_foreign_keys = connection.execute(
                "PRAGMA foreign_key_list(student_profiles)"
            ).fetchall()
            if not any(
                row[2] == "users" and row[3] == "student_id" and row[4] == "user_id"
                for row in profile_foreign_keys
            ):
                raise RuntimeError(
                    "The existing database uses an unsupported profile ownership "
                    "schema. Back it up and configure a fresh database path; "
                    "automatic schema migration is not supported."
                )
            connection.execute("""
                CREATE TABLE IF NOT EXISTS training_plans (
                    id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(student_id) REFERENCES student_profiles(student_id) ON DELETE CASCADE
                )
            """)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS plans_by_student "
                "ON training_plans(student_id, created_at DESC)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS plans_by_id_and_student "
                "ON training_plans(id, student_id)"
            )
            connection.execute("""
                CREATE TABLE IF NOT EXISTS workout_feedback (
                    id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    week_number INTEGER NOT NULL CHECK(week_number >= 1),
                    workout_index INTEGER NOT NULL CHECK(workout_index >= 0),
                    completion_status TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    comment TEXT,
                    created_at TEXT NOT NULL,
                    adapted_week_number INTEGER,
                    adapted_workout_index INTEGER,
                    UNIQUE(plan_id, week_number, workout_index),
                    FOREIGN KEY(plan_id, student_id)
                        REFERENCES training_plans(id, student_id) ON DELETE CASCADE
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS workout_performance (
                    id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    week_number INTEGER NOT NULL CHECK(week_number >= 1),
                    workout_index INTEGER NOT NULL CHECK(workout_index >= 0),
                    exercise_name TEXT NOT NULL,
                    sets_completed INTEGER NOT NULL CHECK(sets_completed >= 0),
                    reps_per_set TEXT NOT NULL,
                    weight_kg REAL,
                    created_at TEXT NOT NULL,
                    UNIQUE(plan_id, week_number, workout_index, exercise_name),
                    FOREIGN KEY(plan_id, student_id)
                        REFERENCES training_plans(id, student_id) ON DELETE CASCADE
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    source_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(plan_id, student_id)
                        REFERENCES training_plans(id, student_id) ON DELETE CASCADE
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS memory_facts (
                    student_id TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(student_id, fact),
                    FOREIGN KEY(student_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS memory_episodes (
                    student_id TEXT NOT NULL,
                    episode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(student_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS calendar_events (
                    event_id TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    week_number INTEGER NOT NULL CHECK(week_number >= 1),
                    workout_index INTEGER NOT NULL CHECK(workout_index >= 0),
                    start_at TEXT NOT NULL,
                    end_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(plan_id, week_number, workout_index),
                    FOREIGN KEY(plan_id, student_id)
                        REFERENCES training_plans(id, student_id) ON DELETE CASCADE
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS pending_actions (
                    id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    arguments_hash TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    proposed_by TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    decided_at TEXT,
                    FOREIGN KEY(student_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS pending_actions_by_student "
                "ON pending_actions(student_id, status, created_at DESC)"
            )
            connection.execute("""
                CREATE TABLE IF NOT EXISTS saved_locations (
                    student_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    consented_at TEXT NOT NULL,
                    FOREIGN KEY(student_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)
            # Columns added after the first release. Older databases get them
            # here, with defaults that keep their existing rows meaningful.
            self._add_missing_column(
                connection,
                "training_plans",
                "status",
                "TEXT NOT NULL DEFAULT 'active'",
            )
            self._add_missing_column(
                connection, "workout_feedback", "health_concern", "INTEGER"
            )
        logger.info("SQLite repository initialized")

    @staticmethod
    def _add_missing_column(
        connection: sqlite3.Connection, table: str, column: str, definition: str
    ) -> None:
        """Add a column to an existing table if it is not there yet.

        Args:
            connection: Open connection inside the initialization transaction.
            table: Table to extend. Only fixed names from this module are passed.
            column: Column name to add.
            definition: SQLite type and default, such as ``"INTEGER"``.
        """
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            logger.info("Database column added table=%s column=%s", table, column)

    def create_user(self, full_name: str, password_hash: str) -> str:
        """Create an account and return its generated internal ID.

        Args:
            full_name: Account name, which must be unique.
            password_hash: Encoded password hash; plaintext is not stored here.

        Returns:
            The UUID string used as the account's internal identity.

        Raises:
            sqlite3.IntegrityError: If the full name is already registered.
        """
        user_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO users(full_name, user_id, password_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (full_name, user_id, password_hash, now()),
            )
        logger.info("Database account created")
        return user_id

    def get_password_hash(self, full_name: str) -> str | None:
        """Look up a password hash by account name.

        Args:
            full_name: Name identifying the account.

        Returns:
            The stored encoded hash, or ``None`` if no account has that name.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT password_hash FROM users WHERE full_name=?", (full_name,)
            ).fetchone()
        return row[0] if row else None

    def get_account_identity(self, user_id: str) -> dict[str, str] | None:
        """Read only the public fields for an authenticated account."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id, full_name FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
        return {"user_id": row[0], "full_name": row[1]} if row else None

    def get_account_by_name(self, full_name: str) -> tuple[str, str, str] | None:
        """Look up the account fields needed for login.

        Args:
            full_name: Name identifying the account.

        Returns:
            A tuple of ``(user_id, password_hash, full_name)``, or ``None`` if
            no account has that name.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id, password_hash, full_name FROM users WHERE full_name=?",
                (full_name,),
            ).fetchone()
        return tuple(row) if row else None

    def save_profile(self, student_id: str, profile: UserProfile) -> None:
        """Create or replace a student's saved profile.

        Args:
            student_id: Internal account ID that owns the profile.
            profile: Validated profile supplied by that account.

        Raises:
            sqlite3.IntegrityError: If ``student_id`` is not a registered account.
        """
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO student_profiles(student_id, profile_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(student_id) DO UPDATE SET
                    profile_json=excluded.profile_json, updated_at=excluded.updated_at
            """,
                (student_id, profile.model_dump_json(), now()),
            )
        logger.info("Database profile saved")

    def get_profile(self, student_id: str) -> UserProfile | None:
        """Load a student's saved profile.

        Args:
            student_id: Internal account ID that owns the profile.

        Returns:
            The validated profile, or ``None`` if no profile has been saved.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT profile_json FROM student_profiles WHERE student_id=?",
                (student_id,),
            ).fetchone()
        return UserProfile.model_validate_json(row[0]) if row else None

    def save_plan(
        self,
        student_id: str,
        request: PlanRequest,
        result: PlanResult,
        status: str = "active",
    ) -> str:
        """Persist a completed plan with its request and evidence snapshot.

        Saving an ``active`` plan archives the student's previous active plan in
        the same transaction, so there is always at most one current plan. A
        ``draft`` plan changes nothing else until :meth:`activate_plan`.

        Args:
            student_id: Internal account ID that owns the plan.
            request: Validated request used to generate the plan.
            result: Completed pipeline result, including the plan and sources.
            status: ``"active"`` (the default, used by the API) or ``"draft"``.

        Returns:
            The generated UUID string identifying the saved plan.

        Raises:
            ValueError: If ``result`` is not completed, contains no plan, or
                ``status`` is not ``active`` or ``draft``.
            sqlite3.IntegrityError: If ``student_id`` has no saved profile.
        """
        if result.status != "completed" or result.plan is None:
            raise ValueError("only completed plans can be saved")
        if status not in {"active", "draft"}:
            raise ValueError("a new plan must be saved as active or draft")
        plan_id = str(uuid.uuid4())
        with self._connect() as connection:
            if status == "active":
                self._archive_active_plans(connection, student_id)
            connection.execute(
                """
                INSERT INTO training_plans(
                    id, student_id, request_json, plan_json, sources_json,
                    created_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    plan_id,
                    student_id,
                    request.model_dump_json(),
                    result.plan.model_dump_json(),
                    json.dumps(
                        [source.model_dump() for source in result.sources],
                        ensure_ascii=False,
                    ),
                    now(),
                    status,
                ),
            )
        return plan_id

    @staticmethod
    def _archive_active_plans(connection: sqlite3.Connection, student_id: str) -> None:
        """Mark every active plan of a student as archived.

        Args:
            connection: Open connection inside the caller's transaction.
            student_id: Account whose active plans are replaced.
        """
        connection.execute(
            "UPDATE training_plans SET status='archived' "
            "WHERE student_id=? AND status='active'",
            (student_id,),
        )

    def activate_plan(self, student_id: str, plan_id: str) -> bool:
        """Make an owned draft (or archived) plan the student's current plan.

        The previously active plan is archived in the same transaction.

        Args:
            student_id: Internal account ID that must own the plan.
            plan_id: ID of the plan to activate.

        Returns:
            ``True`` if the plan exists for this student and is now active,
            ``False`` if the student owns no such plan.
        """
        with self._connect() as connection:
            owned = connection.execute(
                "SELECT status FROM training_plans WHERE id=? AND student_id=?",
                (plan_id, student_id),
            ).fetchone()
            if owned is None:
                return False
            if owned[0] != "active":
                self._archive_active_plans(connection, student_id)
                connection.execute(
                    "UPDATE training_plans SET status='active' "
                    "WHERE id=? AND student_id=?",
                    (plan_id, student_id),
                )
        logger.info("Plan activated")
        return True

    def get_current_plan(self, student_id: str) -> dict[str, object] | None:
        """Load the student's current (newest active) plan with its feedback.

        Args:
            student_id: Internal account ID whose current plan is requested.

        Returns:
            The same record shape as :meth:`list_plans`, or ``None`` when the
            student has no active plan.
        """
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id FROM training_plans
                WHERE student_id=? AND status='active'
                ORDER BY created_at DESC, id DESC LIMIT 1
                """,
                (student_id,),
            ).fetchone()
        if row is None:
            return None
        record = self.get_plan(student_id, row[0])
        if record is None:
            return None
        record["workout_feedback"] = [
            feedback.model_dump()
            for feedback in self.list_workout_feedback(student_id, row[0])
        ]
        return record

    def list_plans(self, student_id: str, limit: int = 20) -> list[dict[str, object]]:
        """List the most recent saved plans for one student.

        Args:
            student_id: Internal account ID whose plans are requested.
            limit: Maximum number of plans to return, from 1 through 100.

        Returns:
            Plan records ordered newest first, each containing its ID, original
            request, plan, sources, creation timestamp, status (``draft``,
            ``active`` or ``archived``), and saved workout feedback.

        Raises:
            ValueError: If ``limit`` is not an integer from 1 through 100.
        """
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise ValueError("limit must be an integer between 1 and 100")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, request_json, plan_json, sources_json, created_at, status
                FROM training_plans WHERE student_id=? ORDER BY created_at DESC LIMIT ?
            """,
                (student_id, limit),
            ).fetchall()
        logger.info("Database plan history read count=%s", len(rows))
        return [
            {
                **self._plan_record(row),
                "workout_feedback": [
                    feedback.model_dump()
                    for feedback in self.list_workout_feedback(student_id, row[0])
                ],
            }
            for row in rows
        ]

    def get_plan(self, student_id: str, plan_id: str) -> dict[str, object] | None:
        """Load a saved plan only when it belongs to the given student.

        Args:
            student_id: Internal account ID that must own the plan.
            plan_id: ID of the plan to load.

        Returns:
            The plan record (ID, request, plan, sources, creation time and
            status), or ``None`` if this student owns no such plan.
        """
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, request_json, plan_json, sources_json, created_at, status
                FROM training_plans WHERE id=? AND student_id=?
                """,
                (plan_id, student_id),
            ).fetchone()
        return None if row is None else self._plan_record(row)

    @staticmethod
    def _plan_record(row: tuple) -> dict[str, object]:
        """Turn one ``training_plans`` row into the record shape the API returns.

        Args:
            row: ``(id, request_json, plan_json, sources_json, created_at, status)``.

        Returns:
            A dictionary with decoded JSON fields.
        """
        return {
            "id": row[0],
            "request": json.loads(row[1]),
            "plan": json.loads(row[2]),
            "sources": json.loads(row[3]),
            "created_at": row[4],
            "status": row[5],
        }

    def save_workout_feedback(
        self,
        student_id: str,
        plan_id: str,
        week_number: int,
        workout_index: int,
        feedback: WorkoutFeedbackRequest,
        health_concern: bool | None = None,
    ) -> WorkoutFeedback:
        """Create or update one feedback record for an owned scheduled workout.

        Args:
            student_id: Internal account ID that must own the plan.
            plan_id: ID of the plan containing the workout.
            week_number: Week of the workout, starting at 1.
            workout_index: Zero-based position of the workout in its week.
            feedback: Completion status, difficulty and optional comment.
            health_concern: Result of the health check on the comment, stored
                so later steps never need to re-read the comment to decide.

        Returns:
            The saved feedback record.

        Raises:
            ValueError: If the plan does not belong to this student.
            RuntimeError: If the saved row cannot be read back.
        """
        with self._connect() as connection:
            owns_plan = connection.execute(
                "SELECT 1 FROM training_plans WHERE id=? AND student_id=?",
                (plan_id, student_id),
            ).fetchone()
            if owns_plan is None:
                raise ValueError("the saved plan does not belong to this student")
            connection.execute(
                """
                INSERT INTO workout_feedback(
                    id, student_id, plan_id, week_number, workout_index,
                    completion_status, difficulty, comment, created_at,
                    health_concern
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id, week_number, workout_index) DO UPDATE SET
                    completion_status=excluded.completion_status,
                    difficulty=excluded.difficulty,
                    comment=excluded.comment,
                    created_at=excluded.created_at,
                    health_concern=excluded.health_concern
                """,
                (
                    str(uuid.uuid4()),
                    student_id,
                    plan_id,
                    week_number,
                    workout_index,
                    feedback.completion_status,
                    feedback.difficulty,
                    feedback.comment,
                    now(),
                    None if health_concern is None else int(health_concern),
                ),
            )
            row = connection.execute(
                f"""
                SELECT {self._FEEDBACK_COLUMNS}
                FROM workout_feedback
                WHERE student_id=? AND plan_id=? AND week_number=? AND workout_index=?
                """,
                (student_id, plan_id, week_number, workout_index),
            ).fetchone()
        if row is None:
            raise RuntimeError("Saved workout feedback could not be retrieved")
        return self._feedback_record(row)

    _FEEDBACK_COLUMNS = (
        "id, plan_id, week_number, workout_index, completion_status, difficulty, "
        "comment, created_at, adapted_week_number, adapted_workout_index, "
        "health_concern"
    )

    @staticmethod
    def _feedback_record(row: tuple) -> WorkoutFeedback:
        """Turn one ``workout_feedback`` row into a validated record.

        Args:
            row: Values in the order of ``_FEEDBACK_COLUMNS``.

        Returns:
            The feedback model, with ``health_concern`` decoded to a boolean.
        """
        return WorkoutFeedback(
            id=row[0],
            plan_id=row[1],
            week_number=row[2],
            workout_index=row[3],
            completion_status=row[4],
            difficulty=row[5],
            comment=row[6],
            created_at=row[7],
            adapted_week_number=row[8],
            adapted_workout_index=row[9],
            health_concern=None if row[10] is None else bool(row[10]),
        )

    def list_workout_feedback(
        self, student_id: str, plan_id: str
    ) -> list[WorkoutFeedback]:
        """Return saved feedback records for one student's plan.

        Args:
            student_id: Internal account ID that owns the plan.
            plan_id: ID of the plan.

        Returns:
            Feedback records ordered by week and workout position.
        """
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {self._FEEDBACK_COLUMNS}
                FROM workout_feedback
                WHERE student_id=? AND plan_id=?
                ORDER BY week_number, workout_index
                """,
                (student_id, plan_id),
            ).fetchall()
        return [self._feedback_record(row) for row in rows]

    def get_latest_plan(self, student_id: str) -> dict[str, object] | None:
        """Load the current active request and plan without its source evidence.

        Args:
            student_id: Internal account ID whose latest plan is requested.

        Returns:
            The saved request and plan of the newest active plan, or ``None``
            when the student has no active plan. Drafts are ignored, so an
            unapproved draft never shapes the next generated plan.
        """
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_json, plan_json
                FROM training_plans
                WHERE student_id=? AND status='active'
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (student_id,),
            ).fetchone()
        if row is None:
            return None
        logger.info("Latest saved plan loaded")
        return {"request": json.loads(row[0]), "plan": json.loads(row[1])}

    def update_plan(self, student_id: str, plan_id: str, plan: TrainingPlan) -> bool:
        """Replace a saved plan only when it belongs to the given student.

        Args:
            student_id: Internal account ID that must own the plan.
            plan_id: ID of the saved plan to replace.
            plan: Validated replacement plan.

        Returns:
            ``True`` if one owned plan was updated, otherwise ``False``.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE training_plans SET plan_json=? WHERE id=? AND student_id=?",
                (plan.model_dump_json(), plan_id, student_id),
            )
        logger.info("Database plan updated found=%s", cursor.rowcount == 1)
        return cursor.rowcount == 1

    def update_plan_after_feedback(
        self,
        student_id: str,
        plan_id: str,
        plan: TrainingPlan,
        feedback_week_number: int,
        feedback_workout_index: int,
        adapted_week_number: int,
        adapted_workout_index: int,
    ) -> bool:
        """Atomically save an adapted plan and its feedback-to-workout link."""
        with self._connect() as connection:
            plan_update = connection.execute(
                "UPDATE training_plans SET plan_json=? WHERE id=? AND student_id=?",
                (plan.model_dump_json(), plan_id, student_id),
            )
            if plan_update.rowcount != 1:
                return False
            feedback_update = connection.execute(
                """
                UPDATE workout_feedback
                SET adapted_week_number=?, adapted_workout_index=?
                WHERE student_id=? AND plan_id=? AND week_number=? AND workout_index=?
                """,
                (
                    adapted_week_number,
                    adapted_workout_index,
                    student_id,
                    plan_id,
                    feedback_week_number,
                    feedback_workout_index,
                ),
            )
            if feedback_update.rowcount != 1:
                raise RuntimeError("Workout feedback adaptation could not be recorded")
        return True

    def save_workout_performance(
        self,
        student_id: str,
        plan_id: str,
        week_number: int,
        workout_index: int,
        exercise_name: str,
        sets_completed: int,
        reps_per_set: str,
        weight_kg: float | None,
    ) -> None:
        """Save what was done for one exercise; saving again replaces the record."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workout_performance (
                    id, student_id, plan_id, week_number, workout_index,
                    exercise_name, sets_completed, reps_per_set, weight_kg, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id, week_number, workout_index, exercise_name)
                DO UPDATE SET
                    sets_completed=excluded.sets_completed,
                    reps_per_set=excluded.reps_per_set,
                    weight_kg=excluded.weight_kg,
                    created_at=excluded.created_at
                """,
                (
                    str(uuid.uuid4()),
                    student_id,
                    plan_id,
                    week_number,
                    workout_index,
                    exercise_name,
                    sets_completed,
                    reps_per_set,
                    weight_kg,
                    now(),
                ),
            )

    def list_workout_performance(
        self, student_id: str, plan_id: str
    ) -> list[dict[str, object]]:
        """Return the exercise records saved for one student's plan."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT week_number, workout_index, exercise_name, sets_completed,
                       reps_per_set, weight_kg, created_at
                FROM workout_performance
                WHERE student_id=? AND plan_id=?
                ORDER BY week_number, workout_index, exercise_name
                """,
                (student_id, plan_id),
            ).fetchall()
        keys = (
            "week_number",
            "workout_index",
            "exercise_name",
            "sets_completed",
            "reps_per_set",
            "weight_kg",
            "created_at",
        )
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def save_recommendation(
        self,
        student_id: str,
        plan_id: str,
        summary: str,
        reason: str,
        source_ids: list[str],
    ) -> str:
        """Store a proposed change for later review and return its ID."""
        recommendation_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO recommendations (
                    id, student_id, plan_id, summary, reason, source_ids_json,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'proposed', ?)
                """,
                (
                    recommendation_id,
                    student_id,
                    plan_id,
                    summary,
                    reason,
                    json.dumps(source_ids),
                    now(),
                ),
            )
        return recommendation_id

    def list_recommendations(self, student_id: str) -> list[dict[str, object]]:
        """Return a student's proposed recommendations, newest first."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, plan_id, summary, reason, source_ids_json, created_at
                FROM recommendations
                WHERE student_id=? AND status='proposed'
                ORDER BY created_at DESC, id DESC
                """,
                (student_id,),
            ).fetchall()
        return [
            {
                "id": row[0],
                "plan_id": row[1],
                "summary": row[2],
                "reason": row[3],
                "source_ids": json.loads(row[4]),
                "created_at": row[5],
            }
            for row in rows
        ]

    def save_memory_fact(self, student_id: str, fact: str) -> None:
        """Keep a stable fact about a student. Saving it twice keeps one copy."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO memory_facts (student_id, fact, created_at)
                VALUES (?, ?, ?)
                """,
                (student_id, fact, now()),
            )

    def list_memory_facts(self, student_id: str, limit: int) -> list[str]:
        """Return a student's remembered facts, newest first."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT fact FROM memory_facts WHERE student_id=?
                ORDER BY rowid DESC LIMIT ?
                """,
                (student_id, limit),
            ).fetchall()
        return [row[0] for row in rows]

    def save_memory_episode(
        self, student_id: str, episode: str, keep_after: str | None = None
    ) -> None:
        """Keep a short record of one completed request.

        Args:
            student_id: Account the episode belongs to.
            episode: Short text describing what happened.
            keep_after: Optional ISO timestamp. Older episodes of this student
                are deleted in the same transaction, which applies the
                retention policy without a separate cleanup job.
        """
        with self._connect() as connection:
            if keep_after is not None:
                connection.execute(
                    "DELETE FROM memory_episodes WHERE student_id=? AND created_at<?",
                    (student_id, keep_after),
                )
            connection.execute(
                """
                INSERT INTO memory_episodes (student_id, episode, created_at)
                VALUES (?, ?, ?)
                """,
                (student_id, episode, now()),
            )

    def list_memory_episodes(
        self, student_id: str, limit: int, keep_after: str | None = None
    ) -> list[str]:
        """Return a student's latest episodes, newest first.

        Args:
            student_id: Account whose episodes are read.
            limit: Maximum number of episodes.
            keep_after: Optional ISO timestamp; older episodes are not returned.

        Returns:
            Episode texts, newest first.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT episode FROM memory_episodes
                WHERE student_id=? AND created_at>=?
                ORDER BY rowid DESC LIMIT ?
                """,
                (student_id, keep_after or "", limit),
            ).fetchall()
        return [row[0] for row in rows]

    # Pending actions: consequential writes waiting for the student's decision.

    _ACTION_COLUMNS = (
        "id, student_id, tool_name, arguments_json, arguments_hash, reason, "
        "proposed_by, status, result_json, created_at, expires_at, decided_at"
    )

    def create_pending_action(
        self,
        student_id: str,
        tool_name: str,
        arguments: dict[str, object],
        arguments_hash: str,
        reason: str,
        proposed_by: str,
        expires_at: str,
    ) -> dict[str, object]:
        """Store a proposed write action that must be approved before it runs.

        Args:
            student_id: Account that must approve the action.
            tool_name: Catalog name of the tool to run once approved.
            arguments: Exact, validated tool arguments to run.
            arguments_hash: SHA-256 of the canonical arguments JSON.
            reason: Why approval is needed, shown to the student.
            proposed_by: Name of the agent that proposed the action.
            expires_at: ISO timestamp after which approval is refused.

        Returns:
            The stored action record (see :meth:`get_pending_action`).
        """
        action_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                f"""
                INSERT INTO pending_actions ({self._ACTION_COLUMNS})
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?, ?, NULL)
                """,
                (
                    action_id,
                    student_id,
                    tool_name,
                    json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                    arguments_hash,
                    reason,
                    proposed_by,
                    now(),
                    expires_at,
                ),
            )
        record = self.get_pending_action(student_id, action_id)
        if record is None:
            raise RuntimeError("The pending action could not be read back")
        return record

    def get_pending_action(
        self, student_id: str, action_id: str
    ) -> dict[str, object] | None:
        """Load one action only when it belongs to the given student.

        Args:
            student_id: Account that must own the action.
            action_id: ID returned when the action was proposed.

        Returns:
            The action record, or ``None`` for an unknown or foreign action.
        """
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {self._ACTION_COLUMNS} FROM pending_actions "
                "WHERE id=? AND student_id=?",
                (action_id, student_id),
            ).fetchone()
        return None if row is None else self._action_record(row)

    def list_pending_actions(
        self, student_id: str, status: str | None = "pending", limit: int = 50
    ) -> list[dict[str, object]]:
        """List a student's actions, newest first.

        Args:
            student_id: Account whose actions are listed.
            status: Only actions with this status, or ``None`` for all.
            limit: Maximum number of actions.

        Returns:
            Action records.
        """
        query = f"SELECT {self._ACTION_COLUMNS} FROM pending_actions WHERE student_id=?"
        parameters: list[object] = [student_id]
        if status is not None:
            query += " AND status=?"
            parameters.append(status)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._action_record(row) for row in rows]

    def decide_pending_action(
        self, student_id: str, action_id: str, new_status: str, current_time: str
    ) -> bool:
        """Move a still-pending, unexpired action to ``approved`` or ``declined``.

        The update only matches a pending action that has not expired, so two
        concurrent approvals cannot both win, and an approval cannot be replayed.

        Args:
            student_id: Account that must own the action.
            action_id: Action to decide.
            new_status: ``"approved"`` or ``"declined"``.
            current_time: Current ISO timestamp, compared with ``expires_at``.

        Returns:
            ``True`` if this call made the decision, ``False`` otherwise.

        Raises:
            ValueError: If ``new_status`` is not approved or declined.
        """
        if new_status not in {"approved", "declined"}:
            raise ValueError("an action can only be approved or declined")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE pending_actions SET status=?, decided_at=?
                WHERE id=? AND student_id=? AND status='pending' AND expires_at>?
                """,
                (new_status, current_time, action_id, student_id, current_time),
            )
        return cursor.rowcount == 1

    def finish_pending_action(
        self, student_id: str, action_id: str, status: str, result: dict[str, object]
    ) -> None:
        """Record what happened when an approved action ran.

        Args:
            student_id: Account that owns the action.
            action_id: Approved action.
            status: ``"executed"`` or ``"failed"``.
            result: The tool's structured result.
        """
        with self._connect() as connection:
            connection.execute(
                "UPDATE pending_actions SET status=?, result_json=? "
                "WHERE id=? AND student_id=? AND status='approved'",
                (
                    status,
                    json.dumps(result, ensure_ascii=False, default=str),
                    action_id,
                    student_id,
                ),
            )

    def expire_pending_actions(self, student_id: str, current_time: str) -> None:
        """Mark a student's pending actions that are past their expiry.

        Args:
            student_id: Account whose actions are checked.
            current_time: Current ISO timestamp.
        """
        with self._connect() as connection:
            connection.execute(
                "UPDATE pending_actions SET status='expired' "
                "WHERE student_id=? AND status='pending' AND expires_at<=?",
                (student_id, current_time),
            )

    @staticmethod
    def _action_record(row: tuple) -> dict[str, object]:
        """Turn one ``pending_actions`` row into a dictionary.

        Args:
            row: Values in the order of ``_ACTION_COLUMNS``.

        Returns:
            The action with decoded ``arguments`` and ``result``.
        """
        return {
            "id": row[0],
            "student_id": row[1],
            "tool_name": row[2],
            "arguments": json.loads(row[3]),
            "arguments_hash": row[4],
            "reason": row[5],
            "proposed_by": row[6],
            "status": row[7],
            "result": json.loads(row[8]) if row[8] else None,
            "created_at": row[9],
            "expires_at": row[10],
            "decided_at": row[11],
        }

    # Saved location: stored only after the student explicitly opts in.

    def save_location(
        self, student_id: str, label: str, latitude: float, longitude: float
    ) -> None:
        """Save or replace the student's opted-in home location.

        Args:
            student_id: Account the location belongs to.
            label: Human-readable place name.
            latitude: Latitude in degrees.
            longitude: Longitude in degrees.
        """
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO saved_locations (
                    student_id, label, latitude, longitude, consented_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(student_id) DO UPDATE SET
                    label=excluded.label,
                    latitude=excluded.latitude,
                    longitude=excluded.longitude,
                    consented_at=excluded.consented_at
                """,
                (student_id, label, latitude, longitude, now()),
            )

    def get_location(self, student_id: str) -> dict[str, object] | None:
        """Return the student's saved location, or ``None`` if none was saved.

        Args:
            student_id: Account whose location is read.

        Returns:
            ``label``, ``latitude``, ``longitude`` and ``consented_at``.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT label, latitude, longitude, consented_at "
                "FROM saved_locations WHERE student_id=?",
                (student_id,),
            ).fetchone()
        if row is None:
            return None
        keys = ("label", "latitude", "longitude", "consented_at")
        return dict(zip(keys, row, strict=True))

    def delete_location(self, student_id: str) -> bool:
        """Forget the student's saved location.

        Args:
            student_id: Account whose location is deleted.

        Returns:
            ``True`` if a location was deleted.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM saved_locations WHERE student_id=?", (student_id,)
            )
        return cursor.rowcount == 1

    def save_calendar_event(
        self,
        student_id: str,
        plan_id: str,
        week_number: int,
        workout_index: int,
        event_id: str,
        start_at: str,
        end_at: str,
    ) -> None:
        """Link a workout to its calendar event; a workout has at most one."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO calendar_events (
                    event_id, student_id, plan_id, week_number, workout_index,
                    start_at, end_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    student_id,
                    plan_id,
                    week_number,
                    workout_index,
                    start_at,
                    end_at,
                    now(),
                ),
            )

    def list_calendar_events(
        self, student_id: str, plan_id: str
    ) -> list[dict[str, object]]:
        """Return the calendar events linked to one student's plan."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, week_number, workout_index, start_at, end_at
                FROM calendar_events
                WHERE student_id=? AND plan_id=?
                ORDER BY start_at
                """,
                (student_id, plan_id),
            ).fetchall()
        keys = ("event_id", "week_number", "workout_index", "start_at", "end_at")
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def delete_plan(self, student_id: str, plan_id: str) -> bool:
        """Delete a saved plan only when it belongs to the given student.

        Args:
            student_id: Internal account ID that must own the plan.
            plan_id: ID of the saved plan to delete.

        Returns:
            ``True`` if one owned plan was deleted, otherwise ``False``.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM training_plans WHERE id=? AND student_id=?",
                (plan_id, student_id),
            )
        logger.info("Database plan deleted found=%s", cursor.rowcount == 1)
        return cursor.rowcount == 1

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Provide a connection that commits on success and rolls back on failure.

        Returns:
            A context manager yielding a SQLite connection with foreign-key
            enforcement enabled. The connection is always closed on exit.

        Notes:
            Exceptions from the database operation are rolled back and re-raised.
        """
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except BaseException as error:
            logger.error(
                "SQLite operation failed error_type=%s",
                type(error).__name__,
            )
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
