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

from ..schemas.models import PlanRequest, PlanResult, TrainingPlan, UserProfile

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
        logger.info("SQLite repository initialized")

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
        self, student_id: str, request: PlanRequest, result: PlanResult
    ) -> str:
        """Persist a completed plan with its request and evidence snapshot.

        Args:
            student_id: Internal account ID that owns the plan.
            request: Validated request used to generate the plan.
            result: Completed pipeline result, including the plan and sources.

        Returns:
            The generated UUID string identifying the saved plan.

        Raises:
            ValueError: If ``result`` is not completed or contains no plan.
            sqlite3.IntegrityError: If ``student_id`` has no saved profile.
        """
        if result.status != "completed" or result.plan is None:
            raise ValueError("only completed plans can be saved")
        plan_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO training_plans(id, student_id, request_json, plan_json, sources_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
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
                ),
            )
        return plan_id

    def list_plans(self, student_id: str, limit: int = 20) -> list[dict[str, object]]:
        """List the most recent saved plans for one student.

        Args:
            student_id: Internal account ID whose plans are requested.
            limit: Maximum number of plans to return, from 1 through 100.

        Returns:
            Plan records ordered newest first, each containing its ID, original
            request, plan, sources, and creation timestamp.

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
                SELECT id, request_json, plan_json, sources_json, created_at
                FROM training_plans WHERE student_id=? ORDER BY created_at DESC LIMIT ?
            """,
                (student_id, limit),
            ).fetchall()
        logger.info("Database plan history read count=%s", len(rows))
        return [
            {
                "id": row[0],
                "request": json.loads(row[1]),
                "plan": json.loads(row[2]),
                "sources": json.loads(row[3]),
                "created_at": row[4],
            }
            for row in rows
        ]

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
