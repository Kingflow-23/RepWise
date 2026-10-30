"""SQLite storage for student profiles and completed plans."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from ..schemas.models import PlanRequest, PlanResult, TrainingPlan, UserProfile


def now() -> str:
    """Return a consistent UTC timestamp."""
    return datetime.now(UTC).isoformat()


class StudentRepository:
    """Keep only the data needed to personalise and revisit generated plans."""

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)

    def initialize(self) -> None:
        """Create tables and migrate username-owned rows to generated account IDs."""
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    full_name TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            user_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(users)")
            }
            if "username" in user_columns and "full_name" not in user_columns:
                connection.execute(
                    "ALTER TABLE users RENAME COLUMN username TO full_name"
                )
                user_columns.remove("username")
                user_columns.add("full_name")
            if "user_id" not in user_columns:
                connection.execute("ALTER TABLE users ADD COLUMN user_id TEXT")
            for (full_name,) in connection.execute(
                "SELECT full_name FROM users WHERE user_id IS NULL OR user_id=''"
            ).fetchall():
                connection.execute(
                    "UPDATE users SET user_id=? WHERE full_name=?",
                    (str(uuid.uuid4()), full_name),
                )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS users_by_id ON users(user_id)"
            )
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS student_profiles (
                    student_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(student_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS training_plans (
                    id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(student_id) REFERENCES student_profiles(student_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS plans_by_student ON training_plans(student_id, created_at DESC);
            """)
            profile_foreign_keys = connection.execute(
                "PRAGMA foreign_key_list(student_profiles)"
            ).fetchall()
            if not any(
                row[2] == "users" and row[3] == "student_id" and row[4] == "user_id"
                for row in profile_foreign_keys
            ):
                connection.commit()
                self._migrate_profile_owner_ids(connection)

    def _migrate_profile_owner_ids(self, connection: sqlite3.Connection) -> None:
        """Preserve old account data while replacing username ownership with user IDs."""
        orphan_profiles = connection.execute("""
            SELECT COUNT(*) FROM student_profiles p
            LEFT JOIN users u ON u.full_name=p.student_id
            WHERE u.user_id IS NULL
            """).fetchone()[0]
        orphan_plans = connection.execute("""
            SELECT COUNT(*) FROM training_plans t
            LEFT JOIN student_profiles p ON p.student_id=t.student_id
            WHERE p.student_id IS NULL
            """).fetchone()[0]
        if orphan_profiles or orphan_plans:
            raise RuntimeError(
                "Cannot add account ownership constraints: the existing database has "
                f"{orphan_profiles} profiles without users and {orphan_plans} plans "
                "without profiles. Reconcile those rows before restarting RepWise."
            )

        connection.execute("PRAGMA foreign_keys=OFF")
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DROP INDEX IF EXISTS plans_by_student")
            connection.execute(
                "ALTER TABLE training_plans RENAME TO training_plans_legacy"
            )
            connection.execute(
                "ALTER TABLE student_profiles RENAME TO student_profiles_legacy"
            )
            connection.execute("""
                CREATE TABLE student_profiles (
                    student_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(student_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)
            connection.execute("""
                CREATE TABLE training_plans (
                    id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(student_id) REFERENCES student_profiles(student_id) ON DELETE CASCADE
                )
            """)
            connection.execute("""
                INSERT INTO student_profiles(student_id, profile_json, updated_at)
                SELECT u.user_id, p.profile_json, p.updated_at
                FROM student_profiles_legacy p
                JOIN users u ON u.full_name=p.student_id
            """)
            connection.execute("""
                INSERT INTO training_plans(
                    id, student_id, request_json, plan_json, sources_json, created_at
                )
                SELECT t.id, p.student_id, t.request_json, t.plan_json,
                    t.sources_json, t.created_at
                FROM training_plans_legacy t
                JOIN student_profiles_legacy old_p ON old_p.student_id=t.student_id
                JOIN users u ON u.full_name=old_p.student_id
                JOIN student_profiles p ON p.student_id=u.user_id
            """)
            connection.execute("DROP TABLE training_plans_legacy")
            connection.execute("DROP TABLE student_profiles_legacy")
            connection.execute("""
                CREATE INDEX plans_by_student
                ON training_plans(student_id, created_at DESC)
            """)
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise RuntimeError(
                    "Database ownership migration would leave foreign-key violations. "
                    "Reconcile the affected profile and plan rows before restarting RepWise."
                )
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.execute("PRAGMA foreign_keys=ON")

    def create_user(self, full_name: str, password_hash: str) -> str:
        """Create a named account and return its generated internal ID."""
        user_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO users(full_name, user_id, password_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (full_name, user_id, password_hash, now()),
            )
        return user_id

    def get_password_hash(self, full_name: str) -> str | None:
        """Return the stored hash for login verification, if the account exists."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT password_hash FROM users WHERE full_name=?", (full_name,)
            ).fetchone()
        return row[0] if row else None

    def get_account_by_name(self, full_name: str) -> tuple[str, str, str] | None:
        """Return the internal ID, password hash, and stored name for login."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id, password_hash, full_name FROM users WHERE full_name=?",
                (full_name,),
            ).fetchone()
        return tuple(row) if row else None

    def save_profile(self, student_id: str, profile: UserProfile) -> None:
        """Create or replace the profile deliberately supplied by a student."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO student_profiles(student_id, profile_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(student_id) DO UPDATE SET
                    profile_json=excluded.profile_json, updated_at=excluded.updated_at
            """,
                (student_id, profile.model_dump_json(), now()),
            )

    def get_profile(self, student_id: str) -> UserProfile | None:
        """Return one profile, or ``None`` when it has not been saved yet."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT profile_json FROM student_profiles WHERE student_id=?",
                (student_id,),
            ).fetchone()
        return UserProfile.model_validate_json(row[0]) if row else None

    def save_plan(
        self, student_id: str, request: PlanRequest, result: PlanResult
    ) -> str:
        """Save a completed plan, its full request, and source evidence snapshots."""
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
        """List recent plans without exposing data belonging to other students."""
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
        """Update a saved plan only when its owner matches the authenticated account."""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE training_plans SET plan_json=? WHERE id=? AND student_id=?",
                (plan.model_dump_json(), plan_id, student_id),
            )
        return cursor.rowcount == 1

    def delete_plan(self, student_id: str, plan_id: str) -> bool:
        """Delete one saved plan only when its owner matches the account."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM training_plans WHERE id=? AND student_id=?",
                (plan_id, student_id),
            )
        return cursor.rowcount == 1

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield one foreign-key-enabled connection and always close it."""
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
