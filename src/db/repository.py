"""SQLite storage for student profiles and completed plans."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ..schemas.models import PlanOptions, PlanResult, TrainingPlan, UserProfile


def now() -> str:
    """Return a consistent UTC timestamp."""
    return datetime.now(UTC).isoformat()


class StudentRepository:
    """Keep only the data needed to personalise and revisit generated plans."""

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)

    def initialize(self) -> None:
        """Create tables explicitly when the application starts."""
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS student_profiles (
                    student_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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

    def create_user(self, username: str, password_hash: str) -> None:
        """Create one account; uniqueness is enforced by SQLite."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, now()),
            )

    def get_password_hash(self, username: str) -> str | None:
        """Return the stored hash for login verification, if the account exists."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT password_hash FROM users WHERE username=?", (username,)
            ).fetchone()
        return row[0] if row else None

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
        self, student_id: str, options: PlanOptions, result: PlanResult
    ) -> str:
        """Save a completed plan and the sources that were returned with it."""
        if result.plan is None:
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
                    options.model_dump_json(),
                    result.plan.model_dump_json(),
                    json.dumps([source.model_dump() for source in result.sources]),
                    now(),
                ),
            )
        return plan_id

    def list_plans(self, student_id: str, limit: int = 20) -> list[dict[str, object]]:
        """List recent plans without exposing data belonging to other students."""
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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
