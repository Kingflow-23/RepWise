"""Repository tests for account, profile, plan, and schema behavior."""

import sqlite3

import pytest

from src.db.repository import StudentRepository
from src.schemas import (
    PlanRequest,
    PlanResult,
    Source,
    TrainingPlan,
    UserProfile,
)


def profile() -> UserProfile:
    """Build the representative profile shared by repository tests."""
    return UserProfile(
        experience="beginner",
        goal="general strength",
        available_days=["Monday"],
        session_minutes=45,
        equipment=["bodyweight"],
    )


def plan() -> TrainingPlan:
    """Build a valid one-week plan for persistence assertions."""
    return TrainingPlan.model_validate(
        {
            "title": "Starter plan",
            "summary": "A simple plan.",
            "assumptions": ["One workout per week."],
            "safety_note": "Stop if anything hurts.",
            "weeks": [
                {
                    "number": 1,
                    "objective": "Practice technique.",
                    "workouts": [
                        {
                            "day": "Monday",
                            "focus": "Full body",
                            "estimated_minutes": 30,
                            "exercises": [
                                {
                                    "name": "Bodyweight squat",
                                    "sets": 3,
                                    "repetitions": "8-12",
                                    "rest_seconds": 60,
                                    "estimated_minutes": 10,
                                    "coaching_note": "Move slowly.",
                                    "source_ids": ["video-1_p0001"],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )


def test_repository_rolls_back_and_closes_connections_on_failure(tmp_path, monkeypatch):
    """Ensure failed repository transactions roll back and always close connections.

    Args:
        tmp_path: Provide a fresh SQLite database for the transaction test.
        monkeypatch: Track connections created by the repository.
    """
    import src.db.repository as repository_module

    original_connect = sqlite3.connect
    connections = []

    class TrackedConnection(sqlite3.Connection):
        """Record whether a connection's close method was called."""

        closed = False

        def close(self):
            """Mark the connection closed before delegating to SQLite."""
            self.closed = True
            super().close()

    def tracked_connect(database_path):
        """Create and retain a connection so cleanup can be asserted.

        Args:
            database_path: SQLite file or connection target forwarded to the
                original connector.
        """
        connection = original_connect(database_path, factory=TrackedConnection)
        connections.append(connection)
        return connection

    monkeypatch.setattr(repository_module.sqlite3, "connect", tracked_connect)
    repository = StudentRepository(tmp_path / "repwise.sqlite3")
    repository.initialize()
    with pytest.raises(RuntimeError, match="forced failure"):
        with repository._connect() as connection:
            connection.execute(
                "INSERT INTO users(full_name, user_id, password_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("rolled-back", "rollback-id", "hash", "now"),
            )
            raise RuntimeError("forced failure")

    with original_connect(repository.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    assert connections
    assert all(connection.closed for connection in connections)


def test_profiles_and_plans_require_their_account_and_profile(tmp_path):
    """Enforce foreign-key ownership from profiles and plans to their parents.

    Args:
        tmp_path: Provide an isolated database for constraint checks.
    """
    repository = StudentRepository(tmp_path / "repwise.sqlite3")
    repository.initialize()

    with pytest.raises(sqlite3.IntegrityError):
        repository.save_profile("missing-user", profile())

    user_id = repository.create_user("alex", "hash")
    repository.save_profile(user_id, profile())
    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO training_plans(
                    id, student_id, request_json, plan_json, sources_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("orphan", "other-user", "{}", "{}", "[]", "now"),
            )

    profile_foreign_keys = repository._connect()
    with profile_foreign_keys as connection:
        assert any(
            row[2] == "users" and row[4] == "user_id"
            for row in connection.execute("PRAGMA foreign_key_list(student_profiles)")
        )
        assert any(
            row[2] == "student_profiles"
            for row in connection.execute("PRAGMA foreign_key_list(training_plans)")
        )


def test_saved_plan_retains_full_request_plan_and_source_snapshot(tmp_path):
    """Preserve the exact request, plan, and evidence snapshot after later profile edits.

    Args:
        tmp_path: Keep persisted test data in a disposable SQLite database.
    """
    repository = StudentRepository(tmp_path / "repwise.sqlite3")
    repository.initialize()
    user_id = repository.create_user("alex", "hash")
    original_profile = profile()
    repository.save_profile(user_id, original_profile)
    request = PlanRequest(
        profile=original_profile,
        focus="Build a simple routine",
        weeks=1,
        extra_constraints=["Keep the session short"],
    )
    saved_plan = plan()
    source = Source(
        source_id="video-1_p0001",
        title="Training basics",
        url="https://example.com/video?t=12",
        excerpt="Short excerpt",
        chunk_id="video-1_p0001-_p0012",
        passage_ids=["video-1_p0001", "video-1_p0012"],
        video_id="video-1",
        start=12,
        end=72,
        video_url="https://example.com/video",
        start_url="https://example.com/video?t=12",
        evidence_text="Complete normalized evidence text used in the prompt.",
    )
    plan_id = repository.save_plan(
        user_id,
        request,
        PlanResult(
            status="completed",
            message="Plan created.",
            plan=saved_plan,
            sources=[source],
        ),
    )

    repository.save_profile(
        user_id,
        original_profile.model_copy(update={"goal": "a different later goal"}),
    )
    saved = repository.list_plans(user_id)[0]
    latest = repository.get_latest_plan(user_id)

    assert saved["id"] == plan_id
    assert PlanRequest.model_validate(saved["request"]) == request
    assert TrainingPlan.model_validate(saved["plan"]) == saved_plan
    assert Source.model_validate(saved["sources"][0]) == source
    assert saved["sources"][0]["evidence_text"] == source.evidence_text
    assert latest == {"request": request.model_dump(), "plan": saved_plan.model_dump()}


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.5])
def test_list_plans_rejects_invalid_limit_values(tmp_path, limit):
    """Reject limits outside the supported integer range.

    Args:
        tmp_path: Provide an initialized database for the repository call.
        limit: Supply each invalid value to exercise input validation.
    """
    repository = StudentRepository(tmp_path / "repwise.sqlite3")
    repository.initialize()

    with pytest.raises(ValueError, match="between 1 and 100"):
        repository.list_plans("alex", limit=limit)


def test_initialize_rejects_legacy_database_without_migrating_it(tmp_path):
    """Reject the old username-based schema without changing its data.

    Args:
        tmp_path: Hold the legacy database to verify it is left untouched.
    """
    database_path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript("""
            CREATE TABLE users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        connection.execute(
            "INSERT INTO users VALUES (?, ?, ?)", ("alex", "hash", "now")
        )
        connection.commit()
    finally:
        connection.close()

    repository = StudentRepository(database_path)
    with pytest.raises(RuntimeError, match="unsupported schema"):
        repository.initialize()

    with sqlite3.connect(database_path) as unchanged:
        assert unchanged.execute(
            "SELECT username, password_hash FROM users"
        ).fetchone() == ("alex", "hash")


def test_only_completed_results_can_be_saved(tmp_path):
    """Prevent incomplete generation results from entering plan history.

    Args:
        tmp_path: Provide the isolated SQLite database for this check.
    """
    repository = StudentRepository(tmp_path / "repwise.sqlite3")
    repository.initialize()
    user_id = repository.create_user("alex", "hash")
    repository.save_profile(user_id, profile())

    with pytest.raises(ValueError, match="only completed plans"):
        repository.save_plan(
            user_id,
            PlanRequest(profile=profile(), focus="Build a routine", weeks=1),
            PlanResult(status="generation_error", message="Not completed."),
        )
