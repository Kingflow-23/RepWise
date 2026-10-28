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
    return UserProfile(
        experience="beginner",
        goal="general strength",
        available_days=["Monday"],
        session_minutes=45,
        equipment=["bodyweight"],
    )


def plan() -> TrainingPlan:
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
    import src.db.repository as repository_module

    original_connect = sqlite3.connect
    connections = []

    class TrackedConnection(sqlite3.Connection):
        closed = False

        def close(self):
            self.closed = True
            super().close()

    def tracked_connect(database_path):
        connection = original_connect(database_path, factory=TrackedConnection)
        connections.append(connection)
        return connection

    monkeypatch.setattr(repository_module.sqlite3, "connect", tracked_connect)
    repository = StudentRepository(tmp_path / "repwise.sqlite3")
    repository.initialize()
    with pytest.raises(RuntimeError, match="forced failure"):
        with repository._connect() as connection:
            connection.execute(
                "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
                ("rolled-back", "hash", "now"),
            )
            raise RuntimeError("forced failure")

    with original_connect(repository.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    assert connections
    assert all(connection.closed for connection in connections)


def test_profiles_and_plans_require_their_account_and_profile(tmp_path):
    repository = StudentRepository(tmp_path / "repwise.sqlite3")
    repository.initialize()

    with pytest.raises(sqlite3.IntegrityError):
        repository.save_profile("missing-user", profile())

    repository.create_user("alex", "hash")
    repository.save_profile("alex", profile())
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
            row[2] == "users"
            for row in connection.execute("PRAGMA foreign_key_list(student_profiles)")
        )
        assert any(
            row[2] == "student_profiles"
            for row in connection.execute("PRAGMA foreign_key_list(training_plans)")
        )


def test_saved_plan_retains_full_request_plan_and_source_snapshot(tmp_path):
    repository = StudentRepository(tmp_path / "repwise.sqlite3")
    repository.initialize()
    repository.create_user("alex", "hash")
    original_profile = profile()
    repository.save_profile("alex", original_profile)
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
        "alex",
        request,
        PlanResult(
            status="completed",
            message="Plan created.",
            plan=saved_plan,
            sources=[source],
        ),
    )

    repository.save_profile(
        "alex",
        original_profile.model_copy(update={"goal": "a different later goal"}),
    )
    saved = repository.list_plans("alex")[0]

    assert saved["id"] == plan_id
    assert PlanRequest.model_validate(saved["request"]) == request
    assert TrainingPlan.model_validate(saved["plan"]) == saved_plan
    assert Source.model_validate(saved["sources"][0]) == source
    assert saved["sources"][0]["evidence_text"] == source.evidence_text


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.5])
def test_list_plans_rejects_invalid_limit_values(tmp_path, limit):
    repository = StudentRepository(tmp_path / "repwise.sqlite3")
    repository.initialize()

    with pytest.raises(ValueError, match="between 1 and 100"):
        repository.list_plans("alex", limit=limit)


def test_initialize_migrates_existing_profile_and_plan_tables(tmp_path):
    database_path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript("""
            CREATE TABLE users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE student_profiles (
                student_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE training_plans (
                id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                request_json TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES student_profiles(student_id) ON DELETE CASCADE
            );
            CREATE INDEX plans_by_student ON training_plans(student_id, created_at DESC);
        """)
        connection.execute(
            "INSERT INTO users VALUES (?, ?, ?)", ("alex", "hash", "now")
        )
        connection.execute(
            "INSERT INTO student_profiles VALUES (?, ?, ?)",
            ("alex", profile().model_dump_json(), "now"),
        )
        connection.execute(
            "INSERT INTO training_plans VALUES (?, ?, ?, ?, ?, ?)",
            ("plan-1", "alex", "{}", "{}", "[]", "now"),
        )
        connection.commit()
    finally:
        connection.close()

    repository = StudentRepository(database_path)
    repository.initialize()

    with repository._connect() as migrated:
        profile_fks = migrated.execute(
            "PRAGMA foreign_key_list(student_profiles)"
        ).fetchall()
        plan_fks = migrated.execute(
            "PRAGMA foreign_key_list(training_plans)"
        ).fetchall()
        assert any(row[2] == "users" for row in profile_fks)
        assert any(row[2] == "student_profiles" for row in plan_fks)
        assert (
            migrated.execute(
                "SELECT COUNT(*) FROM training_plans WHERE id='plan-1'"
            ).fetchone()[0]
            == 1
        )
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []


def test_only_completed_results_can_be_saved(tmp_path):
    repository = StudentRepository(tmp_path / "repwise.sqlite3")
    repository.initialize()
    repository.create_user("alex", "hash")
    repository.save_profile("alex", profile())

    with pytest.raises(ValueError, match="only completed plans"):
        repository.save_plan(
            "alex",
            PlanRequest(profile=profile(), focus="Build a routine", weeks=1),
            PlanResult(status="generation_error", message="Not completed."),
        )
