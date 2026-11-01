"""API endpoint tests with retrieval and generation dependencies stubbed."""

import json
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from config.config import AppConfig
from src.db.repository import StudentRepository
import src.api.main as api_module
from src.api.main import create_app
from src.schemas import RetrievedChunk, TrainingPlan
from src.schemas.models import (
    PlanRequest,
    PlanResult,
    Source,
    UserProfile,
    Workout,
)


class FakeRetriever:
    """Provide deterministic retrieved passages for API integration tests."""

    def search(self, query, top_k):
        """Return five chunks with traceable source metadata.

        Args:
            query: Search text; fixed test evidence makes the exact text
                irrelevant to this API contract test.
            top_k: Requested retrieval depth; the fake supplies five chunks to
                satisfy the generation minimum.
        """
        return [
            RetrievedChunk(
                text="Use controlled repetitions and progress gradually.",
                video_id=f"video-{index}",
                chunk_id=f"video-{index}_chunk",
                passage_ids=[f"video-{index}_p001"],
                title="Training basics",
                video_url="https://example.com",
            )
            for index in range(1, 6)
        ]


class FakeGenerator:
    """Provide a valid plan whose evidence quotes the fake retrieved source."""

    def generate(self, system_prompt, user_prompt, response_schema=None):
        """Return a fixed completed plan for API requests.

        Args:
            system_prompt: System instructions supplied by the production
                pipeline; ignored by this deterministic fake.
            user_prompt: Request and evidence prompt; ignored because the plan
                is fixed.
            response_schema: Structured-output schema; accepted to match the
                generator protocol but not interpreted by the fake.
        """
        return TrainingPlan.model_validate(
            {
                "title": "Starter plan",
                "summary": "A simple plan.",
                "assumptions": [],
                "safety_note": "Stop when an exercise causes pain.",
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
                                        "source_ids": ["video-1_p001"],
                                        "supporting_evidence": [
                                            {
                                                "source_id": "video-1_p001",
                                                "quote": "Use controlled repetitions and progress gradually.",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )


class FakeWorkoutGenerator:
    """Return one deterministic workout adjustment or a chosen failure."""

    def __init__(self, mode: str = "success"):
        self.mode = mode
        self.calls: list[dict[str, object]] = []

    def generate_workout(self, system_prompt, user_prompt, response_schema=None):
        context = json.loads(user_prompt)
        self.calls.append(context)
        if self.mode == "failure":
            raise RuntimeError("mock model failure")
        target = Workout.model_validate(context["target_workout"])
        if self.mode == "invalid":
            exercises = [
                target.exercises[0].model_copy(update={"name": "New exercise"}),
                *target.exercises[1:],
            ]
        elif self.mode == "unchanged":
            exercises = target.exercises
        else:
            exercises = [
                target.exercises[0].model_copy(
                    update={"sets": target.exercises[0].sets + 1}
                ),
                *target.exercises[1:],
            ]
        return target.model_copy(update={"exercises": exercises})


def save_three_day_plan(repository: StudentRepository, student_id: str) -> str:
    profile = UserProfile(
        experience="beginner",
        goal="general strength",
        available_days=["Monday", "Wednesday", "Friday"],
        session_minutes=45,
        equipment=["bodyweight"],
    )
    repository.save_profile(student_id, profile)
    request = PlanRequest(
        profile=profile,
        focus="Build a simple routine",
        workouts_per_week=3,
    )
    plan = TrainingPlan.model_validate(
        {
            "title": "Three-day plan",
            "summary": "A simple plan.",
            "assumptions": [],
            "safety_note": "Stop if anything hurts.",
            "weeks": [
                {
                    "number": 1,
                    "objective": "Practice technique.",
                    "workouts": [
                        {
                            "day": day,
                            "focus": "Full body",
                            "estimated_minutes": 30,
                            "exercises": [
                                {
                                    "name": f"Bodyweight squat {day}",
                                    "sets": 3,
                                    "repetitions": "8-12",
                                    "rest_seconds": 60,
                                    "estimated_minutes": 10,
                                    "coaching_note": "Move slowly.",
                                    "source_ids": ["video-1_p001"],
                                }
                            ],
                        }
                        for day in ("Monday", "Wednesday", "Friday")
                    ],
                }
            ],
        }
    )
    return repository.save_plan(
        student_id,
        request,
        PlanResult(
            status="completed",
            message="Created.",
            plan=plan,
            sources=[
                Source(
                    source_id="video-1_p001",
                    title="Training basics",
                    url="https://example.com",
                    excerpt="Controlled repetitions and gradual progress.",
                )
            ],
        ),
    )


def test_api_saves_profile_and_completed_plan(tmp_path, monkeypatch):
    """Exercise profile creation, plan persistence, ownership, and deletion.

    Args:
        tmp_path: Isolate the API's SQLite database for this test.
        monkeypatch: Replace token decoding so the test can use a simple token.
    """
    monkeypatch.setattr(api_module, "read_access_token", lambda token, secret: token)
    repository = StudentRepository(str(tmp_path / "repwise.sqlite3"))
    repository.initialize()
    user_id = repository.create_user("Alex Example", "test-hash")
    app = create_app(
        AppConfig(
            database_path=str(tmp_path / "repwise.sqlite3"), auth_secret="test-secret"
        ),
        repository=repository,
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )
    client = TestClient(app)
    health_response = client.get("/api/v1/health")
    health = health_response.json()
    assert health["retriever"] == "hybrid"
    assert health["model"] == "qwen3.5:9b"
    assert UUID(health_response.headers["X-Request-ID"])
    profile = {
        "experience": "beginner",
        "goal": "general strength",
        "available_days": ["Monday"],
        "session_minutes": 45,
        "equipment": ["bodyweight"],
    }
    headers = {"Authorization": f"Bearer {user_id}"}
    assert (
        client.put(
            f"/api/v1/students/{user_id}/profile", json=profile, headers=headers
        ).status_code
        == 200
    )
    response = client.post(
        f"/api/v1/students/{user_id}/plans",
        json={"focus": "Build a simple routine"},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    records = client.get(f"/api/v1/students/{user_id}/plans", headers=headers).json()
    assert len(records) == 1
    assert records[0]["request"]["focus"] == "Build a simple routine"
    assert records[0]["request"]["weeks"] == 1
    assert records[0]["request"]["profile"]["goal"] == "general strength"
    assert records[0]["sources"][0]["evidence_text"] == (
        "Use controlled repetitions and progress gradually."
    )
    plan_id = records[0]["id"]
    saved_plan = records[0]["plan"]
    assert (
        client.put(
            f"/api/v1/students/{user_id}/plans/{plan_id}",
            json=saved_plan,
            headers=headers,
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/api/v1/students/alex/plans", headers={"Authorization": "Bearer another"}
        ).status_code
        == 403
    )
    assert (
        client.delete(
            f"/api/v1/students/{user_id}/plans/{plan_id}", headers=headers
        ).status_code
        == 204
    )
    assert client.get(f"/api/v1/students/{user_id}/plans", headers=headers).json() == []


def test_next_plan_uses_latest_saved_plan_and_requested_refinements(
    tmp_path, monkeypatch
):
    """Pass the latest saved plan and current user guidance into the next prompt."""
    monkeypatch.setattr(api_module, "read_access_token", lambda token, secret: token)

    class PromptRecordingGenerator(FakeGenerator):
        prompts = []

        def generate(self, system_prompt, user_prompt, response_schema=None):
            self.prompts.append(user_prompt)
            return super().generate(system_prompt, user_prompt, response_schema)

    generator = PromptRecordingGenerator()
    repository = StudentRepository(str(tmp_path / "repwise.sqlite3"))
    repository.initialize()
    user_id = repository.create_user("Alex Example", "test-hash")
    app = create_app(
        AppConfig(
            database_path=str(tmp_path / "repwise.sqlite3"), auth_secret="test-secret"
        ),
        repository=repository,
        retriever=FakeRetriever(),
        generator=generator,
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {user_id}"}
    profile = {
        "experience": "beginner",
        "goal": "general strength",
        "available_days": ["Monday"],
        "session_minutes": 45,
        "equipment": ["bodyweight"],
    }
    client.put(f"/api/v1/students/{user_id}/profile", json=profile, headers=headers)

    first = client.post(
        f"/api/v1/students/{user_id}/plans",
        json={"focus": "Build a simple routine"},
        headers=headers,
    )
    second = client.post(
        f"/api/v1/students/{user_id}/plans",
        json={
            "focus": "Improve my routine",
            "refinement_instructions": "Use more variety this time.",
        },
        headers=headers,
    )

    assert first.status_code == second.status_code == 200
    assert "PREVIOUS PLAN (COMPACT REFERENCE)" not in generator.prompts[0]
    assert "Starter plan" in generator.prompts[1]
    assert "Bodyweight squat" in generator.prompts[1]
    assert "Use more variety this time." in generator.prompts[1]
    assert (
        client.get(f"/api/v1/students/{user_id}/plans", headers=headers).json()[0][
            "request"
        ]["refinement_instructions"]
        == "Use more variety this time."
    )


def test_api_requires_a_profile_before_creating_a_plan(tmp_path, monkeypatch):
    """Reject plan creation when the account has no saved profile.

    Args:
        tmp_path: Provide an isolated database path for the application.
        monkeypatch: Replace token decoding for the authenticated request.
    """
    monkeypatch.setattr(api_module, "read_access_token", lambda token, secret: token)
    app = create_app(
        AppConfig(
            database_path=str(tmp_path / "repwise.sqlite3"), auth_secret="test-secret"
        ),
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )
    response = TestClient(app).post(
        "/api/v1/students/unknown/plans",
        json={"focus": "Build a simple routine"},
        headers={"Authorization": "Bearer unknown"},
    )

    assert response.status_code == 404


def test_api_rejects_more_weekly_workouts_than_profile_days(tmp_path, monkeypatch):
    """Reject workout counts that exceed the profile's available days.

    Args:
        tmp_path: Keep the test account and profile in a temporary database.
        monkeypatch: Replace token decoding for the authenticated request.
    """
    monkeypatch.setattr(api_module, "read_access_token", lambda token, secret: token)
    repository = StudentRepository(str(tmp_path / "repwise.sqlite3"))
    repository.initialize()
    user_id = repository.create_user("Alex Example", "test-hash")
    repository.save_profile(
        user_id,
        api_module.UserProfile(
            experience="beginner",
            goal="general strength",
            available_days=["Monday"],
            session_minutes=45,
            equipment=["bodyweight"],
        ),
    )
    app = create_app(
        AppConfig(
            database_path=str(tmp_path / "repwise.sqlite3"), auth_secret="test-secret"
        ),
        repository=repository,
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    response = TestClient(app).post(
        f"/api/v1/students/{user_id}/plans",
        json={
            "focus": "Build a simple routine",
            "workouts_per_week": 2,
        },
        headers={"Authorization": f"Bearer {user_id}"},
    )

    assert response.status_code == 422
    assert "cannot exceed" in response.json()["detail"]


def test_api_does_not_allow_selecting_plan_duration(tmp_path, monkeypatch):
    """Keep the plan endpoint's duration fixed rather than client-selectable.

    Args:
        tmp_path: Isolate the application database used by the endpoint.
        monkeypatch: Replace token decoding so the request reaches validation.
    """
    monkeypatch.setattr(api_module, "read_access_token", lambda token, secret: token)
    app = create_app(
        AppConfig(
            database_path=str(tmp_path / "repwise.sqlite3"), auth_secret="test-secret"
        ),
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    response = TestClient(app).post(
        "/api/v1/students/unknown/plans",
        json={"focus": "Build a simple routine", "weeks": 4},
        headers={"Authorization": "Bearer some-user-id"},
    )

    assert response.status_code == 422


def test_api_requires_login_and_blocks_cross_user_access(tmp_path, monkeypatch):
    """Require authentication and deny access to another user's plans.

    Args:
        tmp_path: Supply the database location for the test application.
        monkeypatch: Make the test token resolve to a different owner ID.
    """
    monkeypatch.setattr(api_module, "read_access_token", lambda token, secret: token)
    app = create_app(
        AppConfig(
            database_path=str(tmp_path / "repwise.sqlite3"), auth_secret="test-secret"
        ),
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )
    client = TestClient(app)
    assert client.get("/api/v1/students/alex/plans").status_code == 401
    response = client.get(
        "/api/v1/students/alex/plans",
        headers={"Authorization": "Bearer another-user"},
    )
    assert response.status_code == 403


def test_registration_hashes_password_and_login_checks_credentials(
    tmp_path, monkeypatch
):
    """Check password storage, successful login, invalid credentials, and duplicates.

    Args:
        tmp_path: Store test accounts in a disposable SQLite database.
        monkeypatch: Stub password and token helpers to make auth assertions stable.
    """
    monkeypatch.setattr(
        api_module, "hash_password", lambda password: f"argon2-hash:{password}"
    )
    monkeypatch.setattr(
        api_module,
        "verify_password",
        lambda password, saved_hash: saved_hash == f"argon2-hash:{password}",
    )
    monkeypatch.setattr(
        api_module, "create_access_token", lambda user_id, secret: f"signed:{user_id}"
    )
    repository = StudentRepository(str(tmp_path / "repwise.sqlite3"))
    app = create_app(
        AppConfig(
            database_path=str(tmp_path / "repwise.sqlite3"), auth_secret="test-secret"
        ),
        repository=repository,
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )
    client = TestClient(app)
    credentials = {"full_name": "Alex Example", "password": "correct-horse"}
    registered = client.post("/api/v1/auth/register", json=credentials)
    assert registered.status_code == 200
    assert registered.json()["token_type"] == "bearer"
    user_id = registered.json()["user_id"]
    assert registered.json()["full_name"] == credentials["full_name"]
    assert UUID(user_id).version == 4
    assert user_id != credentials["full_name"]
    assert registered.json()["access_token"] == f"signed:{user_id}"
    assert repository.get_password_hash("Alex Example") != credentials["password"]
    signed_in = client.post("/api/v1/auth/login", json=credentials)
    assert signed_in.status_code == 200
    assert signed_in.json()["user_id"] == user_id
    invalid = client.post(
        "/api/v1/auth/login",
        json={"full_name": "Alex Example", "password": "wrong-password"},
    )
    assert invalid.status_code == 401
    assert client.post("/api/v1/auth/register", json=credentials).status_code == 409


def test_workout_feedback_adapts_only_next_unlogged_workout(tmp_path, monkeypatch):
    """Persist feedback, enforce ownership, and modify only successive targets."""
    monkeypatch.setattr(api_module, "read_access_token", lambda token, secret: token)
    repository = StudentRepository(tmp_path / "repwise.sqlite3")
    repository.initialize()
    user_id = repository.create_user("Alex Example", "test-hash")
    other_user_id = repository.create_user("Other Example", "test-hash")
    plan_id = save_three_day_plan(repository, user_id)
    workout_generator = FakeWorkoutGenerator()
    app = create_app(
        AppConfig(
            database_path=str(tmp_path / "repwise.sqlite3"), auth_secret="test-secret"
        ),
        repository=repository,
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        workout_generator=workout_generator,
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {user_id}"}
    route = f"/api/v1/students/{user_id}/plans/{plan_id}/weeks/1/workouts"

    assert (
        client.post(
            f"/api/v1/students/{other_user_id}/plans/{plan_id}/weeks/1"
            "/workouts/0/feedback",
            json={
                "completion_status": "completed",
                "difficulty": "just_right",
            },
            headers=headers,
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"{route}/3/feedback",
            json={
                "completion_status": "completed",
                "difficulty": "just_right",
            },
            headers=headers,
        ).status_code
        == 404
    )

    initial_plan = repository.get_plan(user_id, plan_id)["plan"]
    first = client.post(
        f"{route}/0/feedback",
        json={
            "completion_status": "partially_completed",
            "difficulty": "slightly_too_hard",
            "comment": "The last set was difficult.",
        },
        headers=headers,
    )
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["adapted"] is True
    assert first_data["adapted_workout"] == {
        "week_number": 1,
        "workout_index": 1,
    }
    first_plan = first_data["plan"]
    assert (
        first_plan["weeks"][0]["workouts"][0] == initial_plan["weeks"][0]["workouts"][0]
    )
    assert (
        first_plan["weeks"][0]["workouts"][1]["exercises"][0]["sets"]
        == initial_plan["weeks"][0]["workouts"][1]["exercises"][0]["sets"] + 1
    )
    assert (
        first_plan["weeks"][0]["workouts"][2] == initial_plan["weeks"][0]["workouts"][2]
    )
    assert first_data["workout_feedback"][0]["completion_status"] == (
        "partially_completed"
    )
    assert first_data["workout_feedback"][0]["adapted_workout_index"] == 1

    second = client.post(
        f"{route}/1/feedback",
        json={
            "completion_status": "completed",
            "difficulty": "just_right",
        },
        headers=headers,
    )
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["adapted_workout"] == {
        "week_number": 1,
        "workout_index": 2,
    }
    assert (
        second_data["plan"]["weeks"][0]["workouts"][0]
        == first_plan["weeks"][0]["workouts"][0]
    )
    assert (
        second_data["plan"]["weeks"][0]["workouts"][1]
        == first_plan["weeks"][0]["workouts"][1]
    )
    assert (
        second_data["plan"]["weeks"][0]["workouts"][2]["exercises"][0]["sets"]
        == initial_plan["weeks"][0]["workouts"][2]["exercises"][0]["sets"] + 1
    )
    assert [call["target_workout"]["day"] for call in workout_generator.calls] == [
        "Wednesday",
        "Friday",
    ]
    assert len(repository.list_workout_feedback(user_id, plan_id)) == 2


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("failure", "left unchanged"),
        ("invalid", "left unchanged"),
        ("unchanged", "did not need changes"),
    ],
)
def test_workout_adaptation_nonchange_preserves_saved_plan(
    tmp_path, monkeypatch, mode, message
):
    """Keep the original plan when the model fails or changes exercise identity."""
    monkeypatch.setattr(api_module, "read_access_token", lambda token, secret: token)
    repository = StudentRepository(tmp_path / "repwise.sqlite3")
    repository.initialize()
    user_id = repository.create_user("Alex Example", "test-hash")
    plan_id = save_three_day_plan(repository, user_id)
    original_plan = repository.get_plan(user_id, plan_id)["plan"]
    app = create_app(
        AppConfig(
            database_path=str(tmp_path / "repwise.sqlite3"), auth_secret="test-secret"
        ),
        repository=repository,
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        workout_generator=FakeWorkoutGenerator(mode),
    )

    response = TestClient(app).post(
        f"/api/v1/students/{user_id}/plans/{plan_id}/weeks/1" "/workouts/0/feedback",
        json={
            "completion_status": "completed",
            "difficulty": "slightly_too_easy",
        },
        headers={"Authorization": f"Bearer {user_id}"},
    )

    assert response.status_code == 200
    assert response.json()["adapted"] is False
    assert message in response.json()["message"]
    assert repository.get_plan(user_id, plan_id)["plan"] == original_plan
    assert len(repository.list_workout_feedback(user_id, plan_id)) == 1


def test_pain_feedback_is_saved_without_automatic_adaptation(tmp_path, monkeypatch):
    """Keep the plan unchanged and do not call the model for pain reports."""
    monkeypatch.setattr(api_module, "read_access_token", lambda token, secret: token)
    repository = StudentRepository(tmp_path / "repwise.sqlite3")
    repository.initialize()
    user_id = repository.create_user("Alex Example", "test-hash")
    plan_id = save_three_day_plan(repository, user_id)
    original_plan = repository.get_plan(user_id, plan_id)["plan"]
    workout_generator = FakeWorkoutGenerator()
    app = create_app(
        AppConfig(
            database_path=str(tmp_path / "repwise.sqlite3"), auth_secret="test-secret"
        ),
        repository=repository,
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        workout_generator=workout_generator,
    )

    response = TestClient(app).post(
        f"/api/v1/students/{user_id}/plans/{plan_id}/weeks/1" "/workouts/0/feedback",
        json={
            "completion_status": "partially_completed",
            "difficulty": "much_too_hard",
            "comment": "I felt pain in my shoulder.",
        },
        headers={"Authorization": f"Bearer {user_id}"},
    )

    assert response.status_code == 200
    assert response.json()["adapted"] is False
    assert "mentions pain" in response.json()["message"]
    assert workout_generator.calls == []
    assert repository.get_plan(user_id, plan_id)["plan"] == original_plan
    assert len(repository.list_workout_feedback(user_id, plan_id)) == 1
