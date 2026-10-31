"""API endpoint tests with retrieval and generation dependencies stubbed."""

from uuid import UUID

from fastapi.testclient import TestClient

from config.config import AppConfig
from src.db.repository import StudentRepository
import src.api.main as api_module
from src.api.main import create_app
from src.schemas import RetrievedChunk, TrainingPlan


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
    assert health["model"] == "llama3.2"
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
