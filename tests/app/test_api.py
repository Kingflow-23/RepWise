from fastapi.testclient import TestClient

from config.config import AppConfig
from src.db.repository import StudentRepository
import src.api.main as api_module
from src.api.main import create_app
from src.schemas import RetrievedChunk, TrainingPlan


class FakeRetriever:
    def search(self, query, top_k):
        return [
            RetrievedChunk(
                text="Use controlled repetitions and progress gradually.",
                video_id="video-1",
                passage_ids=["video-1_p001"],
                title="Training basics",
                video_url="https://example.com",
            )
        ]


class FakeGenerator:
    def generate(self, system_prompt, user_prompt):
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
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )


def test_api_saves_profile_and_completed_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "read_access_token", lambda token, secret: token)
    repository = StudentRepository(str(tmp_path / "repwise.sqlite3"))
    repository.initialize()
    repository.create_user("alex", "test-hash")
    app = create_app(
        AppConfig(
            database_path=str(tmp_path / "repwise.sqlite3"), auth_secret="test-secret"
        ),
        repository=repository,
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )
    client = TestClient(app)
    profile = {
        "experience": "beginner",
        "goal": "general strength",
        "available_days": ["Monday"],
        "session_minutes": 45,
        "equipment": ["bodyweight"],
    }
    headers = {"Authorization": "Bearer alex"}
    assert (
        client.put(
            "/api/v1/students/alex/profile", json=profile, headers=headers
        ).status_code
        == 200
    )
    response = client.post(
        "/api/v1/students/alex/plans",
        json={"focus": "Build a simple routine", "weeks": 1},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    records = client.get("/api/v1/students/alex/plans", headers=headers).json()
    assert len(records) == 1
    assert records[0]["request"]["focus"] == "Build a simple routine"
    assert records[0]["request"]["profile"]["goal"] == "general strength"
    assert records[0]["sources"][0]["evidence_text"] == (
        "Use controlled repetitions and progress gradually."
    )
    plan_id = records[0]["id"]
    saved_plan = records[0]["plan"]
    assert (
        client.put(
            f"/api/v1/students/alex/plans/{plan_id}", json=saved_plan, headers=headers
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
            f"/api/v1/students/alex/plans/{plan_id}", headers=headers
        ).status_code
        == 204
    )
    assert client.get("/api/v1/students/alex/plans", headers=headers).json() == []


def test_api_requires_a_profile_before_creating_a_plan(tmp_path, monkeypatch):
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
        json={"focus": "Build a simple routine", "weeks": 1},
        headers={"Authorization": "Bearer unknown"},
    )

    assert response.status_code == 404


def test_api_requires_login_and_blocks_cross_user_access(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        api_module, "hash_password", lambda password: f"argon2-hash:{password}"
    )
    monkeypatch.setattr(
        api_module,
        "verify_password",
        lambda password, saved_hash: saved_hash == f"argon2-hash:{password}",
    )
    monkeypatch.setattr(
        api_module, "create_access_token", lambda username, secret: f"signed:{username}"
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
    credentials = {"username": "alex", "password": "correct-horse"}
    registered = client.post("/api/v1/auth/register", json=credentials)
    assert registered.status_code == 200
    assert registered.json()["token_type"] == "bearer"
    assert repository.get_password_hash("alex") != credentials["password"]
    assert client.post("/api/v1/auth/login", json=credentials).status_code == 200
    invalid = client.post(
        "/api/v1/auth/login", json={"username": "alex", "password": "wrong-password"}
    )
    assert invalid.status_code == 401
