from fastapi.testclient import TestClient

from config.config import AppConfig
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


def test_api_saves_profile_and_completed_plan(tmp_path):
    app = create_app(
        AppConfig(database_path=str(tmp_path / "repwise.sqlite3")),
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
    assert client.put("/api/v1/students/alex/profile", json=profile).status_code == 200
    response = client.post(
        "/api/v1/students/alex/plans",
        json={"focus": "Build a simple routine", "weeks": 1},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert len(client.get("/api/v1/students/alex/plans").json()) == 1


def test_api_requires_a_profile_before_creating_a_plan(tmp_path):
    app = create_app(
        AppConfig(database_path=str(tmp_path / "repwise.sqlite3")),
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )
    response = TestClient(app).post(
        "/api/v1/students/unknown/plans",
        json={"focus": "Build a simple routine", "weeks": 1},
    )

    assert response.status_code == 404
