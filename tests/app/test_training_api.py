"""Performance and history routes: same rules as the agents' tools.

Temporary database, fake retriever and generator; no model is called.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_module
from config.config import AppConfig
from src.agents.health import FixedHealthChecker
from src.api.main import create_app
from src.db.repository import StudentRepository
from src.retrieval.query_expansion import QueryAnalysis
from tests.app.test_api import FakeGenerator, FakeRetriever

PROFILE = {
    "experience": "beginner",
    "goal": "general strength",
    "available_days": ["Monday"],
    "session_minutes": 45,
    "equipment": ["bodyweight"],
}


class InScopeAnalyzer:
    def analyze(self, query):
        return QueryAnalysis("in_scope", query, "")


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Two accounts, each with a profile and one generated plan."""
    monkeypatch.setattr(api_module, "OllamaQueryExpander", lambda *_a: InScopeAnalyzer())
    monkeypatch.setattr(api_module, "read_access_token", lambda token, secret: token)
    database = str(tmp_path / "training.sqlite3")
    repository = StudentRepository(database)
    repository.initialize()
    app = create_app(
        AppConfig(database_path=database, auth_secret="test-secret"),
        repository=repository,
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
        health_checker=FixedHealthChecker(("pain",)),
        ollama_models=lambda: [],
    )
    client = TestClient(app)
    accounts = {}
    for name in ("Alex Example", "Sam Other"):
        user_id = repository.create_user(name, "test-hash")
        headers = {"Authorization": f"Bearer {user_id}"}
        assert client.put(f"/api/v1/students/{user_id}/profile", json=PROFILE, headers=headers).status_code == 200
        plan = client.post(f"/api/v1/students/{user_id}/plans", json={"focus": "Build a simple routine"}, headers=headers)
        assert plan.status_code == 200
        accounts[name] = (user_id, headers, plan.json()["plan_id"])
    return client, accounts


def workout_url(user_id, plan_id, week=1, index=0):
    return f"/api/v1/students/{user_id}/plans/{plan_id}/weeks/{week}/workouts/{index}/performance"


def test_saved_performance_is_read_back_with_the_plan_exercise_name(api):
    client, accounts = api
    user_id, headers, plan_id = accounts["Alex Example"]
    assert client.get(workout_url(user_id, plan_id), headers=headers).json() == []

    response = client.post(
        workout_url(user_id, plan_id),
        json={"exercises": [{"exercise_name": " bodyweight SQUAT ", "sets_completed": 3, "reps_per_set": "10, 10, 8", "weight_kg": None}]},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["records"][0]["exercise_name"] == "Bodyweight squat"
    saved = client.get(workout_url(user_id, plan_id), headers=headers).json()
    assert saved == [{"exercise_name": "Bodyweight squat", "sets_completed": 3, "reps_per_set": "10, 10, 8", "weight_kg": None}]


def test_saving_again_replaces_the_exercise_record(api):
    client, accounts = api
    user_id, headers, plan_id = accounts["Alex Example"]
    for sets in (2, 4):
        client.post(
            workout_url(user_id, plan_id),
            json={"exercises": [{"exercise_name": "Bodyweight squat", "sets_completed": sets, "reps_per_set": "12", "weight_kg": 10}]},
            headers=headers,
        )

    saved = client.get(workout_url(user_id, plan_id), headers=headers).json()
    assert [(item["sets_completed"], item["weight_kg"]) for item in saved] == [(4, 10)]


def test_an_unknown_exercise_saves_nothing(api):
    client, accounts = api
    user_id, headers, plan_id = accounts["Alex Example"]

    response = client.post(
        workout_url(user_id, plan_id),
        json={"exercises": [
            {"exercise_name": "Bodyweight squat", "sets_completed": 3, "reps_per_set": "10"},
            {"exercise_name": "Deadlift", "sets_completed": 3, "reps_per_set": "5", "weight_kg": 80},
        ]},
        headers=headers,
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "exercise_not_found"
    assert "Bodyweight squat" in response.json()["detail"]["message"]
    assert client.get(workout_url(user_id, plan_id), headers=headers).json() == []


@pytest.mark.parametrize(
    "body",
    [
        {"exercises": []},
        {"exercises": [{"exercise_name": "Bodyweight squat", "sets_completed": -1, "reps_per_set": "10"}]},
        {"exercises": [{"exercise_name": "Bodyweight squat", "sets_completed": 3, "reps_per_set": "10", "weight_kg": 2000}]},
    ],
)
def test_invalid_values_are_refused(api, body):
    client, accounts = api
    user_id, headers, plan_id = accounts["Alex Example"]

    assert client.post(workout_url(user_id, plan_id), json=body, headers=headers).status_code == 422


def test_unknown_workouts_and_other_accounts_are_refused(api):
    client, accounts = api
    user_id, headers, plan_id = accounts["Alex Example"]
    other_id, _, other_plan = accounts["Sam Other"]
    body = {"exercises": [{"exercise_name": "Bodyweight squat", "sets_completed": 3, "reps_per_set": "10"}]}

    assert client.post(workout_url(user_id, plan_id, week=2), json=body, headers=headers).status_code == 404
    # Another account's plan ID behaves as unknown; another account's path is forbidden.
    assert client.post(workout_url(user_id, other_plan), json=body, headers=headers).status_code == 404
    assert client.get(workout_url(other_id, other_plan), headers=headers).status_code == 403
    assert client.get(f"/api/v1/students/{other_id}/history", headers=headers).status_code == 403


def test_history_includes_logged_performance(api):
    client, accounts = api
    user_id, headers, plan_id = accounts["Alex Example"]
    client.post(
        workout_url(user_id, plan_id),
        json={"exercises": [{"exercise_name": "Bodyweight squat", "sets_completed": 3, "reps_per_set": "10", "weight_kg": 12.5}]},
        headers=headers,
    )

    history = client.get(f"/api/v1/students/{user_id}/history", headers=headers).json()

    assert history["summary"]["workouts_logged"] == 1
    entry = history["workouts"][0]
    assert (entry["plan_id"], entry["day"], entry["completion_status"]) == (plan_id, "Monday", None)
    assert entry["performance"][0]["weight_kg"] == 12.5
