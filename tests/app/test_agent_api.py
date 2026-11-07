"""The agent, approval and saved-location routes, with every external service faked."""

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from config.config import AppConfig
from src.agents.health import FixedHealthChecker
from src.agents.orchestrator import Delegation, OrchestratorPlan
from src.api.auth import create_access_token
from src.api.main import create_app
from src.db.demo import seed_demo_account
from src.integrations.photon import PhotonClient
from src.retrieval.query_expansion import QueryAnalysis
from tests.conftest import HEALTH_PHRASES, ScriptedModel, tool_call

SECRET = "y" * 64


class InScope:
    def analyze(self, query):
        return QueryAnalysis("in_scope", query, "")


class NoRetriever:
    def search(self, query, top_k):
        return []


def geocoder(_request):
    feature = {
        "properties": {"name": "Lyon"},
        "geometry": {"coordinates": [4.83, 45.76]},
    }
    return httpx.Response(200, json={"features": [feature]})


@pytest.fixture
def api(tmp_path):
    """An app on a temporary database with two demo accounts and a scripted model."""
    database = tmp_path / "api.sqlite3"
    alice, alice_plan = seed_demo_account(database, "Alice Example")
    bob, _ = seed_demo_account(database, "Bob Example")
    model = ScriptedModel()
    app = create_app(
        AppConfig(database_path=str(database), auth_secret=SECRET),
        retriever=NoRetriever(),
        scope_analyzer=InScope(),
        generator=object(),
        transcriber=object(),
        health_checker=FixedHealthChecker(HEALTH_PHRASES),
        agent_llm=model,
        places_client=PhotonClient(
            "https://photon.test", "tests", transport=httpx.MockTransport(geocoder)
        ),
        calendar_loader=lambda _student: None,
    )
    client = TestClient(app)

    def headers(user):
        return {"Authorization": f"Bearer {create_access_token(user, SECRET)}"}

    return client, model, headers, alice, alice_plan, bob


def move_plan():
    return OrchestratorPlan(
        steps=[Delegation(agent="schedule", task="Swap Monday and Friday.")]
    )


def swap_call():
    moves = [
        {"week_number": 1, "workout_index": 0, "new_day": "Friday"},
        {"week_number": 1, "workout_index": 2, "new_day": "Monday"},
    ]
    return tool_call("organize_schedule", moves=moves)


def ask_for_swap(api):
    client, model, headers, alice, _plan, _bob = api
    model.plans.append(move_plan())
    model.replies += [
        swap_call(),
        AIMessage("Proposed the swap."),
        AIMessage("I proposed swapping the days."),
    ]
    response = client.post(
        f"/api/v1/students/{alice}/agent/requests",
        json={
            "message": "Swap my Monday and Friday workouts",
            "timezone": "Europe/Paris",
        },
        headers=headers(alice),
    )
    assert response.status_code == 200, response.text
    return response.json()


def first_day(client, headers, alice):
    plans = client.get(f"/api/v1/students/{alice}/plans", headers=headers(alice)).json()
    return plans[0]["plan"]["weeks"][0]["workouts"][0]["day"]


def test_an_agent_request_returns_the_plan_and_the_pending_action(api):
    body = ask_for_swap(api)

    assert [step["agent"] for step in body["steps"]] == ["schedule"]
    assert body["pending_actions"][0]["tool_name"] == "organize_schedule"
    assert body["conversation_id"]
    assert "Waiting for your approval" in body["answer"]


def test_approving_runs_the_action_once(api):
    client, _model, headers, alice, _plan, _bob = api
    action_id = ask_for_swap(api)["pending_actions"][0]["action_id"]
    assert first_day(client, headers, alice) == "Monday"

    approved = client.post(
        f"/api/v1/students/{alice}/agent/actions/{action_id}/approve",
        headers=headers(alice),
    )
    replay = client.post(
        f"/api/v1/students/{alice}/agent/actions/{action_id}/approve",
        headers=headers(alice),
    )

    assert (
        approved.status_code == 200
        and approved.json()["action"]["status"] == "executed"
    )
    assert replay.status_code == 409
    assert first_day(client, headers, alice) == "Friday"


def test_declining_changes_nothing(api):
    client, _model, headers, alice, _plan, _bob = api
    action_id = ask_for_swap(api)["pending_actions"][0]["action_id"]

    declined = client.post(
        f"/api/v1/students/{alice}/agent/actions/{action_id}/decline",
        headers=headers(alice),
    )

    assert declined.json()["action"]["status"] == "declined"
    assert first_day(client, headers, alice) == "Monday"
    pending = client.get(
        f"/api/v1/students/{alice}/agent/actions", headers=headers(alice)
    ).json()
    assert pending == []


def test_another_account_cannot_approve_or_list_the_action(api):
    client, _model, headers, alice, _plan, bob = api
    action_id = ask_for_swap(api)["pending_actions"][0]["action_id"]

    own_path = client.post(
        f"/api/v1/students/{bob}/agent/actions/{action_id}/approve",
        headers=headers(bob),
    )
    other_path = client.post(
        f"/api/v1/students/{alice}/agent/actions/{action_id}/approve",
        headers=headers(bob),
    )

    assert own_path.status_code == 404
    assert other_path.status_code == 403
    assert (
        client.get(f"/api/v1/students/{bob}/agent/actions", headers=headers(bob)).json()
        == []
    )
    assert first_day(client, headers, alice) == "Monday"


def test_agent_routes_require_the_owner(api):
    client, _model, headers, alice, _plan, bob = api
    path = f"/api/v1/students/{alice}/agent/requests"

    assert client.post(path, json={"message": "hi"}).status_code == 401
    assert (
        client.post(path, json={"message": "hi"}, headers=headers(bob)).status_code
        == 403
    )


def test_an_unknown_time_zone_is_rejected(api):
    client, _model, headers, alice, _plan, _bob = api

    response = client.post(
        f"/api/v1/students/{alice}/agent/requests",
        json={"message": "hi", "timezone": "Mars/Olympus"},
        headers=headers(alice),
    )

    assert response.status_code == 422


def test_a_saved_location_is_opt_in_geocoded_and_deletable(api):
    client, _model, headers, alice, _plan, bob = api
    path = f"/api/v1/students/{alice}/location"

    assert client.get(path, headers=headers(alice)).status_code == 404
    saved = client.put(path, json={"label": "Lyon"}, headers=headers(alice))
    assert saved.status_code == 200
    assert (saved.json()["latitude"], saved.json()["longitude"]) == (45.76, 4.83)
    assert client.get(path, headers=headers(bob)).status_code == 403
    assert client.delete(path, headers=headers(alice)).status_code == 204
    assert client.get(path, headers=headers(alice)).status_code == 404


def test_half_a_coordinate_is_rejected(api):
    client, _model, headers, alice, _plan, _bob = api

    response = client.put(
        f"/api/v1/students/{alice}/location",
        json={"label": "X", "latitude": 1},
        headers=headers(alice),
    )

    assert response.status_code == 422
