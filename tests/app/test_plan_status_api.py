"""Active plans and drafts must remain distinct across the HTTP boundary."""

import pytest
from fastapi.testclient import TestClient

from config.config import AppConfig
from src.agents.approvals import propose
from src.api.auth import create_access_token
from src.api.main import create_app
from src.retrieval.query_expansion import QueryAnalysis
from src.schemas.models import PlanRequest, PlanResult, Source, TrainingPlan, WorkoutFeedbackRequest

SECRET = "plan-status-tests-" * 4


class EmptyRetriever:
    def search(self, query, top_k):
        return []


class InScope:
    def analyze(self, query):
        return QueryAnalysis("in_scope", query, "")


@pytest.fixture
def api(demo):
    app = create_app(
        AppConfig(database_path=demo.repository.database_path, auth_secret=SECRET),
        repository=demo.repository,
        retriever=EmptyRetriever(), scope_analyzer=InScope(), generator=object(),
        health_checker=demo.health, calendar_loader=lambda _: None,
    )
    headers = {"Authorization": f"Bearer {create_access_token(demo.student_id, SECRET)}"}
    with TestClient(app) as client:
        yield client, headers, demo


def save_draft(demo):
    current = demo.repository.get_current_plan(demo.student_id)
    return demo.repository.save_plan(
        demo.student_id, PlanRequest.model_validate(current["request"]),
        PlanResult(status="completed", message="Created", plan=TrainingPlan.model_validate(current["plan"]).model_copy(update={"title": "New draft"}), sources=[Source.model_validate(s) for s in current["sources"]]),
        status="draft",
    )


def test_newer_draft_never_replaces_current_program_and_feedback(api):
    client, headers, demo = api
    draft_id = save_draft(demo)
    with demo.repository._connect() as connection:
        connection.execute("UPDATE training_plans SET created_at=? WHERE id=?", ("2099-01-01T00:00:00+00:00", draft_id))
    demo.repository.save_workout_feedback(demo.student_id, demo.plan_id, 1, 0, WorkoutFeedbackRequest(completion_status="completed", difficulty="just_right"))
    base = f"/api/v1/students/{demo.student_id}/plans"
    assert client.get(base + "?limit=1", headers=headers).json()[0]["id"] == draft_id
    response = client.get(base + "/current", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["id"] == demo.plan_id
    assert response.json()["status"] == "active"
    assert response.json()["workout_feedback"][0]["completion_status"] == "completed"
    assert response.json()["sources"] == demo.repository.get_current_plan(demo.student_id)["sources"]
    assert client.get(base + "?status=active&limit=1", headers=headers).json()[0]["id"] == demo.plan_id
    assert client.get(base + "?status=draft&limit=1", headers=headers).json()[0]["id"] == draft_id


def test_only_drafts_and_archives_mean_no_current_program(api):
    client, headers, demo = api
    draft_id = save_draft(demo)
    with demo.repository._connect() as connection:
        connection.execute("UPDATE training_plans SET status='archived' WHERE id=?", (demo.plan_id,))
    base = f"/api/v1/students/{demo.student_id}/plans"
    response = client.get(base + "/current", headers=headers)
    assert response.status_code == 200 and response.json() is None
    assert client.get(base + "?status=draft", headers=headers).json()[0]["id"] == draft_id
    assert client.get(base + "?status=archived", headers=headers).json()[0]["id"] == demo.plan_id


def test_plan_reads_require_authentication_and_owner_and_reject_invalid_status(api, other_student):
    client, headers, demo = api
    for suffix in ["/current", "?status=draft"]:
        assert client.get(f"/api/v1/students/{demo.student_id}/plans" + suffix).status_code == 401
        assert client.get(f"/api/v1/students/{other_student}/plans" + suffix, headers=headers).status_code == 403
    assert client.get(f"/api/v1/students/{demo.student_id}/plans?status=invalid", headers=headers).status_code == 422
    with pytest.raises(ValueError, match="status"):
        demo.repository.list_plans(demo.student_id, status="invalid")


def test_approved_activation_changes_current_program_and_archives_previous(api):
    client, headers, demo = api
    draft_id = save_draft(demo)
    action = propose(demo.context, "activate_plan", {"plan_id": draft_id}, "Activate the proposed program", "planning")
    base = f"/api/v1/students/{demo.student_id}"
    assert client.get(base + "/plans/current", headers=headers).json()["id"] == demo.plan_id
    response = client.post(base + f"/agent/actions/{action.action_id}/approve", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json()["executed"] is True
    assert client.get(base + "/plans/current", headers=headers).json()["id"] == draft_id
    assert client.get(base + "/plans?status=draft", headers=headers).json() == []
    assert client.get(base + "/plans?status=archived", headers=headers).json()[0]["id"] == demo.plan_id
