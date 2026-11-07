"""The Streamlit assistant, approvals and location screens, with the API faked."""

from streamlit.testing.v1 import AppTest

from tests.app.test_streamlit_app import (
    APP_PATH,
    FakeResponse,
    health_response,
    set_signed_in_state,
)

PENDING = {
    "action_id": "action-1",
    "tool_name": "organize_schedule",
    "reason": "This moves workouts to other days in your plan.",
    "expires_at": "2030-01-09T10:30:00+00:00",
}
AGENT_RESPONSE = {
    "answer": "I proposed moving your Friday workout.",
    "steps": [
        {"agent": "schedule", "task": "Reorganise the week.", "status": "completed"}
    ],
    "reports": [
        {
            "agent": "schedule",
            "task": "Reorganise the week.",
            "status": "completed",
            "summary": "Proposed a move.",
            "tools_used": [{"name": "organize_schedule", "ok": True, "code": None}],
            "results": {},
            "pending_actions": [PENDING],
            "error": None,
        }
    ],
    "pending_actions": [PENDING],
    "notices": [],
    "replans": 0,
    "health_concern": False,
    "conversation_id": "conversation-1",
    "seconds": 12.0,
    "metrics": {
        "model": "qwen3.5:9b",
        "total_seconds": 12.0,
        "phases": [
            {"name": "plan", "seconds": 4.0},
            {"name": "delegate: schedule", "seconds": 8.0},
        ],
        "model_calls": [
            {
                "purpose": "plan",
                "seconds": 4.0,
                "input_tokens": 800,
                "output_tokens": 20,
            }
        ],
        "tool_calls": 1,
        "tool_seconds": 0.05,
        "input_tokens": 800,
        "output_tokens": 20,
    },
}


def start(monkeypatch, handler):
    """Start the app signed in, with ``handler`` answering every API call."""
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if url.endswith("/api/v1/health"):
            return health_response()
        return handler(method, url, kwargs)

    monkeypatch.setattr("requests.request", fake_request)
    app = AppTest.from_file(str(APP_PATH))
    set_signed_in_state(app)
    app.session_state["plan_history"] = []
    app.run(timeout=20)
    return app, calls


def test_opening_the_app_only_checks_health(monkeypatch):
    def handler(method, url, kwargs):
        raise AssertionError(f"Unexpected API request: {method} {url}")

    app, calls = start(monkeypatch, handler)

    assert not app.exception
    assert [url for _, url, _ in calls] == ["http://127.0.0.1:8000/api/v1/health"]


def test_a_message_shows_the_plan_and_the_action_can_be_approved(monkeypatch):
    def handler(method, url, kwargs):
        if url.endswith("/agent/requests"):
            assert kwargs["json"]["message"] == "Organise my week"
            assert kwargs["headers"]["Authorization"] == "Bearer test-token"
            return FakeResponse(AGENT_RESPONSE)
        if url.endswith("/agent/actions/action-1/approve"):
            action = {
                **PENDING,
                "id": "action-1",
                "status": "executed",
                "result": {"ok": True},
            }
            return FakeResponse({"action": action, "executed": True})
        if method == "GET" and url.endswith("/plans"):
            # After an approved change, the app reloads the plans on purpose.
            return FakeResponse([])
        raise AssertionError(f"Unexpected API request: {method} {url}")

    app, calls = start(monkeypatch, handler)
    app.text_area(key="agent-draft").set_value("Organise my week")
    next(b for b in app.button if b.label == "Send to the agents").click().run(
        timeout=20
    )

    assert not app.exception
    assert any(
        "I proposed moving your Friday workout." in m.value for m in app.markdown
    )
    assert app.session_state["agent_conversation_id"] == "conversation-1"
    assert app.text_area(key="agent-draft").value == ""  # cleared after sending
    logged = app.session_state["request_log"][0]
    assert (logged["agents"], logged["input_tokens"], logged["server_s"]) == (
        "schedule",
        800,
        12.0,
    )
    assert any(metric.label == "Tokens in / out" for metric in app.metric)
    app.button(key="approve-msg1-action-1").click().run(timeout=20)

    assert any(url.endswith("/approve") for _, url, _ in calls)
    assert any(
        "organize_schedule ran after your approval" in s.value for s in app.success
    )


def test_the_approvals_inbox_loads_and_declines(monkeypatch):
    stored = {
        **PENDING,
        "id": "action-1",
        "status": "pending",
        "proposed_by": "schedule",
        "arguments": {},
    }

    def handler(method, url, kwargs):
        if url.endswith("/agent/actions"):
            return FakeResponse([stored])
        if url.endswith("/agent/actions/action-1/decline"):
            return FakeResponse(
                {"action": {**stored, "status": "declined"}, "executed": False}
            )
        raise AssertionError(f"Unexpected API request: {method} {url}")

    app, _calls = start(monkeypatch, handler)
    next(b for b in app.button if b.label == "Load pending actions").click().run(
        timeout=20
    )
    app.button(key="decline-inbox-action-1").click().run(timeout=20)

    assert not app.exception
    assert any("did not run" in message.value for message in app.info)
    assert app.session_state["agent_actions"] == []


def test_an_expired_approval_is_explained(monkeypatch):
    def handler(method, url, kwargs):
        if url.endswith("/agent/actions"):
            return FakeResponse([{**PENDING, "id": "action-1", "status": "pending"}])
        if url.endswith("/approve"):
            return FakeResponse({"detail": "This action expired."}, status_code=410)
        raise AssertionError(f"Unexpected API request: {method} {url}")

    app, _calls = start(monkeypatch, handler)
    next(b for b in app.button if b.label == "Load pending actions").click().run(
        timeout=20
    )
    app.button(key="approve-inbox-action-1").click().run(timeout=20)

    assert any("it expired" in message.value for message in app.error)


def test_a_saved_location_is_loaded_on_request(monkeypatch):
    def handler(method, url, kwargs):
        if method == "GET" and url.endswith("/location"):
            return FakeResponse(
                {
                    "label": "Cachan",
                    "latitude": 48.79,
                    "longitude": 2.33,
                    "consented_at": "2030-01-09T10:00:00",
                }
            )
        raise AssertionError(f"Unexpected API request: {method} {url}")

    app, _calls = start(monkeypatch, handler)
    next(b for b in app.button if b.label == "Load saved location").click().run(
        timeout=20
    )

    assert not app.exception
    assert any("Cachan" in message.value for message in app.success)


def test_signing_out_forgets_the_account(monkeypatch):
    app, _calls = start(monkeypatch, lambda *args: None)
    next(b for b in app.button if b.label == "Sign out").click().run(timeout=20)

    assert "access_token" not in app.session_state
    assert "user_id" not in app.session_state


def test_an_empty_message_is_not_sent(monkeypatch):
    def handler(method, url, **kwargs):
        if "/agent/requests" in url:
            raise AssertionError("An empty message must not reach the API.")
        return FakeResponse([])

    app, _ = start(monkeypatch, handler)
    app.text_area(key="agent-draft").set_value("   ")
    next(b for b in app.button if b.label == "Send to the agents").click().run(
        timeout=20
    )

    assert any("Write a message first." in error.value for error in app.error)
