import json
from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"


class FakeResponse:
    def __init__(self, value: Any, status_code: int = 200) -> None:
        self.value = value
        self.status_code = status_code
        self.ok = status_code < 400
        self.content = json.dumps(value).encode() if status_code != 204 else b""
        self.text = self.content.decode()
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        return self.value


def health_response() -> FakeResponse:
    return FakeResponse(
        {"retrieval": "ready", "retriever": "hybrid", "model": "qwen3.5:9b"}
    )


def set_signed_in_state(app: AppTest) -> None:
    app.session_state["access_token"] = "test-token"
    app.session_state["user_id"] = "student-id"
    app.session_state["full_name"] = "Test Student"
    app.session_state["profile_form"] = {
        "available_days": ["Monday", "Wednesday", "Friday"],
        "equipment": ["bodyweight"],
    }


def test_generation_failure_explains_reason_and_retries_last_request(
    monkeypatch: Any,
) -> None:
    requests: list[dict[str, Any]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        if url.endswith("/api/v1/health"):
            return health_response()
        if method == "POST" and url.endswith("/plans"):
            assert kwargs["headers"]["Authorization"] == "Bearer test-token"
            requests.append(kwargs["json"])
            if len(requests) == 1:
                return FakeResponse(
                    {
                        "detail": {
                            "status": "generation_error",
                            "message": (
                                "an exercise requires equipment that is not in "
                                "the student profile"
                            ),
                        }
                    },
                    status_code=422,
                )
            return FakeResponse(
                {
                    "plan_id": "new-plan",
                    "plan": {
                        "title": "Legs",
                        "summary": "Test plan",
                        "weeks": [
                            {
                                "number": 1,
                                "objective": "Build leg strength.",
                                "workouts": [
                                    {
                                        "day": "Monday",
                                        "focus": "Leg strength",
                                        "estimated_minutes": 40,
                                        "exercises": [
                                            {
                                                "name": "Goblet squat",
                                                "sets": 3,
                                                "repetitions": "8-10",
                                                "rest_seconds": 60,
                                                "estimated_minutes": 10,
                                                "coaching_note": "Stay controlled.",
                                                "supporting_evidence": [],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    },
                    "sources": [],
                }
            )
        raise AssertionError(f"Unexpected API request: {method} {url}")

    monkeypatch.setattr("requests.request", fake_request)
    app = AppTest.from_file(str(APP_PATH))
    set_signed_in_state(app)
    app.session_state["plan_history"] = []
    app.run(timeout=20)

    app.text_input(key="plan-focus").set_value("Legs strength")
    app.text_area(key="plan-refinement").set_value("Add more variety this time.")
    app.form_submit_button[1].click().run(timeout=20)

    assert requests[0]["refinement_instructions"] == "Add more variety this time."
    assert any(
        "equipment outside your saved profile" in error.value for error in app.error
    )
    assert app.session_state["plan_history"] == []
    assert "latest_plan" not in app.session_state

    next(
        button for button in app.button if button.label == "Try generating again"
    ).click().run(timeout=20)

    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert app.session_state["latest_plan"]["id"] == "new-plan"
    assert app.session_state["plan_history"][0]["id"] == "new-plan"
    assert "plan_generation_error" not in app.session_state
    assert any(
        "Goblet squat" in item.value and "~10 min" in item.value
        for item in app.markdown
    )


def test_saved_plan_delete_action_is_available_outside_details_expander(
    monkeypatch: Any,
) -> None:
    deleted_urls: list[str] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        if url.endswith("/api/v1/health"):
            return health_response()
        if method == "DELETE":
            deleted_urls.append(url)
            return FakeResponse(None, status_code=204)
        raise AssertionError(f"Unexpected API request: {method} {url}")

    monkeypatch.setattr("requests.request", fake_request)
    app = AppTest.from_file(str(APP_PATH))
    set_signed_in_state(app)
    app.session_state["plan_history"] = [
        {
            "id": "first-plan",
            "plan": {"title": "First plan", "weeks": []},
            "request": {"focus": "First", "weeks": 1},
            "sources": [],
            "created_at": "2026-10-31T00:00:00+00:00",
        },
        {
            "id": "second-plan",
            "plan": {"title": "Second plan", "weeks": []},
            "request": {"focus": "Second", "weeks": 1},
            "sources": [],
            "created_at": "2026-10-30T00:00:00+00:00",
        },
    ]
    app.run(timeout=20)

    assert sum(button.label == "Delete plan" for button in app.button) == 2
    app.button(key="delete-plan-second-plan").click().run(timeout=20)
    assert app.session_state["pending_delete_plan_id"] == "second-plan"
    app.button(key="confirm-delete-second-plan").click().run(timeout=20)

    assert deleted_urls == [
        "http://127.0.0.1:8000/api/v1/students/student-id/plans/second-plan"
    ]
    assert [plan["id"] for plan in app.session_state["plan_history"]] == ["first-plan"]
