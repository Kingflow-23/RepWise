import json
from io import BytesIO
from pathlib import Path
from typing import Any
import wave

import pytest

from streamlit.testing.v1 import AppTest
import streamlit as st

APP_PATH = Path(__file__).resolve().parents[2] / "genai_app" / "streamlit_app.py"


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


def fake_microphone(_label, *, key, **_kwargs):
    """Supply a recording as if the browser microphone had returned WAV data."""
    st.session_state[key] = BytesIO(b"microphone recording")
    return st.session_state[key]


def test_voice_fills_editable_draft_without_submitting(monkeypatch):
    plan_requests = []
    transcription_calls = []

    def fake_request(method, url, **kwargs):
        if url.endswith("/api/v1/health"):
            return health_response()
        if url.endswith("/api/v1/speech/transcribe"):
            transcription_calls.append(kwargs)
            assert kwargs["headers"]["Authorization"] == "Bearer test-token"
            assert kwargs["headers"]["Content-Type"] == "audio/wav"
            assert kwargs["data"] == b"microphone recording"
            return FakeResponse({"text": "Je veux travailler les jambes."})
        if method == "POST" and url.endswith("/plans"):
            plan_requests.append(kwargs["json"])
            return FakeResponse({"status": "out_of_scope", "message": "Test response"})
        raise AssertionError(f"Unexpected API request: {method} {url}")

    monkeypatch.setattr("streamlit.audio_input", fake_microphone)
    monkeypatch.setattr("requests.request", fake_request)
    app = AppTest.from_file(str(APP_PATH))
    set_signed_in_state(app)
    app.session_state["plan_history"] = []
    app.run(timeout=20)
    app.button(key="use-plan-recording").click().run(timeout=20)
    assert not app.exception
    assert app.text_input(key="plan-focus").value == "Je veux travailler les jambes."
    assert plan_requests == []
    app.text_input(key="plan-focus").set_value("An easier leg workout").run(timeout=20)
    assert app.text_input(key="plan-focus").value == "An easier leg workout"
    assert len(transcription_calls) == 1
    assert plan_requests == []
    app.form_submit_button[1].click().run(timeout=20)
    assert plan_requests[0]["focus"] == "An easier leg workout"
    assert len(transcription_calls) == 1


def test_voice_failure_preserves_draft_and_can_retry(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        if url.endswith("/api/v1/health"):
            return health_response()
        if url.endswith("/api/v1/speech/transcribe"):
            calls.append(url)
            if len(calls) == 1:
                return FakeResponse(
                    {"detail": "Speech transcription is unavailable."}, status_code=503
                )
            return FakeResponse({"text": "A new request"})
        raise AssertionError(f"Unexpected API request: {method} {url}")

    monkeypatch.setattr("streamlit.audio_input", fake_microphone)
    monkeypatch.setattr("requests.request", fake_request)
    app = AppTest.from_file(str(APP_PATH))
    set_signed_in_state(app)
    app.session_state["plan_history"] = []
    app.session_state["plan-focus"] = "My typed request"
    app.run(timeout=20)
    app.button(key="use-plan-recording").click().run(timeout=20)
    assert app.text_input(key="plan-focus").value == "My typed request"
    assert any("unavailable" in error.value for error in app.error)
    app.button(key="use-plan-recording").click().run(timeout=20)
    assert not app.exception
    assert app.text_input(key="plan-focus").value == "A new request"
    assert not app.error


def test_long_transcription_requires_editing_before_generation(monkeypatch):
    def fake_request(method, url, **kwargs):
        if url.endswith("/api/v1/health"):
            return health_response()
        if url.endswith("/api/v1/speech/transcribe"):
            return FakeResponse({"text": "a" * 301})
        raise AssertionError(f"Unexpected API request: {method} {url}")

    monkeypatch.setattr("streamlit.audio_input", fake_microphone)
    monkeypatch.setattr("requests.request", fake_request)
    app = AppTest.from_file(str(APP_PATH))
    set_signed_in_state(app)
    app.session_state["plan_history"] = []
    app.run(timeout=20)
    app.button(key="use-plan-recording").click().run(timeout=20)
    assert len(app.text_input(key="plan-focus").value) == 301
    app.form_submit_button[1].click().run(timeout=20)
    assert any("300 characters" in error.value for error in app.error)
    assert not app.exception


def test_native_recording_preserves_draft_and_requires_explicit_transcription(
    monkeypatch,
):
    calls = []
    audio = BytesIO()
    with wave.open(audio, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16000)
        recording.writeframes(b"\0" * 32000)

    def fake_request(method, url, **kwargs):
        if url.endswith("/api/v1/health"):
            return health_response()
        if url.endswith("/api/v1/speech/transcribe"):
            calls.append(kwargs["data"])
            return FakeResponse({"text": "A recorded request"})
        raise AssertionError(f"Unexpected API request: {method} {url}")

    monkeypatch.setattr("requests.request", fake_request)
    app = AppTest.from_file(str(APP_PATH))
    set_signed_in_state(app)
    app.session_state["plan_history"] = []
    app.run(timeout=20)
    recorder = app.get("audio_input")[0]
    if not hasattr(recorder, "set_value"):
        pytest.skip("This Streamlit version does not support audio widgets in AppTest.")
    assert recorder.sample_rate == 16000
    assert app.button(key="use-plan-recording").disabled
    app.text_input(key="plan-focus").set_value("Keep this typed draft").run(timeout=20)
    app.get("audio_input")[0].set_value(
        ("request.wav", audio.getvalue(), "audio/wav")
    ).run(timeout=20)
    assert app.text_input(key="plan-focus").value == "Keep this typed draft"
    assert calls == []
    app.button(key="use-plan-recording").click().run(timeout=20)
    assert not app.exception
    assert calls == [audio.getvalue()]
    assert app.text_input(key="plan-focus").value == "A recorded request"


def test_scope_refusal_shows_analysis_message_without_plan(monkeypatch: Any) -> None:
    """Display the scope analysis response directly and keep plan history intact."""
    refusal = (
        "RepWise cannot diagnose pain or prescribe treatment. "
        "Please consult a health professional."
    )

    def fake_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        if url.endswith("/api/v1/health"):
            return health_response()
        if method == "POST" and url.endswith("/plans"):
            return FakeResponse({"status": "out_of_scope", "message": refusal})
        raise AssertionError(f"Unexpected API request: {method} {url}")

    monkeypatch.setattr("requests.request", fake_request)
    app = AppTest.from_file(str(APP_PATH))
    set_signed_in_state(app)
    app.session_state["plan_history"] = []
    app.run(timeout=20)

    app.text_input(key="plan-focus").set_value("Diagnose my knee pain")
    app.form_submit_button[1].click().run(timeout=20)

    assert any(refusal in message.value for message in app.info)
    assert app.session_state["plan_history"] == []
    assert "latest_plan" not in app.session_state
    assert "plan_generation_error" not in app.session_state


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


def test_saved_workout_feedback_form_refreshes_the_displayed_plan(
    monkeypatch: Any,
) -> None:
    submitted_urls: list[str] = []
    feedback = {
        "id": "feedback-id",
        "plan_id": "plan-id",
        "week_number": 1,
        "workout_index": 0,
        "completion_status": "completed",
        "difficulty": "just_right",
        "comment": "Felt good.",
        "created_at": "2026-11-01T10:00:00+00:00",
        "adapted_week_number": None,
        "adapted_workout_index": None,
    }

    def fake_request(method: str, url: str, **kwargs: Any) -> FakeResponse:
        if url.endswith("/api/v1/health"):
            return health_response()
        if method == "POST" and url.endswith("/weeks/1/workouts/0/feedback"):
            submitted_urls.append(url)
            return FakeResponse(
                {
                    "plan_id": "plan-id",
                    "plan": {
                        "title": "Plan",
                        "summary": "Updated plan",
                        "weeks": [
                            {
                                "number": 1,
                                "objective": "Build consistency.",
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
                                                "coaching_note": "Stay controlled.",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    },
                    "workout_feedback": [feedback],
                    "adapted": False,
                    "adapted_workout": None,
                    "message": "Feedback saved. No later workout was available.",
                }
            )
        raise AssertionError(f"Unexpected API request: {method} {url}")

    monkeypatch.setattr("requests.request", fake_request)
    record = {
        "id": "plan-id",
        "plan": {
            "title": "Plan",
            "summary": "A weekly plan.",
            "weeks": [
                {
                    "number": 1,
                    "objective": "Build consistency.",
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
                                    "coaching_note": "Stay controlled.",
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        "request": {"focus": "Strength", "weeks": 1},
        "sources": [],
        "workout_feedback": [],
        "created_at": "2026-11-01T09:00:00+00:00",
    }
    app = AppTest.from_file(str(APP_PATH))
    set_signed_in_state(app)
    app.session_state["latest_plan"] = record
    app.session_state["plan_history"] = [json.loads(json.dumps(record))]
    app.run(timeout=20)

    log_buttons = [
        button for button in app.form_submit_button if button.label == "Log workout"
    ]
    assert len(log_buttons) == 2
    log_buttons[0].click().run(timeout=20)

    assert len(submitted_urls) == 1
    assert app.session_state["latest_plan"]["workout_feedback"] == [feedback]
    assert any("Workout logged" in item.value for item in app.success)
