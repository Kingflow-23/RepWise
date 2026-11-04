"""Behavior and contract tests for the three agent tools."""

import asyncio
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from fastmcp import Client
from pydantic import ValidationError

from agentic import llm_check, tools
from agentic.mcp_demo_client import seed_demo_account
from agentic.mcp_server import mcp
from config.config import AppConfig
from src.api import main as api_module
from src.db.repository import StudentRepository
from src.schemas.models import (
    UserProfile,
    Workout,
    WorkoutFeedback,
    WorkoutFeedbackRequest,
)

ROOT = Path(__file__).resolve().parents[2]
BODY = ("completed", "slightly_too_hard", "Last set was difficult")


@pytest.fixture
def database(tmp_path, monkeypatch):
    """Point the tools at a temporary database."""
    path = tmp_path / "repwise.sqlite3"
    monkeypatch.setenv("REPWISE_DATABASE_PATH", str(path))
    return path


@pytest.fixture
def account(database):
    """Create an account with a three-workout plan, and return its IDs."""
    return seed_demo_account(database)


def api_app(repository, tmp_path):
    """Build the real API with its model services replaced by unused stubs."""
    return api_module.create_app(
        AppConfig(database_path=str(tmp_path / "api.sqlite3"), auth_secret="secret"),
        repository=repository,
        retriever=object(),
        scope_analyzer=object(),
        generator=object(),
        transcriber=object(),
    )


def test_profile_is_the_saved_user_profile(account):
    student_id, _ = account
    result = tools.get_user_profile(student_id)
    assert isinstance(result, tools.ProfileResult)
    assert isinstance(result.profile, UserProfile)
    assert result.student_id == student_id
    json.dumps(result.model_dump())


def test_timetable_gives_values_that_can_be_passed_straight_to_save(account):
    student_id, plan_id = account
    result = tools.get_timetable(student_id)
    assert result.available_days == ["Monday", "Wednesday", "Friday"]
    assert [item.workout_index for item in result.planned_workouts] == [0, 1, 2]
    workout = result.planned_workouts[2]
    assert isinstance(workout, Workout)
    assert workout.plan_id == plan_id
    assert workout.feedback is None
    saved = tools.save_workout_feedback(
        student_id, workout.plan_id, workout.week_number, workout.workout_index, *BODY
    )
    assert saved.saved is True
    json.dumps(result.model_dump())


def test_saved_record_is_the_api_record_and_can_be_reloaded(account):
    student_id, plan_id = account
    result = tools.save_workout_feedback(student_id, plan_id, 1, 0, *BODY)
    assert result.ok is True and result.saved is True
    assert isinstance(result.feedback, WorkoutFeedback)
    assert (result.feedback.plan_id, result.feedback.workout_index) == (plan_id, 0)
    assert result.feedback.comment == "Last set was difficult"
    workouts = tools.get_timetable(student_id).planned_workouts
    assert workouts[0].feedback == result.feedback
    assert workouts[1].feedback is None


def test_saved_feedback_appears_in_the_real_api(
    account, database, tmp_path, monkeypatch
):
    student_id, plan_id = account
    monkeypatch.setattr(api_module, "read_access_token", lambda token, secret: token)
    saved = tools.save_workout_feedback(student_id, plan_id, 1, 1, *BODY)
    client = TestClient(api_app(StudentRepository(database), tmp_path))
    response = client.get(
        f"/api/v1/students/{student_id}/plans",
        headers={"Authorization": f"Bearer {student_id}"},
    )
    assert response.json()[0]["workout_feedback"] == [saved.feedback.model_dump()]


def test_saving_again_updates_the_same_record(account, database):
    student_id, plan_id = account
    tools.save_workout_feedback(student_id, plan_id, 1, 0, *BODY)
    tools.save_workout_feedback(
        student_id, plan_id, 1, 0, "not_completed", "much_too_easy"
    )
    records = StudentRepository(database).list_workout_feedback(student_id, plan_id)
    assert [item.difficulty for item in records] == ["much_too_easy"]


def test_reads_do_not_change_the_database(account, database):
    student_id, _ = account
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    tools.get_user_profile(student_id)
    tools.get_timetable(student_id)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"student_id": "unknown"}, "student_not_found"),
        ({"plan_id": "unknown"}, "plan_not_found"),
        ({"week_number": 9}, "workout_not_found"),
        ({"workout_index": 9}, "workout_not_found"),
        ({"workout_index": -1}, "workout_not_found"),
        ({"completion_status": "finished"}, "invalid_request"),
        ({"difficulty": "hard"}, "invalid_request"),
        ({"comment": "x" * 501}, "invalid_request"),
    ],
)
def test_invalid_feedback_is_rejected_without_writing(account, database, change, code):
    student_id, plan_id = account
    arguments = {
        "student_id": student_id,
        "plan_id": plan_id,
        "week_number": 1,
        "workout_index": 0,
        "completion_status": "completed",
        "difficulty": "just_right",
        **change,
    }
    result = tools.save_workout_feedback(**arguments)
    assert isinstance(result, tools.ToolFailure)
    assert result.code == code
    assert StudentRepository(database).list_workout_feedback(student_id, plan_id) == []


def test_a_student_cannot_save_to_another_students_plan(account, database):
    owner, plan_id = account
    repository = StudentRepository(database)
    other = repository.create_user("Other Example", "test-hash")
    repository.save_profile(other, repository.get_profile(owner))
    assert (
        tools.save_workout_feedback(other, plan_id, 1, 0, *BODY).code
        == "plan_not_found"
    )


def test_a_student_with_no_plan_has_an_empty_timetable(account, database):
    owner, _ = account
    repository = StudentRepository(database)
    other = repository.create_user("Other Example", "test-hash")
    repository.save_profile(other, repository.get_profile(owner))
    assert tools.get_timetable(other).planned_workouts == []


@pytest.mark.parametrize("read", [tools.get_user_profile, tools.get_timetable])
def test_reads_reject_unknown_students(account, read):
    assert read("unknown").code == "student_not_found"


def test_a_missing_database_is_reported_and_never_created(database):
    assert tools.get_user_profile("anyone").code == "storage_error"
    assert tools.get_timetable("anyone").code == "storage_error"
    assert (
        tools.save_workout_feedback("anyone", "plan", 1, 0, *BODY).code
        == "storage_error"
    )
    assert not database.exists()


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (tools.ProfileResult, {"ok": False, "student_id": "x", "profile": {}}),
        (tools.ToolFailure, {"code": "made_up", "error": "x"}),
        (tools.FeedbackResult, {"student_id": "x", "feedback": {"difficulty": "hard"}}),
    ],
)
def test_output_models_reject_malformed_results(model, values):
    with pytest.raises(ValidationError):
        model(**values)


def test_parameter_names_match_the_feedback_route_and_body(tmp_path):
    app = api_app(StudentRepository(tmp_path / "api.sqlite3"), tmp_path)
    route = next(
        item for item in app.routes if getattr(item, "path", "").endswith("/feedback")
    )
    expected = list(route.param_convertors) + list(WorkoutFeedbackRequest.model_fields)
    assert list(inspect.signature(tools.save_workout_feedback).parameters) == expected


def test_schema_file_matches_the_live_server_and_api_values():
    async def discover():
        async with Client(mcp) as client:
            return await client.list_tools()

    live = {tool.name: tool for tool in asyncio.run(discover())}
    published = json.loads(
        (ROOT / "docs/agent/s2_tool_schemas.json").read_text(encoding="utf-8")
    )
    types = {
        "get_user_profile": "read",
        "get_timetable": "read",
        "save_workout_feedback": "write",
    }

    assert set(live) == set(types)
    for item in published["tools"]:
        assert list(item) == ["name", "type", "description", "parameters"]
        assert item["type"] == types[item["name"]]
        assert item["description"] == live[item["name"]].description
        assert item["parameters"] == live[item["name"]].input_schema

    def constraints(schema: dict) -> dict:
        return {k: v for k, v in schema.items() if k not in ("title", "description")}

    feedback = live["save_workout_feedback"].input_schema
    api_body = WorkoutFeedbackRequest.model_json_schema()["properties"]
    for field in WorkoutFeedbackRequest.model_fields:
        assert constraints(feedback["properties"][field]) == constraints(
            api_body[field]
        )
    assert feedback["properties"]["week_number"]["minimum"] == 1
    assert feedback["properties"]["workout_index"]["minimum"] == 0


def test_mcp_demo_runs_end_to_end_over_stdio(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "agentic.mcp_demo_client"],
        cwd=ROOT,
        env={**os.environ, "REPWISE_LOG_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Demonstration passed." in completed.stdout


def test_llm_check_runs_the_tools_the_model_chooses(account, database):
    student_id, plan_id = account
    arguments = {
        "student_id": student_id,
        "plan_id": plan_id,
        "week_number": 1,
        "workout_index": 0,
        "completion_status": "completed",
        "difficulty": "slightly_too_hard",
    }
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="save_workout_feedback", arguments=arguments)
    )
    replies = iter(
        [
            SimpleNamespace(content="", tool_calls=[tool_call]),
            SimpleNamespace(content="Logged.", tool_calls=None),
        ]
    )
    client = SimpleNamespace(chat=lambda **_: SimpleNamespace(message=next(replies)))

    calls, answer = llm_check.run_agent(client, "model", "Log it.", student_id)

    assert calls == [("save_workout_feedback", arguments)]
    assert answer == "Logged."
    case = {
        "tool": "save_workout_feedback",
        "saved": (0, "completed", {"slightly_too_hard"}),
    }
    assert llm_check.check_case(case, calls, student_id, plan_id) == []


def test_llm_check_fails_a_write_nobody_asked_for(account):
    student_id, plan_id = account
    arguments = {"student_id": student_id}
    tools.save_workout_feedback(student_id, plan_id, 1, 0, *BODY)
    case = {"tool": "get_timetable", "saved": None}
    calls = [("get_timetable", arguments)]
    assert llm_check.check_case(case, calls, student_id, plan_id) == [
        "saved feedback without being asked"
    ]
