"""Behavior and contract tests for the Session 2 tools."""

import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import get_args

import pytest

from agentic import tools
from config.config import AppConfig
from src.db.repository import StudentRepository
from src.schemas.models import (
    UserProfile,
    Workout,
    WorkoutFeedback,
    WorkoutFeedbackRequest,
)

ROOT = Path(__file__).resolve().parents[1]
DATABASE_NAME = "agentic.sqlite3"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the tools at a directory that does not exist yet."""
    path = tmp_path / "agentic-data"
    monkeypatch.setenv("REPWISE_DATA_DIR", str(path))
    return path


def table_columns(path: Path, table: str) -> list[tuple]:
    """Return a table's column names, types, and constraints."""
    connection = sqlite3.connect(path)
    try:
        return [row[1:] for row in connection.execute(f"PRAGMA table_info({table})")]
    finally:
        connection.close()


def test_demo_data_labels_the_user_and_is_synthetic():
    data = json.loads(
        (ROOT / "agentic/data/demo_data.json").read_text(encoding="utf-8")
    )
    assert "SYNTHETIC" in data["notice"]
    assert [user["user_id"] for user in data["users"]] == ["demo_user"]


def test_profile_matches_the_api_profile_schema(data_dir):
    result = tools.get_user_profile("demo_user")
    assert result["ok"] is True
    assert result["user_id"] == "demo_user"
    assert UserProfile.model_validate(result["profile"]).model_dump() == result["profile"]
    json.dumps(result)


def test_timetable_uses_profile_availability_and_workout_schema(data_dir):
    result = tools.get_timetable("demo_user")
    assert result["ok"] is True
    assert result["user_id"] == "demo_user"
    days = result["availability"]["available_days"]
    assert days == tools.get_user_profile("demo_user")["profile"]["available_days"]
    workouts = result["planned_workouts"]
    ids = [item["workout_id"] for item in workouts]
    assert "leg_day" in ids
    assert len(ids) == len(set(ids))
    for item in workouts:
        assert item["day"] in days
        assert item["estimated_minutes"] <= result["availability"]["session_minutes"]
        assert item["feedback"] is None
        Workout.model_validate(
            {key: item[key] for key in Workout.model_fields}
        )
    json.dumps(result)


def test_reads_do_not_create_storage(data_dir):
    tools.get_user_profile("demo_user")
    tools.get_timetable("demo_user")
    assert tools.get_saved_feedback("demo_user", "leg_day") is None
    assert not data_dir.exists()


def test_reads_do_not_change_saved_data(data_dir):
    tools.save_workout_feedback("demo_user", "leg_day", "hard")
    path = data_dir / DATABASE_NAME
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    tools.get_user_profile("demo_user")
    tools.get_timetable("demo_user")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_feedback_record_matches_the_api_record(data_dir):
    result = tools.save_workout_feedback(
        "demo_user", "leg_day", "hard", "Last set was difficult"
    )
    assert result["ok"] is True and result["saved"] is True
    record = result["feedback"]
    assert set(record) == set(WorkoutFeedback.model_fields) | {"user_id", "workout_id"}
    assert record["user_id"] == "demo_user"
    assert record["workout_id"] == "leg_day"
    assert record["plan_id"] == "demo_plan"
    assert (record["week_number"], record["workout_index"]) == (1, 0)
    assert record["completion_status"] == "completed"
    assert record["difficulty"] == "slightly_too_hard"
    assert record["comment"] == "Last set was difficult"
    assert record["created_at"]
    WorkoutFeedback.model_validate(
        {key: record[key] for key in WorkoutFeedback.model_fields}
    )
    json.dumps(result)


def test_feedback_is_stored_in_the_main_schema_with_the_user_id(data_dir, tmp_path):
    tools.save_workout_feedback("demo_user", "leg_day", "hard", "Heavy")
    path = data_dir / DATABASE_NAME

    reference = tmp_path / "reference.sqlite3"
    StudentRepository(reference).initialize()
    for table in ("users", "student_profiles", "training_plans", "workout_feedback"):
        assert table_columns(path, table) == table_columns(reference, table)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT user_id FROM users").fetchall() == [
            ("demo_user",)
        ]
        assert connection.execute(
            "SELECT student_id, plan_id, difficulty, comment FROM workout_feedback"
        ).fetchall() == [("demo_user", "demo_plan", "slightly_too_hard", "Heavy")]
    finally:
        connection.close()


def test_feedback_is_accessible_after_reload(data_dir):
    saved = tools.save_workout_feedback("demo_user", "leg_day", "hard", "Heavy")
    assert tools.get_saved_feedback("demo_user", "leg_day") == saved["feedback"]
    workouts = tools.get_timetable("demo_user")["planned_workouts"]
    by_id = {item["workout_id"]: item["feedback"] for item in workouts}
    assert by_id["leg_day"] == saved["feedback"]
    assert by_id["upper_day"] is None


def test_saving_again_updates_the_same_workout_record(data_dir):
    tools.save_workout_feedback("demo_user", "leg_day", "hard", "First")
    second = tools.save_workout_feedback("demo_user", "leg_day", "easy", "Second")
    assert second["feedback"]["difficulty"] == "slightly_too_easy"
    connection = sqlite3.connect(data_dir / DATABASE_NAME)
    try:
        assert connection.execute("SELECT COUNT(*) FROM workout_feedback").fetchone() == (1,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("notes", "expected"),
    [("", None), ("   ", None), ("  Good  ", "Good"), ("x" * 500, "x" * 500)],
)
def test_notes_are_normalized_like_the_api_comment(data_dir, notes, expected):
    result = tools.save_workout_feedback("demo_user", "leg_day", "moderate", notes)
    assert result["feedback"]["comment"] == expected
    assert result["feedback"]["difficulty"] == "just_right"


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        (("nobody", "leg_day", "hard"), "unknown_user"),
        (("", "leg_day", "hard"), "unknown_user"),
        ((None, "leg_day", "hard"), "unknown_user"),
        (("demo_user", "missing", "hard"), "unknown_workout"),
        (("demo_user", "", "hard"), "unknown_workout"),
        (("demo_user", "leg_day", "impossible"), "invalid_difficulty"),
        (("demo_user", "leg_day", "much_too_hard"), "invalid_difficulty"),
        (("demo_user", "leg_day", ["hard"]), "invalid_difficulty"),
        (("demo_user", "leg_day", "hard", "x" * 501), "invalid_notes"),
        (("demo_user", "leg_day", "hard", 7), "invalid_notes"),
    ],
)
def test_invalid_feedback_is_rejected_without_writing(data_dir, arguments, code):
    result = tools.save_workout_feedback(*arguments)
    assert result["ok"] is False and result["saved"] is False
    assert result["code"] == code
    assert result["error"]
    assert not data_dir.exists()


@pytest.mark.parametrize("function", [tools.get_user_profile, tools.get_timetable])
def test_unknown_users_are_rejected_by_reads(data_dir, function):
    result = function("nobody")
    assert result["ok"] is False and result["code"] == "unknown_user"
    assert not data_dir.exists()


def test_storage_failure_is_not_reported_as_saved(tmp_path, monkeypatch):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("blocked", encoding="utf-8")
    monkeypatch.setenv("REPWISE_DATA_DIR", str(blocked))
    result = tools.save_workout_feedback("demo_user", "leg_day", "hard")
    assert result["ok"] is False and result["saved"] is False
    assert result["code"] == "storage_error"


def test_limits_and_values_stay_aligned_with_the_api_schema():
    difficulty = WorkoutFeedbackRequest.model_fields["difficulty"].annotation
    assert set(tools.DIFFICULTY_TO_API.values()) <= set(get_args(difficulty))
    assert set(tools.DIFFICULTY_TO_API) == set(get_args(tools.Difficulty))
    comment = WorkoutFeedbackRequest.model_json_schema()["properties"]["comment"]
    limits = [item["maxLength"] for item in comment["anyOf"] if "maxLength" in item]
    assert limits == [tools.MAX_NOTES_LENGTH]


def test_default_storage_is_not_the_application_database(monkeypatch):
    monkeypatch.delenv("REPWISE_DATA_DIR", raising=False)
    default = (tools._data_dir() / DATABASE_NAME).resolve()
    assert default != (ROOT / AppConfig.database_path).resolve()


def test_published_schemas_match_the_live_server():
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    from agentic.mcp_server import mcp

    async def discover():
        async with Client(mcp) as client:
            return await client.list_tools()

    live = {tool.name: tool for tool in asyncio.run(discover())}
    assert set(live) == {"get_user_profile", "get_timetable", "save_workout_feedback"}

    published = json.loads((ROOT / "s2_tool_schemas.json").read_text(encoding="utf-8"))
    assert {item["name"]: item["inputSchema"] for item in published["tools"]} == {
        name: tool.input_schema for name, tool in live.items()
    }
    assert {item["name"]: item["description"] for item in published["tools"]} == {
        name: tool.description for name, tool in live.items()
    }

    catalog = json.loads((ROOT / "docs/agent/tools.json").read_text(encoding="utf-8"))
    entries = {item["name"]: item for item in catalog["tools"]}
    for name, tool in live.items():
        assert entries[name]["parameters"] == tool.input_schema

    feedback = live["save_workout_feedback"].input_schema
    assert feedback["properties"]["difficulty"]["enum"] == ["easy", "moderate", "hard"]
    assert feedback["properties"]["notes"]["default"] == ""
    assert feedback["properties"]["notes"]["maxLength"] == tools.MAX_NOTES_LENGTH
    assert feedback["required"] == ["user_id", "workout_id", "difficulty"]


def test_mcp_demo_runs_end_to_end_over_stdio(tmp_path):
    pytest.importorskip("fastmcp")
    completed = subprocess.run(
        [sys.executable, "-m", "agentic.mcp_demo_client"],
        cwd=ROOT,
        env={
            **os.environ,
            "REPWISE_DATA_DIR": str(tmp_path / "demo"),
            "REPWISE_LOG_DIR": str(tmp_path / "logs"),
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Demonstration passed." in completed.stdout
    assert (tmp_path / "demo" / DATABASE_NAME).exists()
