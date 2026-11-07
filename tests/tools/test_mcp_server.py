"""The MCP server: authentication at startup, tool discovery and execution."""

import asyncio
import json

import pytest
from fastmcp import Client

from config.config import AppConfig
from scripts import mcp_smoke
from src.api.auth import create_access_token
from src.mcp_server.server import StartupError, authenticate, create_server
from src.tools.catalog import load_catalog

SECRET = "x" * 64


def run(coroutine):
    return asyncio.run(coroutine)


def config_for(demo):
    return AppConfig(database_path=demo.repository.database_path, auth_secret=SECRET)


def test_every_catalog_tool_is_discovered_without_account_arguments(demo):
    async def scenario():
        async with Client(create_server(demo.context)) as client:
            return await client.list_tools()

    tools = run(scenario())

    assert {tool.name for tool in tools} == {tool.name for tool in load_catalog().tools}
    descriptions = {tool.name: tool.description for tool in tools}
    assert descriptions["get_timetable"] == next(
        t.description for t in load_catalog().tools if t.name == "get_timetable"
    )
    assert not any("student_id" in json.dumps(tool.input_schema) for tool in tools)


def test_tools_run_for_the_bound_account_and_writes_wait_for_approval(demo):
    async def scenario():
        async with Client(create_server(demo.context)) as client:
            timetable = await client.call_tool("get_timetable", {})
            held = await client.call_tool(
                "organize_schedule",
                {
                    "moves": [
                        {"week_number": 1, "workout_index": 0, "new_day": "Friday"}
                    ]
                },
            )
            invalid = await client.call_tool(
                "get_workout",
                {"week_number": 9, "workout_index": 0},
                raise_on_error=False,
            )
            return timetable.structured_content, held.structured_content, invalid

    timetable, held, invalid = run(scenario())

    assert timetable["plan_id"] == demo.plan_id
    assert held["status"] == "pending_approval"
    assert (
        demo.repository.list_pending_actions(demo.student_id)[0]["proposed_by"] == "mcp"
    )
    # Nothing moved: the plan still has Monday first.
    assert (
        demo.repository.get_plan(demo.student_id, demo.plan_id)["plan"]["weeks"][0][
            "workouts"
        ][0]["day"]
        == "Monday"
    )
    assert (
        invalid.is_error and invalid.structured_content["code"] == "workout_not_found"
    )


def test_schema_validation_rejects_bad_arguments_before_running(demo):
    async def scenario():
        async with Client(create_server(demo.context)) as client:
            return await client.call_tool(
                "save_workout_feedback",
                {
                    "week_number": 1,
                    "workout_index": 0,
                    "completion_status": "completed",
                    "difficulty": "hard",
                },
                raise_on_error=False,
            )

    result = run(scenario())

    assert result.is_error
    assert demo.repository.list_workout_feedback(demo.student_id, demo.plan_id) == []


def test_a_valid_token_binds_its_account(demo):
    student_id, _repository = authenticate(
        config_for(demo), create_access_token(demo.student_id, SECRET)
    )

    assert student_id == demo.student_id


@pytest.mark.parametrize(
    ("token", "message"),
    [
        ("", "REPWISE_MCP_ACCESS_TOKEN"),
        ("not-a-token", "invalid or expired"),
        (create_access_token("deleted-account", SECRET), "no longer exists"),
        (create_access_token("someone", "another-secret" * 4), "invalid or expired"),
    ],
)
def test_startup_is_refused_without_a_valid_token(demo, token, message):
    with pytest.raises(StartupError, match=message):
        authenticate(config_for(demo), token)


def test_startup_is_refused_without_a_secret_or_database(demo, tmp_path):
    token = create_access_token(demo.student_id, SECRET)
    with pytest.raises(StartupError, match="REPWISE_AUTH_SECRET"):
        authenticate(AppConfig(database_path=demo.repository.database_path), token)
    with pytest.raises(StartupError, match="Database not found"):
        authenticate(
            AppConfig(database_path=str(tmp_path / "none.sqlite3"), auth_secret=SECRET),
            token,
        )


def test_server_py_runs_over_stdio_end_to_end():
    """Launch the real entry point as a subprocess, like an MCP host does."""
    assert mcp_smoke.main() == 0
