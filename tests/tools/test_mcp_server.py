"""Check the real MCP tool boundary without requiring Ollama."""

import asyncio
import json

import httpx
import pytest

from src.mcp_server.server import create_server


def test_discovery_exposes_structured_plan_without_credentials():
    tools = asyncio.run(create_server("http://backend", "owner", "private-token").list_tools())
    assert [tool.name for tool in tools] == ["generate_workout_plan"]
    tool = tools[0]
    assert "focus" in tool.inputSchema["properties"]
    assert "user_id" not in tool.inputSchema["properties"]
    assert "access_token" not in tool.inputSchema["properties"]
    assert "private-token" not in tool.model_dump_json()
    assert "plan_id" in tool.outputSchema["properties"]


def test_tool_forwards_validated_arguments_and_bound_account():
    calls = []

    def backend(request):
        calls.append(request)
        assert request.url.path == "/api/v1/students/owner/plans"
        assert request.headers["Authorization"] == "Bearer private-token"
        assert json.loads(request.content)["workouts_per_week"] == 1
        return httpx.Response(200, json={"status": "out_of_scope", "message": "Not supported"})

    server = create_server("http://backend", "owner", "private-token", transport=httpx.MockTransport(backend))
    content, structured = asyncio.run(server.call_tool("generate_workout_plan", {
        "focus": "Build leg muscles", "workouts_per_week": 1,
    }))
    assert len(calls) == 1
    assert structured["status"] == "out_of_scope"
    assert structured["plan_id"] is None


def test_invalid_arguments_do_not_reach_backend():
    def backend(request):
        pytest.fail("Invalid arguments reached backend")

    server = create_server("http://backend", "owner", "token", transport=httpx.MockTransport(backend))
    with pytest.raises(Exception, match="workouts_per_week"):
        asyncio.run(server.call_tool("generate_workout_plan", {"focus": "Build legs", "workouts_per_week": 8}))


def test_unauthorized_backend_is_an_error_without_leaking_token():
    server = create_server("http://backend", "owner", "private-token", transport=httpx.MockTransport(
        lambda request: httpx.Response(401, json={"detail": "Expired token"})
    ))
    with pytest.raises(Exception, match="HTTP 401") as error:
        asyncio.run(server.call_tool("generate_workout_plan", {"focus": "Build legs"}))
    assert "private-token" not in str(error.value)


def test_missing_profile_and_pipeline_failure_stay_structured():
    for status_code, detail, expected in [
        (404, "Save the student profile before creating a plan", "missing_profile"),
        (503, {"status": "retrieval_error", "message": "Index unavailable"}, "retrieval_error"),
    ]:
        server = create_server("http://backend", "owner", "token", transport=httpx.MockTransport(
            lambda request: httpx.Response(status_code, json={"detail": detail})
        ))
        _, structured = asyncio.run(server.call_tool("generate_workout_plan", {"focus": "Build legs"}))
        assert structured["status"] == expected
        assert structured["plan_id"] is None
