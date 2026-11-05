"""Verify that the agent reports success only after a real tool result."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace

from mcp import StdioServerParameters
from mcp.types import CallToolResult, Tool
from ollama import Message
import pytest

from src.agent import mcp_client
from src.schemas.models import PlanOptions


def completed():
    return {
        "status": "completed",
        "message": "Saved",
        "plan_id": "saved-123",
        "sources": [],
        "plan": {
            "title": "Leg program",
            "summary": "Beginner program",
            "safety_note": "Stop for pain",
            "weeks": [
                {
                    "number": 1,
                    "objective": "Build strength",
                    "workouts": [
                        {
                            "day": "Monday",
                            "focus": "Legs",
                            "estimated_minutes": 30,
                            "exercises": [
                                {
                                    "name": "Bodyweight squat",
                                    "sets": 3,
                                    "repetitions": "8-12",
                                    "rest_seconds": 60,
                                    "estimated_minutes": 10,
                                    "coaching_note": "Controlled movement",
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    }


def setup_agent(monkeypatch, messages, result=None):
    calls = []
    prompts = []

    @asynccontextmanager
    async def transport(*args, **kwargs):
        yield None, None

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    Tool(
                        name="generate_workout_plan",
                        inputSchema=PlanOptions.model_json_schema(),
                    )
                ]
            )

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return CallToolResult(content=[], structuredContent=result or completed())

    class Model:
        def __init__(self, **kwargs):
            pass

        async def chat(self, **kwargs):
            prompts.append(deepcopy(kwargs))
            return SimpleNamespace(message=messages.pop(0))

    monkeypatch.setattr(mcp_client, "logged_stdio", transport)
    monkeypatch.setattr(mcp_client, "ClientSession", Session)
    monkeypatch.setattr(mcp_client, "AsyncClient", Model)
    return calls, prompts


def call_message():
    return Message(
        role="assistant",
        tool_calls=[
            {
                "function": {
                    "name": "generate_workout_plan",
                    "arguments": {"focus": "Build legs", "workouts_per_week": 1},
                }
            }
        ],
    )


def run():
    return asyncio.run(
        mcp_client.run_workout_agent(
            "Build my leg program",
            StdioServerParameters(
                command="unused", env={"REPWISE_MCP_ACCESS_TOKEN": "private-token"}
            ),
            model="test-model",
            verbose=False,
        )
    )


def test_success_requires_mcp_call_and_validated_saved_result(monkeypatch):
    calls, prompts = setup_agent(
        monkeypatch,
        [call_message(), Message(role="assistant", content="Your plan was saved.")],
    )
    result = run()
    assert len(calls) == 1
    assert result["result"]["plan_id"] == "saved-123"
    assert result["steps"] == 2
    assert any(message["role"] == "tool" for message in prompts[1]["messages"])
    assert "private-token" not in str(prompts)


def test_text_without_tool_use_is_not_success(monkeypatch):
    calls, _ = setup_agent(
        monkeypatch, [Message(role="assistant", content="I made a plan.")]
    )
    with pytest.raises(RuntimeError, match="without making"):
        run()
    assert calls == []


def test_agent_does_not_repeat_a_write_tool(monkeypatch):
    calls, _ = setup_agent(monkeypatch, [call_message(), call_message()])
    with pytest.raises(RuntimeError, match="duplicate"):
        run()
    assert len(calls) == 1


def test_pipeline_failure_cannot_be_claimed_as_success(monkeypatch):
    calls, _ = setup_agent(
        monkeypatch,
        [call_message()],
        result={
            "status": "retrieval_error",
            "message": "Unavailable",
            "plan": None,
            "sources": [],
        },
    )
    with pytest.raises(RuntimeError, match="retrieval_error"):
        run()
    assert len(calls) == 1


def test_invalid_model_arguments_are_repaired_before_any_mcp_write(monkeypatch):
    invalid = Message(
        role="assistant",
        tool_calls=[
            {
                "function": {
                    "name": "generate_workout_plan",
                    "arguments": {
                        "focus": "Build legs",
                        "extra_constraints": "not an array",
                    },
                }
            }
        ],
    )
    calls, prompts = setup_agent(
        monkeypatch,
        [invalid, call_message(), Message(role="assistant", content="Saved.")],
    )
    result = run()
    assert len(calls) == 1
    assert result["steps"] == 3
    assert "rejected before execution" in prompts[1]["messages"][-1]["content"]
