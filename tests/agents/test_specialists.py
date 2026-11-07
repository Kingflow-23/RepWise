"""The bounded tool loop every specialist runs, with a scripted model."""

from time import monotonic

import httpx
from langchain_core.messages import AIMessage, ToolMessage

from src.agents.specialists import SPECIALISTS, run_specialist
from tests.conftest import ScriptedModel, tool_call


def run(demo, agent, model, task="Do the task.", deadline=None):
    return run_specialist(
        SPECIALISTS[agent],
        task,
        "Context: test.",
        demo.context,
        model,
        deadline or monotonic() + 60,
    )


def test_a_specialist_calls_its_tool_then_reports(demo):
    model = ScriptedModel(
        [tool_call("get_workout_history"), AIMessage("Two workouts logged.")]
    )

    report = run(demo, "performance", model)

    assert report.status == "completed"
    assert report.summary == "Two workouts logged."
    assert [item["name"] for item in report.tools_used] == ["get_workout_history"]
    assert report.results["get_workout_history"]["ok"] is True
    # The tool result reached the model before its final answer.
    assert any(
        isinstance(m, ToolMessage) and m.name == "get_workout_history"
        for m in model.seen[-1]
    )


def test_a_specialist_is_only_offered_and_allowed_its_own_tools(demo):
    model = ScriptedModel(
        [tool_call("organize_schedule", moves=[]), AIMessage("Could not.")]
    )

    report = run(demo, "knowledge", model)

    assert model.bound == [["search_documents"]]
    assert [(t["name"], t["ok"], t["code"]) for t in report.tools_used] == [
        ("organize_schedule", False, "not_allowed")
    ]
    assert report.status == "failed"


def test_a_write_needing_approval_comes_back_pending(demo):
    move = {"week_number": 1, "workout_index": 0, "new_day": "Friday"}
    model = ScriptedModel(
        [
            tool_call("organize_schedule", moves=[move]),
            AIMessage("Waiting for approval."),
        ]
    )

    report = run(demo, "schedule", model)

    assert report.status == "completed"
    assert [action.tool_name for action in report.pending_actions] == [
        "organize_schedule"
    ]
    assert (
        demo.repository.get_plan(demo.student_id, demo.plan_id)["plan"]["weeks"][0][
            "workouts"
        ][0]["day"]
        == "Monday"
    )


def test_the_tool_call_limit_stops_the_loop(demo):
    calls = [tool_call("get_workout_history", call_id=f"c{i}") for i in range(10)]

    report = run(demo, "performance", ScriptedModel(calls))

    assert report.status == "limit_reached"
    assert len(report.tools_used) == demo.context.config.agent_limits.max_tool_calls
    assert "Stopped after" in report.error


def test_the_time_limit_stops_before_more_tools_run(demo):
    report = run(
        demo,
        "performance",
        ScriptedModel([tool_call("get_workout_history")]),
        deadline=monotonic() - 1,
    )

    assert report.status == "limit_reached"
    assert report.tools_used == []


def test_an_unreachable_model_is_a_failed_step(demo):
    report = run(demo, "performance", ScriptedModel([httpx.ConnectError("down")]))

    assert report.status == "failed"
    assert "could not be reached" in report.error


def test_a_step_where_every_tool_failed_is_failed(demo):
    model = ScriptedModel([tool_call("get_calendar_busy"), AIMessage("No calendar.")])

    report = run(demo, "schedule", model)

    assert report.status == "failed"
    assert "calendar_not_connected" in report.error


def test_invalid_arguments_go_back_to_the_model(demo):
    model = ScriptedModel(
        [
            tool_call("get_workout_history", max_plans=999),
            tool_call("get_workout_history", call_id="c2"),
            AIMessage("ok"),
        ]
    )

    report = run(demo, "performance", model)

    assert [item["code"] for item in report.tools_used] == ["invalid_arguments", None]
    assert report.status == "completed"
