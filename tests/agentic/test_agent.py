"""Tests for the LangGraph agent, using a scripted model and no network."""

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agentic import agent as agent_module
from agentic import places, tools
from agentic.agent import (
    LIMIT_MESSAGE,
    MAX_TOOL_STEPS,
    MODEL_FAILURE_MESSAGE,
    ask,
    build_agent,
)
from agentic.agent_tools import AgentContext, build_tools
from agentic.mcp_demo_client import seed_demo_account
from agentic.planning import Plan, PlanStep
from src.schemas.models import PlanResult, TrainingPlan

LOCATION = places.Location(label="Test", latitude=48.79, longitude=2.33)


class ScriptedPlanner:
    """Give each plan in the list in turn, then the default one-step plan."""

    def __init__(self, plans):
        self.plans = plans

    def invoke(self, _messages):
        if self.plans:
            return self.plans.pop(0)
        return Plan(steps=[PlanStep(task="Answer the request.")])


class ScriptedModel:
    """Return prepared replies in order, and record what the model was shown."""

    def __init__(self, *replies, plans=()):
        self.replies = list(replies)
        self.plans = list(plans)
        self.seen = []
        self.bound_tools = []

    def bind_tools(self, bound):
        self.bound_tools = bound
        return self

    def with_structured_output(self, _schema):
        return ScriptedPlanner(self.plans)

    def invoke(self, messages):
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def call(name, step=0, **arguments):
    """Build a model reply that asks for one tool."""
    return AIMessage(
        "", tool_calls=[{"name": name, "args": arguments, "id": f"{name}-{step}"}]
    )


@pytest.fixture
def account(tmp_path, monkeypatch):
    path = tmp_path / "agent.sqlite3"
    monkeypatch.setenv("REPWISE_DATABASE_PATH", str(path))
    return seed_demo_account(path)


def test_model_cannot_choose_the_student(account):
    schemas = {item.name: item.args for item in build_tools(AgentContext(account[0]))}

    assert set(schemas) == {
        "get_user_profile",
        "get_timetable",
        "save_workout_feedback",
        "find_nearby_places",
        "generate_workout_plan",
        "get_workout_history",
        "log_workout_performance",
        "update_upcoming_workout",
        "organize_schedule",
        "save_recommendation",
        "remember_fact",
        "search_documents",
        "get_calendar_busy",
        "propose_training_slots",
        "add_workouts_to_calendar",
    }
    assert all("student_id" not in arguments for arguments in schemas.values())


def test_model_is_not_offered_the_plan_focus(account):
    plan_tool = next(
        item
        for item in build_tools(AgentContext(account[0]))
        if item.name == "generate_workout_plan"
    )

    assert "focus" not in plan_tool.tool_call_schema.model_json_schema()["properties"]


def test_tool_uses_the_session_student_and_loop_answers(account):
    model = ScriptedModel(call("get_timetable"), AIMessage("Lundi, mercredi."))
    graph = build_agent(AgentContext(account[0]), model)

    reply = ask(graph, "Mes séances ?", thread_id="t")

    assert reply.answer == "Lundi, mercredi."
    assert reply.tool_calls == [("get_timetable", {})]
    assert f'"student_id":"{account[0]}"' in reply.tool_results[0][1]
    assert isinstance(model.seen[1][-1], ToolMessage)


def test_conversation_is_remembered_only_within_its_thread(account):
    model = ScriptedModel(AIMessage("a"), AIMessage("b"), AIMessage("c"))
    graph = build_agent(AgentContext(account[0]), model)

    ask(graph, "first", thread_id="one")
    ask(graph, "second", thread_id="one")
    ask(graph, "third", thread_id="two")

    shown = [[m.content for m in messages[1:]] for messages in model.seen]
    assert shown[1] == ["first", "a", "second"]
    assert shown[2] == ["third"]


def test_step_limit_stops_the_loop_and_conversation_can_continue(account):
    calls = [call("get_timetable", step) for step in range(MAX_TOOL_STEPS + 1)]
    model = ScriptedModel(*calls, AIMessage("still here"))
    graph = build_agent(AgentContext(account[0]), model)

    reply = ask(graph, "loop", thread_id="t")

    assert reply.answer == LIMIT_MESSAGE
    assert (
        len([r for r in reply.tool_results if r[1].startswith("{")]) == MAX_TOOL_STEPS
    )
    assert ask(graph, "hello", thread_id="t").answer == "still here"


def test_time_limit_stops_before_more_tools_run(account, monkeypatch):
    monkeypatch.setattr(agent_module, "MAX_SECONDS", -1)
    model = ScriptedModel(call("get_timetable"))
    graph = build_agent(AgentContext(account[0]), model)

    reply = ask(graph, "slow", thread_id="t")

    assert reply.answer == LIMIT_MESSAGE
    assert reply.tool_results == [("get_timetable", "Not run: limit reached.")]


def test_invalid_arguments_go_back_to_the_model(account):
    model = ScriptedModel(call("find_nearby_places", category="zoo"), AIMessage("ok"))
    graph = build_agent(AgentContext(account[0], LOCATION), model)

    reply = ask(graph, "zoo", thread_id="t")

    assert reply.answer == "ok"
    assert reply.tool_results[0][1].startswith("Invalid tool call")


def test_model_failure_is_reported(account):
    model = ScriptedModel(httpx.ConnectError("down"))
    graph = build_agent(AgentContext(account[0]), model)

    assert ask(graph, "hi", thread_id="t").answer == MODEL_FAILURE_MESSAGE


def feedback_call(account, comment):
    return call(
        "save_workout_feedback",
        plan_id=account[1],
        week_number=1,
        workout_index=0,
        completion_status="completed",
        difficulty="just_right",
        comment=comment,
    )


def saved_feedback(student_id):
    timetable = tools.get_timetable(student_id)
    return [item.feedback for item in timetable.planned_workouts if item.feedback]


@pytest.mark.parametrize("comment", ["My knee hurts", "J'ai une douleur au genou"])
def test_pain_feedback_is_not_saved_when_declined(account, comment):
    model = ScriptedModel(feedback_call(account, comment), AIMessage("Not saved."))
    graph = build_agent(AgentContext(account[0]), model)

    reply = ask(graph, "log it", thread_id="t", approve=lambda _request: False)

    assert saved_feedback(account[0]) == []
    assert reply.approvals[0][1] is False
    assert "declined" in reply.tool_results[0][1]
    assert reply.answer == "Not saved."


def test_pain_feedback_is_saved_when_approved(account):
    model = ScriptedModel(feedback_call(account, "My knee hurts"), AIMessage("Saved."))
    graph = build_agent(AgentContext(account[0]), model)
    requests = []

    ask(
        graph,
        "log it",
        thread_id="t",
        approve=lambda request: requests.append(request) or True,
    )

    assert len(saved_feedback(account[0])) == 1
    assert "knee hurts" in requests[0]["reason"]


def test_ordinary_feedback_needs_no_approval(account):
    model = ScriptedModel(feedback_call(account, "Good session"), AIMessage("Saved."))
    graph = build_agent(AgentContext(account[0]), model)

    reply = ask(graph, "log it", thread_id="t")

    assert reply.approvals == []
    assert len(saved_feedback(account[0])) == 1


class StubPipeline:
    """Return the synthetic plan, and count generations."""

    def __init__(self, status="completed"):
        self.status = status
        self.calls = 0

    def create_plan(self, request, *, previous_plan=None):
        self.calls += 1
        self.focus = request.focus
        plan = TrainingPlan.model_validate(
            tools.open_repository().list_plans(self.student_id, limit=1)[0]["plan"]
        )
        if self.status != "completed":
            return PlanResult(status=self.status, message="Out of scope.")
        return PlanResult(status="completed", message="Done.", plan=plan)


def plan_agent(account, pipeline):
    pipeline.student_id = account[0]
    model = ScriptedModel(
        call("generate_workout_plan", focus="Build leg muscles"),
        AIMessage("Nothing was changed."),
    )
    context = AgentContext(account[0], pipeline_factory=lambda: pipeline)
    return build_agent(context, model), model


def test_plan_replacing_an_active_one_needs_approval(account):
    pipeline = StubPipeline()
    graph, _model = plan_agent(account, pipeline)

    reply = ask(graph, "legs", thread_id="t", approve=lambda _request: False)

    assert pipeline.calls == 0
    assert reply.answer == "Nothing was changed."


def test_plan_workflow_ends_without_another_model_call(account):
    pipeline = StubPipeline()
    graph, model = plan_agent(account, pipeline)
    before = len(tools.open_repository().list_plans(account[0]))

    reply = ask(graph, "Build my legs", thread_id="t", approve=lambda _request: True)

    assert pipeline.calls == 1
    # The script asked for "Build leg muscles": the student's request wins.
    assert pipeline.focus == "Build my legs"
    assert len(model.seen) == 1
    assert "Goblet squat" in reply.answer
    assert len(tools.open_repository().list_plans(account[0])) == before + 1


def test_refused_plan_saves_nothing_and_says_so(account):
    pipeline = StubPipeline(status="refused")
    graph, _model = plan_agent(account, pipeline)
    before = len(tools.open_repository().list_plans(account[0]))

    reply = ask(graph, "legs", thread_id="t", approve=lambda _request: True)

    assert reply.answer.startswith("No plan was saved (refused)")
    assert len(tools.open_repository().list_plans(account[0])) == before


def test_missing_corpus_is_reported_as_unavailable(account):
    def broken():
        raise FileNotFoundError("no corpus")

    model = ScriptedModel(call("generate_workout_plan", focus="Build leg muscles"))
    graph = build_agent(AgentContext(account[0], pipeline_factory=broken), model)

    reply = ask(graph, "legs", thread_id="t", approve=lambda _request: True)

    assert reply.answer.startswith("No plan was saved (unavailable)")


def feature(name, latitude, longitude=2.33, **properties):
    return {
        "properties": {"name": name, **properties},
        "geometry": {"coordinates": [longitude, latitude]},
    }


def photon_reply(url, status=200, features=()):
    return httpx.Response(
        status, json={"features": list(features)}, request=httpx.Request("GET", url)
    )


def test_places_are_ranked_and_filtered(monkeypatch):
    features = [
        feature("Far Gym", 48.80, street="Rue A"),
        feature("Near Gym", 48.791),
        feature(None, 48.791),
        feature("Out of range", 50.0),
    ]
    monkeypatch.setattr(
        places.httpx, "get", lambda url, **_: photon_reply(url, features=features)
    )

    result = places.find_nearby_places("gym", LOCATION)

    assert [place.name for place in result.places] == ["Near Gym", "Far Gym"]
    assert result.places[1].address == "Rue A"
    assert "stock" in result.note


def test_every_supplement_tag_is_searched(monkeypatch):
    searched = []

    def get(url, params, **_):
        searched.append(params["osm_tag"])
        return photon_reply(url, features=[feature(params["osm_tag"], 48.791)])

    monkeypatch.setattr(places.httpx, "get", get)

    result = places.find_nearby_places("supplement_store", LOCATION)

    assert searched == ["shop:nutrition_supplements", "shop:health_food"]
    assert len(result.places) == 2


@pytest.mark.parametrize("failure", ["status", "network"])
def test_places_report_failure_when_the_service_is_down(monkeypatch, failure):
    def get(url, **_):
        if failure == "network":
            raise httpx.ConnectError("down")
        return photon_reply(url, status=502)

    monkeypatch.setattr(places.httpx, "get", get)

    assert places.find_nearby_places("gym", LOCATION).code == "places_unavailable"


def test_places_never_guess_an_unknown_location(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("no location should be searched")

    monkeypatch.setattr(places.httpx, "get", forbidden)

    assert places.find_nearby_places("gym", None).code == "location_unknown"
