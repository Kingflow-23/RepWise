"""Tests for Plan-and-Execute and the agent's memory, using a scripted model."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentic import memory, tools
from agentic.agent import ask, build_agent
from agentic.agent_tools import AgentContext
from agentic.mcp_demo_client import seed_demo_account
from agentic.planning import Plan, PlanStep


class ScriptedPlanner:
    """Give each plan in turn, then a plan with one step and no required tool."""

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

    def bind_tools(self, _bound):
        return self

    def with_structured_output(self, _schema):
        return ScriptedPlanner(self.plans)

    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def call(name, **arguments):
    return AIMessage(
        "", tool_calls=[{"name": name, "args": arguments, "id": f"{name}-1"}]
    )


def steps(*tool_names):
    """Build a plan with one step per tool."""
    return Plan(steps=[PlanStep(task=f"Use {name}", tool=name) for name in tool_names])


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


@pytest.fixture
def account(tmp_path, monkeypatch):
    path = tmp_path / "agent.sqlite3"
    monkeypatch.setenv("REPWISE_DATABASE_PATH", str(path))
    return seed_demo_account(path)


def session(account, *replies, plans=()):
    """Build a new graph, so only what the database holds carries over."""
    model = ScriptedModel(*replies, plans=plans)
    return build_agent(AgentContext(account[0]), model), model


# ------------------------------------------------------------- plan and execute


def test_steps_run_in_order_and_the_last_one_answers(account):
    graph, model = session(
        account,
        call("get_user_profile"),
        AIMessage("Profile read."),
        call("get_timetable"),
        AIMessage("Full answer."),
        plans=[steps("get_user_profile", "get_timetable")],
    )

    reply = ask(graph, "Both, please", thread_id="t")

    assert reply.answer == "Full answer."
    assert reply.plan == ["Use get_user_profile", "Use get_timetable"]
    assert [name for name, _ in reply.tool_calls] == ["get_user_profile", "get_timetable"]
    assert "You are on step 1." in model.seen[0][0].content
    assert "You are on step 2." in model.seen[2][0].content


def test_the_next_step_never_sees_an_earlier_step_s_text_reply(account):
    graph, model = session(
        account,
        call("get_user_profile"),
        AIMessage("Profile read."),
        AIMessage("Full answer."),
        plans=[steps("get_user_profile", "get_timetable")],
    )
    model.replies.insert(2, call("get_timetable"))

    ask(graph, "Both, please", thread_id="t")

    assert isinstance(model.seen[2][-1], ToolMessage)
    assert "Profile read." not in [m.content for m in model.seen[2]]


def test_a_rejected_claim_is_hidden_from_the_retry(account):
    graph, model = session(
        account,
        AIMessage("I saved it."),
        AIMessage("I saved it again."),
        plans=[steps("save_recommendation"), steps("save_recommendation")],
    )

    ask(graph, "Keep this idea", thread_id="t")

    assert [type(m) for m in model.seen[1][1:]] == [HumanMessage]


def test_earlier_requests_keep_their_answers(account):
    graph, model = session(account, AIMessage("First answer."), AIMessage("Second."))

    ask(graph, "first", thread_id="t")
    ask(graph, "second", thread_id="t")

    assert "First answer." in [m.content for m in model.seen[1]]


def test_a_claim_without_its_tool_revises_the_plan(account):
    graph, _model = session(
        account,
        AIMessage("I saved it."),
        call("save_recommendation", summary="More squats", reason="To progress"),
        AIMessage("Really saved."),
        plans=[steps("save_recommendation"), steps("save_recommendation")],
    )

    reply = ask(graph, "Keep this idea", thread_id="t")

    assert reply.revisions == 1
    assert reply.answer == "Really saved."
    assert len(tools.open_repository().list_recommendations(account[0])) == 1


def test_a_step_that_still_fails_is_reported_whatever_the_model_says(account):
    graph, _model = session(
        account,
        AIMessage("I saved it."),
        AIMessage("I saved it again."),
        plans=[steps("save_recommendation"), steps("save_recommendation")],
    )

    reply = ask(graph, "Keep this idea", thread_id="t")

    assert reply.revisions == 1
    assert "did not complete" in reply.answer
    assert tools.open_repository().list_recommendations(account[0]) == []


def test_a_failed_tool_counts_as_a_failed_step(account):
    graph, _model = session(
        account,
        call("find_nearby_places", category="gym"),
        AIMessage("No location."),
        AIMessage("Still no location."),
        plans=[steps("find_nearby_places"), steps("find_nearby_places")],
    )

    reply = ask(graph, "Nearest gym?", thread_id="t")

    assert reply.revisions == 1
    assert "did not complete" in reply.answer


def test_a_declined_action_is_not_replanned(account):
    graph, _model = session(
        account,
        feedback_call(account, "My knee hurts"),
        AIMessage("Nothing was saved."),
        plans=[steps("save_workout_feedback", "get_timetable")],
    )

    reply = ask(graph, "log it", thread_id="t", approve=lambda _request: False)

    assert reply.answer == "Nothing was saved."
    assert reply.revisions == 0


def test_an_unusable_plan_becomes_one_step_for_the_whole_request(account):
    class NoPlanModel(ScriptedModel):
        def with_structured_output(self, _schema):
            return self

        def invoke(self, messages):
            if "You plan the work" in str(messages[0].content):
                raise ValueError("not JSON")
            return super().invoke(messages)

    graph = build_agent(AgentContext(account[0]), NoPlanModel(AIMessage("Answered.")))

    reply = ask(graph, "hi", thread_id="t")

    assert reply.answer == "Answered."
    assert reply.plan == ["Complete the student's request."]


def test_a_made_up_tool_in_a_plan_is_ignored(account):
    graph, _model = session(
        account, AIMessage("Answer."), plans=[steps("invented_tool")]
    )

    reply = ask(graph, "hi", thread_id="t")

    assert reply.answer == "Answer."
    assert reply.revisions == 0


# ------------------------------------------------------------- semantic memory


def test_a_remembered_fact_reaches_a_new_session(account):
    graph, _model = session(
        account,
        call("remember_fact", fact="Prefers morning workouts"),
        AIMessage("Noted."),
    )
    ask(graph, "Remember this", thread_id="t")

    graph, model = session(account, AIMessage("In the morning."))
    ask(graph, "When do I like to train?", thread_id="other")

    assert "Prefers morning workouts" in model.seen[0][0].content


def test_saving_a_fact_twice_keeps_one_copy(account):
    memory.remember_fact(account[0], "Prefers mornings")
    memory.remember_fact(account[0], "Prefers mornings")

    assert tools.open_repository().list_memory_facts(account[0], 20) == [
        "Prefers mornings"
    ]


@pytest.mark.parametrize("fact", ["My knee hurts", "J'ai mal au genou"])
def test_health_details_are_not_remembered(account, fact):
    result = memory.remember_fact(account[0], fact)

    assert result.code == "health_not_stored"
    assert memory.recall(account[0]) == ""


# ------------------------------------------------------------- episodic memory


def test_a_completed_task_reaches_a_new_session(account):
    graph, _model = session(account, call("get_timetable"), AIMessage("Monday."))
    ask(graph, "My workouts?", thread_id="t")

    graph, model = session(account, AIMessage("ok"))
    ask(graph, "hello", thread_id="other")

    assert "Asked: My workouts?. Completed: get_timetable" in model.seen[0][0].content


def test_a_request_without_tools_leaves_no_episode(account):
    graph, _model = session(account, AIMessage("Hello."))

    ask(graph, "hi", thread_id="t")

    assert memory.recall(account[0]) == ""


def test_a_declined_action_is_recorded_as_a_decision(account):
    graph, _model = session(
        account, feedback_call(account, "My knee hurts"), AIMessage("Not saved.")
    )

    ask(graph, "Log my workout", thread_id="t", approve=lambda _request: False)

    recalled = memory.recall(account[0])
    assert "Declined by the student: save_workout_feedback" in recalled
    assert "Completed" not in recalled


def test_an_episode_leaves_out_a_health_detail(account):
    graph, _model = session(account, call("get_timetable"), AIMessage("ok"))

    ask(graph, "My knee hurts, what is planned?", thread_id="t")

    recalled = memory.recall(account[0])
    assert "knee" not in recalled
    assert "health-related" in recalled


# ------------------------------------------------------------- whose memory it is


def test_memory_is_private_to_each_student(account):
    memory.remember_fact(account[0], "Prefers mornings")
    memory.save_episode(account[0], "Asked: something")

    other = tools.open_repository().create_user("Another Student", "hash")

    assert memory.recall(other) == ""


def test_deleting_an_account_removes_its_memory(account):
    memory.remember_fact(account[0], "Prefers mornings")
    memory.save_episode(account[0], "Asked: something")
    assert memory.recall(account[0])

    with tools.open_repository()._connect() as connection:
        connection.execute("DELETE FROM users WHERE user_id=?", (account[0],))

    assert memory.recall(account[0]) == ""
