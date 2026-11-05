"""RepWise's agent: Plan-and-Execute in LangGraph.

    plan --> execute --> check --> remember, ending with the last step's answer

    plan: load memory, then break the request into steps
    execute --> tools --> execute, until the step's work is done
    execute --> approval --> tools, or back to execute if declined
    execute --> stop, when the step or time limit is reached
    check: a passed step moves to the next, and a failed step revises the plan once
    remember: save what happened in this request

Two kinds of request skip part of this, because code does them more reliably:
training times run the slot tool first (``offer_slots``), and a generated plan is shown
as it was validated (``finish_plan``).

Working memory is this graph's state, kept per thread until the process ends. Semantic
and episodic memory are in ``memory.py``. Planning and checking are in ``planning.py``,
and the tools are in ``agent_tools.py``.

To debug a tool without a model, use ``python -m agentic.call_tool``.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic, perf_counter
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from ollama import ResponseError
from pydantic import ValidationError

from agentic import memory, planning
from agentic.agent_tools import (
    PLAN_TOOL,
    SLOTS_TOOL,
    AgentContext,
    approval_reason,
    build_tools,
)
from agentic.planning import DECLINED_MESSAGE, Plan
from config.config import AppConfig
from src.tools.generate_workout_plan import GenerateWorkoutPlanResult

logger = logging.getLogger(__name__)

MAX_TOOL_STEPS = 5
MAX_SECONDS = 180
MAX_REVISIONS = 1
# Ollama's default of 4096 tokens is too small: the tool schemas and a timetable fill
# it, and the model then stops mid-sentence.
CONTEXT_TOKENS = 16384

SYSTEM_PROMPT = """You are RepWise's workout assistant.
Answer in the language the student writes in.
Use the tools; never guess saved data, and only name places, workouts, or plans that a tool returned.
Call the tool rather than announcing it. Act when the request gives enough information, and do not ask for what the student already said. For example, "terminée" means completed, and "un peu trop dure" means slightly_too_hard.
Never say something was saved, kept, booked, moved, or changed unless a tool confirmed it during this request.
Never work out dates or weekdays yourself: use the ones in tool results.
- Nearby gyms and shops: call find_nearby_places. For a gym use "gym", for equipment use "sports_store", for supplements use "supplement_store". Results are sorted nearest first: never reorder them or call anything but the first one the nearest. Give the name and distance of the results. If a result has try_first, say to try that shop first, without calling it the nearest. Never say a shop sells or should sell a product: say that the results do not confirm it is in stock. If it finds nothing or fails, say so plainly.
- Organizing the week or finding training slots: use propose_training_slots, whose result may already be in the conversation, and show its options with their dates and times. Say that nothing is booked yet. Only if it fails with calendar_not_connected, call get_timetable and suggest days from available_days, and no others.
- Booking: once the student has chosen, including "the first slot", call add_workouts_to_calendar with the exact start times offered. If no slots were offered yet, call propose_training_slots first. Only say a workout is booked if its outcome's status is "booked".
- Only when asked whether the student is busy or free: call get_calendar_busy, which gives times but no details of events.
- Moving workouts within the plan: call organize_schedule. To swap two workouts, give both moves in one call.
- Questions about training knowledge: call search_documents and answer only from its results, and end with a line listing the source titles. If they do not cover the question, say so. Do not use it to build a program.
- Progress or past workouts: call get_workout_history, and use its summary counts rather than counting yourself.
- Weights and reps the student reports: call log_workout_performance once, with all the exercises of that workout.
- Adjusting the next workout: when the student asks, make sure the feedback is saved, then call update_upcoming_workout with the logged workout. Report the before and after honestly, including when nothing changed.
- When the student asks to keep an idea for changing their training later: call save_recommendation, the only way to keep it. It does not change the plan.
- When the student asks you to remember a preference or fact about themselves: call remember_fact, the only way to remember it.
- A new program: call generate_workout_plan once. It uses the student's request as the program's focus. Set extra_constraints only for what the student asked for in this request or asked you to remember. Do not write a program yourself.
- Log feedback with save_workout_feedback only when the student explicitly asks.
If a tool says the student declined an action, do not retry it. Say nothing was changed."""

# A request for training times goes straight to the slot tool, because the model
# otherwise read the timetable and got the weekdays wrong. Bookings stay with the model.
ASKS_FOR_SLOTS = re.compile(
    r"\bcr[ée]neaux?\b|\bslots?\b"
    r"|\borgani[sz]\w*\b.*\b(?:s[ée]ances?|semaine|sessions?|week|workouts?)\b",
    re.IGNORECASE,
)
ASKS_FOR_BOOKING = re.compile(
    r"\b(?:r[ée]serv\w*|book\w*|ajoute\w*|add|confirm\w*)\b", re.IGNORECASE
)

LIMIT_MESSAGE = (
    "I could not reach a reliable result within my limits of "
    f"{MAX_TOOL_STEPS} tool steps and {MAX_SECONDS // 60} minutes, so I stopped. "
    "Please try a simpler request or ask for help."
)
MODEL_FAILURE_MESSAGE = (
    "I could not reach the language model, so I have no answer. Nothing was changed "
    "by this failed step. Please check that Ollama is running."
)


@dataclass
class AgentReply:
    """What happened while answering one request."""

    answer: str
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    tool_results: list[tuple[str, str]] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    revisions: int = 0
    approvals: list[tuple[str, bool]] = field(default_factory=list)
    seconds: float = 0.0


class AgentState(MessagesState):
    """Working memory: the conversation, and the current request's plan and limits."""

    tool_steps: int
    started_at: float
    plan: list[dict[str, Any]]
    current_step: int
    revisions: int
    unfinished: list[str]
    recalled: str


def build_chat_model(config: AppConfig) -> ChatOllama:
    """Create the agent's local model."""
    return ChatOllama(
        model=config.ollama_model,
        base_url=config.ollama_url,
        temperature=0,
        reasoning=False,
        num_ctx=CONTEXT_TOKENS,
        client_kwargs={"timeout": MAX_SECONDS},
    )


def wants_training_slots(text: str) -> bool:
    """Say whether a request asks for training times, rather than to book them."""
    return bool(ASKS_FOR_SLOTS.search(text)) and not ASKS_FOR_BOOKING.search(text)


def format_plan(result: GenerateWorkoutPlanResult) -> str:
    """Show a plan exactly as it was validated, without passing it through the model."""
    if result.status != "completed" or result.plan is None:
        return f"No plan was saved ({result.status}): {result.message}"
    lines = [f"**{result.plan.title}**", result.plan.summary, ""]
    for week in result.plan.weeks:
        for workout in week.workouts:
            lines.append(
                f"**{workout.day}: {workout.focus}** ({workout.estimated_minutes} min)"
            )
            lines.extend(
                f"- {item.name}: {item.sets} x {item.repetitions}, "
                f"{item.rest_seconds} s rest"
                for item in workout.exercises
            )
    lines += ["", f"Saved as plan {result.plan_id}, with {len(result.sources)} sources."]
    return "\n".join(lines)


def format_slots(output: str) -> str:
    """Show the slot tool's JSON output as text, without passing it through the model."""
    result = json.loads(output)
    if not result["ok"]:
        return f"I could not read your calendar: {result['error']}"
    lines = ["**Free times in your calendar, not booked yet:**"]
    for workout in result["workouts"]:
        # Times are already local, like "2030-01-14T07:00:00+01:00".
        times = [
            f"{slot['day']} {slot['start'][:16].replace('T', ' ')}"
            for slot in workout["options"]
        ]
        lines.append(
            f"**{workout['planned_day']}: {workout['focus']}**: "
            f"{', '.join(times) or 'no free time that week.'}"
        )
    lines.append("Tell me which times you want, and I will book them.")
    return "\n".join(lines)


def tool_notices(results: list[tuple[str, str]]) -> list[str]:
    """Write the warnings that must reach the student, whatever the model said."""
    notices = []
    for name, content in results:
        if not content.startswith("{"):
            continue
        result = json.loads(content)
        if result.get("try_first"):
            notices.append(
                f"From the shop names alone, try {result['try_first']} first. "
                "Check that it has what you need."
            )
        for workout in result.get("workouts", []):
            if name == SLOTS_TOOL and workout["moved_to_another_day"]:
                notices.append(
                    f"{workout['planned_day']} has no free time for "
                    f"\"{workout['focus']}\", so its options are on another day."
                )
        for outcome in result.get("outcomes", []):
            where = f"Week {outcome['week_number']}, workout {outcome['workout_index']}"
            if outcome["plan_day_differs"]:
                notices.append(
                    f"{where} was booked on a different day than in your plan. "
                    "The plan has not changed."
                )
            elif outcome["status"] in ("conflict", "failed"):
                notices.append(f"{where} was not booked: {outcome['detail']}")
        if result.get("calendar_events_now_out_of_date"):
            notices.append(
                "A moved workout already has a calendar event, which still shows the "
                "old time. Change it in your calendar."
            )
    return notices


def _latest_request(state: AgentState) -> str:
    return next(
        str(message.content)
        for message in reversed(state["messages"])
        if isinstance(message, HumanMessage)
    )


def _finished_plan(state: AgentState) -> ToolMessage | None:
    """Return the plan tool's result, if the last tools to run included it."""
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            return None
        if message.name == PLAN_TOOL and message.status == "success":
            return message
    return None


def _out_of_time(state: AgentState) -> bool:
    return monotonic() - state["started_at"] > MAX_SECONDS


def build_agent(context: AgentContext, llm: Any, checkpointer: Any = None):
    """Compile the graph for one signed-in student."""
    tools_by_name = {tool.name: tool for tool in build_tools(context)}
    model = llm.bind_tools(list(tools_by_name.values()))
    planner = llm.with_structured_output(Plan)

    def run_one_tool(call: dict, state: AgentState) -> ToolMessage:
        """Run one call and log it. A failure goes back to the model as text."""
        name, arguments = call["name"], call["args"]
        if name == PLAN_TOOL:
            # The sheet makes this a fixed workflow: the focus is what was asked.
            arguments = {**arguments, "focus": _latest_request(state)[:300]}
        started = perf_counter()
        status = "success"
        try:
            output = tools_by_name[name].invoke(arguments)
        except (KeyError, TypeError, ValidationError) as error:
            output, status = f"Invalid tool call: {error}", "error"
        except Exception as error:  # A tool bug must not lose the conversation.
            logger.exception("Agent tool crashed tool=%s", name)
            output, status = f"The tool failed unexpectedly: {error}", "error"
        logger.info(
            "Agent tool=%s status=%s elapsed_ms=%.0f",
            name,
            status,
            (perf_counter() - started) * 1000,
        )
        logger.debug("Agent tool=%s args=%s output=%s", name, arguments, output)
        return ToolMessage(output, tool_call_id=call["id"], name=name, status=status)

    def slot_messages(state: AgentState) -> list:
        """Run the slot tool from code, recorded as if the model had asked for it.

        The model then sees the real plan ID and times, which it needs for a booking.
        """
        call = {"name": SLOTS_TOOL, "args": {}, "id": f"code-{uuid.uuid4().hex}"}
        return [AIMessage("", tool_calls=[call]), run_one_tool(call, state)]

    def plan(state: AgentState) -> Command:
        recalled = memory.recall(context.student_id)
        # Code runs this one, so the plan needs no model call.
        if wants_training_slots(_latest_request(state)) and context.calendar:
            slot_step = planning.one_step(
                "Show the free training times from the result. Nothing is booked yet.",
                SLOTS_TOOL,
            )
            return Command(
                goto="offer_slots", update={"recalled": recalled, "plan": [slot_step]}
            )
        steps = planning.make_plan(planner, state["messages"], tools_by_name, recalled)
        logger.info("Agent plan: %s", [step["task"] for step in steps])
        return Command(goto="execute", update={"recalled": recalled, "plan": steps})

    def offer_slots(state: AgentState) -> dict:
        return {"messages": slot_messages(state), "tool_steps": state["tool_steps"] + 1}

    def execute(state: AgentState) -> dict:
        today = context.clock().astimezone(ZoneInfo(context.timezone))
        parts = [
            SYSTEM_PROMPT,
            f"Today is {today:%A %Y-%m-%d}. "
            f"The student's time zone is {context.timezone}.",
            state["recalled"],
            planning.step_instructions(state["plan"], state["current_step"]),
        ]
        prompt = "\n".join(part for part in parts if part)
        try:
            conversation = planning.messages_for_executor(state["messages"])
            response = model.invoke([SystemMessage(prompt), *conversation])
        except (ConnectionError, httpx.HTTPError, ResponseError):
            logger.exception("Agent model call failed")
            response = AIMessage(MODEL_FAILURE_MESSAGE)
        return {"messages": [response]}

    def route_after_execute(state: AgentState) -> str:
        calls = state["messages"][-1].tool_calls
        if not calls:
            return "check"
        if state["tool_steps"] + len(calls) > MAX_TOOL_STEPS or _out_of_time(state):
            return "stop"
        if any(approval_reason(c["name"], c["args"], context) for c in calls):
            return "approval"
        return "tools"

    def approval(state: AgentState) -> Command:
        calls = state["messages"][-1].tool_calls
        reasons = [approval_reason(c["name"], c["args"], context) for c in calls]
        approved = interrupt(
            {
                "reason": " ".join(reason for reason in reasons if reason),
                "calls": [{"name": c["name"], "args": c["args"]} for c in calls],
            }
        )
        # Waiting for the student does not count, so the clock restarts here.
        if approved:
            return Command(goto="tools", update={"started_at": monotonic()})
        declined = [
            ToolMessage(
                DECLINED_MESSAGE,
                tool_call_id=c["id"],
                name=c["name"],
                status="error",
            )
            for c in calls
        ]
        return Command(
            goto="execute",
            update={
                "messages": declined,
                "started_at": monotonic(),
                "tool_steps": state["tool_steps"] + len(calls),
            },
        )

    def run_tools(state: AgentState) -> dict:
        calls = state["messages"][-1].tool_calls
        return {
            "messages": [run_one_tool(call, state) for call in calls],
            "tool_steps": state["tool_steps"] + len(calls),
        }

    def route_after_tools(state: AgentState) -> str:
        if _finished_plan(state):
            return "finish_plan"
        return "stop" if _out_of_time(state) else "execute"

    def check(state: AgentState) -> Command:
        """Check the step that just ended, then choose what happens next."""
        messages = state["messages"]
        steps, index = state["plan"], state["current_step"]
        # Without a model, or after a refusal, nothing more should run.
        if messages[-1].content == MODEL_FAILURE_MESSAGE or planning.was_declined(messages):
            return Command(goto="remember")

        failure = planning.step_failure(steps[index], messages)
        if failure is None:
            if index == len(steps) - 1:
                return Command(goto="remember")
            return Command(goto="execute", update={"current_step": index + 1})

        logger.info("Agent step failed: %s", failure)
        if _out_of_time(state) or state["revisions"] >= MAX_REVISIONS:
            return Command(goto="remember", update={"unfinished": [steps[index]["task"]]})
        done = [step["task"] for step in steps[:index]]
        remaining = planning.make_plan(
            planner, messages, tools_by_name, state["recalled"], failure, done
        )
        logger.info("Agent revised plan: %s", [step["task"] for step in remaining])
        return Command(
            goto="execute",
            update={"plan": steps[:index] + remaining, "revisions": state["revisions"] + 1},
        )

    def finish_plan(state: AgentState) -> dict:
        result = GenerateWorkoutPlanResult.model_validate_json(
            _finished_plan(state).content
        )
        answer = format_plan(result)
        slots = []
        # Offer free times right away for a new plan, when a calendar is connected.
        if result.status == "completed" and context.calendar:
            slots = slot_messages(state)
            answer += "\n\n" + format_slots(slots[-1].content)
        return {"messages": [*slots, AIMessage(answer)]}

    def stop(state: AgentState) -> dict:
        last = state["messages"][-1]
        pending = last.tool_calls if isinstance(last, AIMessage) else []
        # Every tool call needs an answer, or the next turn would be rejected.
        skipped = [
            ToolMessage(
                "Not run: limit reached.",
                tool_call_id=c["id"],
                name=c["name"],
                status="error",
            )
            for c in pending
        ]
        return {"messages": [*skipped, AIMessage(LIMIT_MESSAGE)]}

    def remember(state: AgentState) -> dict:
        """Save an episode, from what code saw, if this request used a tool."""
        results = [
            message
            for message in planning.latest_request_messages(state["messages"])
            if isinstance(message, ToolMessage)
        ]
        if results:
            declined = [m.name for m in results if m.content == DECLINED_MESSAGE]
            completed = [m.name for m in results if not planning.tool_result_failed(m)]
            memory.save_episode(
                context.student_id,
                memory.describe_episode(
                    _latest_request(state), completed, declined, state["unfinished"]
                ),
            )
        return {}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan, destinations=("offer_slots", "execute"))
    graph.add_node("offer_slots", offer_slots)
    graph.add_node("execute", execute)
    graph.add_node("approval", approval, destinations=("tools", "execute"))
    graph.add_node("tools", run_tools)
    graph.add_node("check", check, destinations=("execute", "remember"))
    graph.add_node("finish_plan", finish_plan)
    graph.add_node("stop", stop)
    graph.add_node("remember", remember)
    graph.add_edge(START, "plan")
    graph.add_edge("offer_slots", "execute")
    graph.add_conditional_edges("execute", route_after_execute)
    graph.add_conditional_edges("tools", route_after_tools)
    graph.add_edge("finish_plan", "remember")
    graph.add_edge("stop", "remember")
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def ask(
    graph: Any,
    text: str,
    *,
    thread_id: str,
    approve: Callable[[dict[str, Any]], bool] = lambda _request: False,
) -> AgentReply:
    """Send one request in a conversation. Risky actions are declined by default."""
    config = {"configurable": {"thread_id": thread_id}}
    earlier = graph.get_state(config).values.get("messages", [])
    started = monotonic()
    approvals = []
    payload: Any = {
        "messages": [HumanMessage(text)],
        "tool_steps": 0,
        "started_at": started,
        "plan": [],
        "current_step": 0,
        "revisions": 0,
        "unfinished": [],
        "recalled": "",
    }
    while True:
        graph.invoke(payload, config)
        snapshot = graph.get_state(config)
        if not snapshot.interrupts:
            break
        request = snapshot.interrupts[0].value
        approved = bool(approve(request))
        approvals.append((request["reason"], approved))
        payload = Command(resume=approved)

    new = snapshot.values["messages"][len(earlier) + 1 :]
    results = [
        (message.name, str(message.content))
        for message in new
        if isinstance(message, ToolMessage)
    ]
    values = snapshot.values
    unfinished = [
        f'The step "{task}" did not complete, so do not rely on anything above that '
        "says it did."
        for task in values["unfinished"]
    ]
    notices = [f"**Note:** {text}" for text in [*tool_notices(results), *unfinished]]
    return AgentReply(
        answer="\n\n".join([str(new[-1].content), *notices]),
        tool_calls=[
            (call["name"], call["args"])
            for message in new
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ],
        tool_results=results,
        plan=[step["task"] for step in values["plan"]],
        revisions=values["revisions"],
        approvals=approvals,
        seconds=monotonic() - started,
    )
