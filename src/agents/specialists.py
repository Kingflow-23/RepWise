"""The five specialist agents and the bounded tool loop they all share.

Each specialist is a small LangGraph graph::

    think --(tool calls)--> act --> think --> ... --> done
      |                                         ^
      +--(no tool calls: final answer)----------+
      +--(too many calls or out of time)--> limit

- ``think`` asks the language model, which sees only this specialist's tools
  (those whose ``agents`` list in ``config/tools.json`` contains its name);
- ``act`` runs the requested tools through the approval gate
  (:func:`src.agents.approvals.run_tool_with_approval`), so a write that needs
  approval is stored as a pending action instead of running;
- ``limit`` stops after ``max_tool_calls`` tool calls or when the time budget
  is spent, and says so.

A specialist never sees the student's ID or any credential. It receives the
task from the orchestrator and the shared context (profile, plan, earlier
specialists' results) as text, and returns a :class:`SpecialistReport`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from time import monotonic, perf_counter
from typing import Any, Literal, TypedDict

import httpx
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from src.agents.approvals import PendingActionResult, run_tool_with_approval
from src.agents.metrics import ModelCall, record_call
from src.schemas.models import Model
from src.tools.catalog import AgentName, tools_for_agent
from src.tools.context import ToolContext
from src.tools.registry import implementation

logger = logging.getLogger(__name__)

SpecialistName = Literal["planning", "performance", "schedule", "location", "knowledge"]
MODEL_ERRORS = (ConnectionError, httpx.HTTPError, OSError, TimeoutError)
# Failure codes that mean "RepWise must not do this", so the step is complete.
SAFETY_REFUSALS = {"health_question", "health_concern", "health_not_stored"}

COMMON_RULES = """Rules for every RepWise agent:
- Answer in the language of the student's request.
- Use your tools; never guess saved data. Only name plans, workouts, places or sources that a tool returned.
- Never say something was saved, booked, moved or changed unless a tool result in this task confirms it. A result with status "pending_approval" means nothing changed yet: say it waits for the student's approval.
- Never compute dates or weekdays yourself; use the ones in the context or tool results.
- RepWise gives general fitness guidance, never medical advice. If the student mentions pain, injury or a medical issue, do not increase training load and suggest a health professional.
- Finish with a short report of what you did and found, for the orchestrator."""


@dataclass(frozen=True)
class Specialist:
    """One specialist agent.

    Attributes:
        name: Agent name used in the catalog and in plans.
        title: Display name.
        purpose: One line the orchestrator reads to decide when to delegate.
        instructions: The specialist's own rules, added to ``COMMON_RULES``.
    """

    name: SpecialistName
    title: str
    purpose: str
    instructions: str


SPECIALISTS: dict[SpecialistName, Specialist] = {
    "planning": Specialist(
        name="planning",
        title="Workout Planning Agent",
        purpose="Creates a new personalised weekly workout programme (sets, reps, rest) from the profile, goals and history.",
        instructions="""You are the Workout Planning Agent.
- To create a programme, call generate_workout_plan exactly once, with a focus that restates the student's goal for this programme. Never write a programme yourself.
- Put only constraints the student stated (in this request, in the remembered facts, or from a previous agent such as the equipment of a chosen gym) in extra_constraints.
- Call get_workout_history first only when the request is about past training or progression.
- If the result has plan_status "draft", say the new plan is saved and waits for the student's approval to replace their current plan.
- Use save_recommendation only to keep a suggested change for later; it changes nothing.
- Report the plan title, each workout's day and focus, and how many sources support it.""",
    ),
    "performance": Specialist(
        name="performance",
        title="Performance & Adaptation Agent",
        purpose="Records completed, partial or missed workouts and feedback, analyses progress and adjusts the next workout.",
        instructions="""You are the Performance & Adaptation Agent.
- Record how a workout went with save_workout_feedback only when the student reports it. A missed workout is completion_status "not_completed" with difficulty "just_right" unless the student said otherwise. Use week_number and workout_index from the context; record each reported workout once.
- Record weights and reps with log_workout_performance, all exercises of a workout in one call.
- Call update_upcoming_workout only when the student asks to adapt the next workout because of how a completed workout felt, after its feedback is saved. It needs the student's approval. Never call it for missed workouts: rescheduling them is the Schedule Agent's job.
- Only mention workouts that are in the timetable of the context.
- For progress questions call get_workout_history and use its summary counts instead of counting yourself.
- A health_concern on feedback means: no automatic change; recommend a health professional.
- Use save_recommendation for a progression idea the student should review later.""",
    ),
    "schedule": Specialist(
        name="schedule",
        title="Schedule & Calendar Agent",
        purpose="Finds free training times in the connected calendar, reorganises workout days in the plan, and books approved sessions.",
        instructions="""You are the Schedule & Calendar Agent.
- To organise the week or find times, call propose_training_slots and give each workout's options by copying each slot's label exactly (it already has the right weekday and date). Nothing is booked yet.
- If it fails with calendar_not_connected, say no calendar is connected, and propose new days for the workouts using only the student's available days from the context.
- After missed workouts (they are logged as not_completed in the timetable), call organize_schedule with catch_up_week set to their week_number and no moves; when the student talks about the rest of the current week, also pass from_day = today's weekday from the context. Code puts the missed workouts first on the available days left. Never write those moves yourself. It needs approval.
- To move named workouts (for example "move Thursday to Saturday"), call organize_schedule once with every move. It needs approval.
- Only when the student asked to book, call add_workouts_to_calendar with exact start times returned by propose_training_slots. It needs approval.
- Use get_calendar_busy only to answer whether the student is busy or free.""",
    ),
    "location": Specialist(
        name="location",
        title="Location Agent",
        purpose="Finds nearby gyms, sports or fitness equipment shops, and supplement shops around the student or a named place.",
        instructions="""You are the Location Agent.
- Call search_nearby_places with category "gym", "sports_store" (any fitness equipment) or "supplement_store". Pass place only when the student named one.
- Results are nearest first: never reorder them. Give each place's name, distance and address. A result with name_suggests_fitness_equipment is only a hint from its name.
- Never say a shop sells a product, has it in stock, or is open: the map data cannot confirm it.
- Supplements: only legal, non-prescription sports nutrition. Refuse anabolic steroids, SARMs, prescription drugs or anything medical, without searching.
- Searches only work in metropolitan France. If the location is unknown, too vague (place_too_vague) or outside France (outside_service_area), do not search elsewhere: ask the student for a town, a postcode with the town, or an address in France, or for permission to use their location.
- Only search for gyms and fitness-related shops, never unrelated places.""",
    ),
    "knowledge": Specialist(
        name="knowledge",
        title="Fitness Knowledge Agent",
        purpose="Answers training questions with evidence retrieved from the curated fitness documents, and supplies evidence to other agents.",
        instructions="""You are the Fitness Knowledge Agent.
- Call search_documents (at most two searches, in English or French) and base your answer only on the returned passages.
- Cite each claim with the passage's title. Put your own reasoning in a separate line starting with "Interpretation:".
- If evidence_found is false or the passages do not cover the question, say the documents do not cover it. Never invent sources or studies.
- End with a line "Sources:" listing the titles and links you used.""",
    ),
}


class SpecialistReport(Model):
    """What a specialist did for one task, for the orchestrator.

    Attributes:
        agent: Specialist name.
        task: The task it was given.
        status: ``completed``, ``failed`` (no tool succeeded or the model
            failed), or ``limit_reached``.
        summary: The specialist's final text.
        tools_used: Each call, with whether it succeeded and its failure code.
        results: The last result of each tool, as JSON data, so later
            specialists and the final answer can rely on facts, not prose.
        all_results: Every tool result in call order, as ``{"name", "data"}``
            (a tool called twice keeps both results here).
        pending_actions: Actions waiting for the student's approval.
        error: What went wrong, for a failed step.
        seconds: Wall-clock duration of the task.
        model_calls: Each model call of the task, with time and tokens.
    """

    agent: SpecialistName
    task: str
    status: Literal["completed", "failed", "limit_reached"]
    summary: str = ""
    tools_used: list[dict[str, Any]] = Field(default_factory=list)
    results: dict[str, Any] = Field(default_factory=dict)
    all_results: list[dict[str, Any]] = Field(default_factory=list)
    pending_actions: list[PendingActionResult] = Field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0
    model_calls: list[ModelCall] = Field(default_factory=list)


class SpecialistState(TypedDict):
    """Working memory of one specialist task."""

    messages: list[BaseMessage]
    tool_calls_used: int
    report: dict[str, Any]


def tool_schemas(agent: AgentName) -> list[dict[str, Any]]:
    """Build the tool definitions the model sees, from the catalog.

    Args:
        agent: Agent whose tools are listed.

    Returns:
        OpenAI-style function definitions (the format ``bind_tools`` accepts),
        with the catalog's name and description.
    """
    schemas = []
    for spec in tools_for_agent(agent):
        schema = convert_to_openai_tool(implementation(spec.name).arguments)
        schema["function"]["name"] = spec.name
        schema["function"]["description"] = spec.description
        schemas.append(schema)
    return schemas


def run_specialist(
    specialist: Specialist,
    task: str,
    shared_context: str,
    context: ToolContext,
    llm: Any,
    deadline: float,
) -> SpecialistReport:
    """Run one specialist on one task, within its step and time limits.

    Args:
        specialist: Which specialist to run.
        task: What the orchestrator asks it to do.
        shared_context: Profile, plan, date and earlier results, as text.
        context: Trusted tool context for the authenticated student.
        llm: A LangChain chat model with ``bind_tools`` (``ChatOllama`` in
            production, a scripted fake in tests).
        deadline: ``time.monotonic()`` value after which the task must stop.

    Returns:
        The specialist's report. It never raises for model or tool failures.
    """
    limits = context.config.agent_limits
    allowed = {spec.name for spec in tools_for_agent(specialist.name)}
    model = llm.bind_tools(tool_schemas(specialist.name))
    report: dict[str, Any] = {
        "agent": specialist.name,
        "task": task,
        "tools_used": [],
        "results": {},
        "all_results": [],
        "pending_actions": [],
        "model_calls": [],
    }
    task_started = perf_counter()

    def think(state: SpecialistState) -> dict:
        """Ask the model for the next tool calls or the final answer."""
        started = perf_counter()
        try:
            reply = model.invoke(state["messages"])
            report["model_calls"].append(
                record_call(f"{specialist.name}: think", started, reply)
            )
        except MODEL_ERRORS as error:
            logger.warning(
                "Specialist model failed agent=%s error=%s", specialist.name, error
            )
            report["error"] = f"The language model could not be reached: {error}"
            reply = AIMessage("")
        return {"messages": [*state["messages"], reply]}

    def act(state: SpecialistState) -> dict:
        """Run every tool the model asked for, through the approval gate."""
        calls = state["messages"][-1].tool_calls
        new_messages: list[BaseMessage] = []
        for call in calls:
            if call["name"] not in allowed:
                # A specialist only reaches its own tools, even if the model
                # names another agent's tool.
                content = json.dumps(
                    {
                        "ok": False,
                        "code": "not_allowed",
                        "error": f"{call['name']} is not one of your tools.",
                    }
                )
                report["tools_used"].append(
                    {"name": call["name"], "ok": False, "code": "not_allowed", "ms": 0}
                )
            else:
                started = perf_counter()
                run = run_tool_with_approval(
                    context, call["name"], call["args"], specialist.name
                )
                elapsed_ms = round((perf_counter() - started) * 1000)
                content = run.as_json()
                data = json.loads(content)
                report["results"][call["name"]] = data
                report["all_results"].append({"name": call["name"], "data": data})
                report["pending_actions"].extend(run.pending)
                report["tools_used"].append(
                    {
                        "name": call["name"],
                        "ok": run.succeeded,
                        "code": data.get("code"),
                        "ms": elapsed_ms,
                        "pending_approval": data.get("status") == "pending_approval",
                    }
                )
                logger.info(
                    "Specialist tool agent=%s tool=%s ok=%s",
                    specialist.name,
                    call["name"],
                    run.succeeded,
                )
            new_messages.append(
                ToolMessage(content, tool_call_id=call["id"], name=call["name"])
            )
        return {
            "messages": [*state["messages"], *new_messages],
            "tool_calls_used": state["tool_calls_used"] + len(calls),
        }

    def route(state: SpecialistState) -> str:
        """Decide whether to run tools, stop at a limit, or finish."""
        calls = state["messages"][-1].tool_calls
        if not calls:
            return "done"
        if (
            state["tool_calls_used"] + len(calls) > limits.max_tool_calls
            or monotonic() > deadline
        ):
            return "limit"
        return "act"

    def limit(state: SpecialistState) -> dict:
        """Stop without running the requested calls, and say why."""
        report["status"] = "limit_reached"
        report["error"] = (
            f"Stopped after {state['tool_calls_used']} tool calls: the limit is "
            f"{limits.max_tool_calls} calls and {limits.specialist_seconds} seconds per task."
        )
        return {}

    graph = StateGraph(SpecialistState)
    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.add_node("limit", limit)
    graph.add_edge(START, "think")
    graph.add_conditional_edges(
        "think", route, {"act": "act", "limit": "limit", "done": END}
    )
    graph.add_edge("act", "think")
    graph.add_edge("limit", END)

    system = "\n\n".join([COMMON_RULES, specialist.instructions, shared_context])
    final = graph.compile().invoke(
        {
            "messages": [SystemMessage(system), HumanMessage(f"Your task: {task}")],
            "tool_calls_used": 0,
            "report": {},
        },
        # Each think/act pair is two graph steps; leave room for the limit node.
        {"recursion_limit": 2 * limits.max_tool_calls + 6},
    )
    last = final["messages"][-1]
    report["summary"] = str(last.content).strip() if isinstance(last, AIMessage) else ""
    report.setdefault("status", _status(report))
    report["seconds"] = round(perf_counter() - task_started, 3)
    return SpecialistReport.model_validate(report)


def _status(report: dict[str, Any]) -> Literal["completed", "failed"]:
    """Decide whether a finished task succeeded.

    Args:
        report: The report being built.

    Returns:
        ``failed`` when the model could not answer, or when tools were called
        and none of them succeeded; ``completed`` otherwise.
    """
    if report.get("error") or not report.get("summary"):
        return "failed"
    used = report["tools_used"]
    # A safety refusal is the correct outcome, not something to retry.
    handled = [item for item in used if item["ok"] or item["code"] in SAFETY_REFUSALS]
    if used and not handled:
        report["error"] = "Every tool call failed: " + ", ".join(
            f"{item['name']} ({item['code']})" for item in used
        )
        return "failed"
    return "completed"
