"""The Orchestrator Agent: Plan-and-Execute across the five specialists.

One request runs through this LangGraph graph::

    load_context --> plan --> delegate --> check --+--> respond --> remember
                      ^                            |
                      +------- replan (once) ------+  (a step failed)
                               next step ----------> delegate

1. ``load_context`` - the orchestrator's own read tools (``get_user_profile``,
   ``get_timetable``, ``get_current_plan``) run once, in code, together with
   memory recall. The result is the shared context every specialist receives,
   so no specialist re-reads it.
2. ``plan`` - the model writes a short plan: which specialists, in which
   order, with which task, and which workout (if any) a step is about. It also
   reports facts the student asked to remember and whether the request may
   mention a health issue. At most ``max_delegations`` steps.
3. ``delegate`` - runs the current step's specialist (``specialists.py``) with
   the shared context, the referenced workout (``get_workout``) and the
   reports of earlier steps.
4. ``check`` - code validates the step's report. A failed step triggers one
   revision of the remaining plan (``max_replans``); a time-out ends the run.
5. ``respond`` - the model combines the reports into one answer. Code then
   appends what must never be lost: pending approvals, failures, limits.
6. ``remember`` - code writes one episode (episodic memory).

The orchestrator has no write tools of its own except ``remember_fact``; every
training or calendar change is done by a specialist and goes through the
approval gate. Conversation turns are kept per ``conversation_id`` by the
LangGraph checkpointer, so a follow-up such as "the first slot" has context.
"""

from __future__ import annotations

import json
import logging
import operator
from dataclasses import dataclass
from time import monotonic, perf_counter
from typing import Annotated, Any, Literal, TypedDict
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from config.config import AppConfig
from src.agents import memory
from src.agents.approvals import PendingActionResult, run_tool_with_approval
from src.agents.metrics import Recorder, RequestMetrics, record_call
from src.agents.specialists import (
    MODEL_ERRORS,
    SPECIALISTS,
    SpecialistName,
    SpecialistReport,
    run_specialist,
)
from src.schemas.models import Model
from src.tools.context import ToolContext
from src.tools.registry import execute_tool
from src.tools.results import ToolFailure

logger = logging.getLogger(__name__)

HISTORY_TURNS = 4

PLANNER_PROMPT = """You are RepWise's Orchestrator Agent. You plan; specialists do the work.
Break the student's latest request into the fewest steps, in the order they must run.
Each step names one specialist and its task, written as an instruction to it.
Specialists:
{specialists}
Planning rules:
- Make one step for each distinct thing the student asks for, and give each step only that specialist's part of the work. A step that needs another step's result comes after it.
- Requests that combine needs get several steps, for example:
  "find a gym near me and create a programme" -> 1. location: find the nearest gyms; 2. planning: create the programme.
  "I missed two workouts, organise the rest of my week" -> 1. performance: record the missed workouts; 2. schedule: reorganise the remaining workouts of the week.
  "what do the documents say about rest, then build me a plan" -> 1. knowledge: find the evidence; 2. planning: create the plan.
- Never give one specialist another's job: moving or reorganising days is schedule's job; recording how workouts went is performance's job.
- Any question about training knowledge (rest times, exercises, technique, what the documents or research say) needs a knowledge step, even if you think you know the answer. A preference in the profile (such as "prefers short rests") is not evidence and never answers such a question.
- Use zero steps only when the request is about the student's own saved data shown in the context below (for example "what is my goal?"), or is a greeting.
- If a step is about one specific workout from the timetable, set its workout to that week_number and workout_index.
- remember: only facts the student explicitly asked you to remember. Otherwise an empty list.
- health_concern: true if the request mentions or asks about pain, injury, illness, medication or another medical issue.
- Never plan a training-load increase when health_concern is true.
At most {max_steps} steps.
{context}"""

REPLAN_NOTE = """
A step failed: {failure}
Steps already done: {done}
Plan only the remaining work, with a different way to do the failed step if there is one, or zero steps if it cannot be done."""

RESPONDER_PROMPT = """You are RepWise's Orchestrator Agent. Write the final answer to the student.
Combine the specialists' reports below into one coherent answer, in the language of the student's request.
- Answer what the student asked. Do not add a summary of their profile or plan unless they asked for it.
- Only state what the reports and tool results confirm. Do not invent plans, places, times, sources or numbers.
- The "Key facts" lines were computed by code from the saved data and are always right: never contradict them, and use them for any day, weekday or count you mention.
- Available days are when the student could train; only the timetable says which days have workouts.
- Places and free training times are listed by code right after your answer, nearest or earliest first. Do not repeat that list or reorder it; you may name the nearest place or refer to "the list below".
- If an action waits for approval, say what it will do and that nothing changed yet.
- If a step failed, say plainly what could not be done.
- Keep source titles and links from the Fitness Knowledge Agent.
- RepWise gives general fitness guidance, not medical advice.
{context}"""

SPECIALIST_CHOICE_PROMPT = """Choose the one RepWise specialist that should handle the student's request.
{specialists}
- none: the request is only about the student's own profile or plan (goal, days, equipment, current workouts), or is a greeting.
Training-knowledge questions (rest, exercises, technique, what the documents say) go to knowledge.
{context}"""
MISSING_STEPS_NOTE = (
    "\nYou classified this request as needs_specialists but planned no step. "
    "Plan the steps now."
)
NO_REPORTS_NOTE = (
    "No specialist ran and no documents were searched for this request. Answer only "
    "from the student's saved data above. Never cite a document, a source or an "
    "agent's findings; for a training question, say you need to search the "
    "documents and invite the student to ask again."
)
HEALTH_NOTICE = (
    "RepWise gives general fitness guidance, not medical advice. For pain, injury "
    "or another health concern, please talk to a qualified health professional "
    "before changing your training."
)
NO_MODEL_ANSWER = (
    "I could not reach the language model, so I could not plan this request. "
    "Nothing was changed. Please check that Ollama is running and try again."
)


class WorkoutRef(Model):
    """A workout from the timetable that a step is about."""

    week_number: int = Field(ge=1)
    workout_index: int = Field(ge=0)


class Delegation(Model):
    """One step of the orchestrator's plan."""

    agent: SpecialistName
    task: str = Field(min_length=3, max_length=400)
    workout: WorkoutRef | None = None


class SpecialistChoice(Model):
    """The fallback answer: the one specialist that should handle a request.

    ``none`` means the student's saved data, shown in the context, answers it.
    """

    agent: SpecialistName | Literal["none"]


class OrchestratorPlan(Model):
    """The plan the model writes for one request.

    ``request_kind`` comes first on purpose: the model classifies the request
    before writing steps, which makes skipping the specialists a deliberate,
    checkable decision.
    """

    request_kind: Literal["saved_data", "greeting", "needs_specialists"] = Field(
        default="needs_specialists",
        description=(
            "saved_data: only about the student's own profile or plan shown in the "
            "context. greeting: small talk. needs_specialists: anything else, "
            "including every training-knowledge question."
        ),
    )
    steps: list[Delegation] = Field(default_factory=list, max_length=8)
    remember: list[str] = Field(default_factory=list, max_length=5)
    health_concern: bool = False


class StepOutcome(Model):
    """A planned step and how it ended."""

    agent: SpecialistName
    task: str
    status: str


class AgentResponse(Model):
    """Everything the caller needs from one request.

    Attributes:
        answer: The final answer, followed by code-written notices.
        steps: The executed plan, with each step's status.
        reports: Each specialist's full report.
        pending_actions: Actions waiting for the student's approval.
        notices: The code-written notices (also appended to ``answer``).
        replans: How many times the plan was revised.
        health_concern: Whether the request may mention a health issue.
        conversation_id: Pass it back to continue the conversation.
        seconds: Wall-clock time of the request.
        metrics: Time per phase, every model call with its tokens, and tool
            timings, for debugging and evaluation.
    """

    answer: str
    steps: list[StepOutcome] = Field(default_factory=list)
    reports: list[SpecialistReport] = Field(default_factory=list)
    pending_actions: list[PendingActionResult] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    replans: int = 0
    health_concern: bool = False
    conversation_id: str
    seconds: float = 0.0
    metrics: RequestMetrics | None = None


class OrchestratorState(TypedDict):
    """Working memory of one request (``history`` persists across turns)."""

    request: str
    history: Annotated[list[dict[str, str]], operator.add]
    shared_context: str
    steps: list[dict[str, Any]]
    current: int
    reports: list[dict[str, Any]]
    replans: int
    health_concern: bool
    planner_failed: bool
    started_at: float
    answer: str
    notices: list[str]


@dataclass
class Orchestrator:
    """A compiled orchestrator graph for one authenticated student.

    Attributes:
        context: Trusted tool context.
        llm: LangChain chat model with ``bind_tools`` and
            ``with_structured_output`` (``ChatOllama`` in production).
        graph: The compiled LangGraph graph.
        recorder: Collects the measurements of the current request.
    """

    context: ToolContext
    llm: Any
    graph: Any
    recorder: Recorder

    def ask(self, request: str, conversation_id: str) -> AgentResponse:
        """Answer one request.

        Args:
            request: The student's message.
            conversation_id: Groups turns of one conversation.

        Returns:
            The answer, the executed plan, the reports and pending actions.
        """
        started = monotonic()
        self.recorder.reset()
        state = self.graph.invoke(
            {
                "request": request,
                "history": [],
                "shared_context": "",
                "steps": [],
                "current": 0,
                "reports": [],
                "replans": 0,
                "health_concern": False,
                "planner_failed": False,
                "started_at": started,
                "answer": "",
                "notices": [],
            },
            {"configurable": {"thread_id": conversation_id}, "recursion_limit": 60},
        )
        reports = [SpecialistReport.model_validate(item) for item in state["reports"]]
        notices = state["notices"]
        return AgentResponse(
            answer="\n\n".join(
                [state["answer"], *[f"**Note:** {text}" for text in notices]]
            ),
            steps=[
                StepOutcome(
                    agent=step["agent"],
                    task=step["task"],
                    status=step.get("status", "not_run"),
                )
                for step in state["steps"]
            ],
            reports=reports,
            pending_actions=[
                action for report in reports for action in report.pending_actions
            ],
            notices=notices,
            replans=state["replans"],
            health_concern=state["health_concern"],
            conversation_id=conversation_id,
            seconds=round(monotonic() - started, 2),
            metrics=self.recorder.build(
                self.context.config.agent_model, monotonic() - started, reports
            ),
        )


def build_chat_model(config: AppConfig) -> ChatOllama:
    """Create the local tool-calling model used by every agent.

    Args:
        config: Application settings (model name, URL, limits).

    Returns:
        A deterministic ``ChatOllama`` model.
    """
    limits = config.agent_limits
    return ChatOllama(
        model=config.agent_model,
        base_url=config.ollama_url,
        temperature=0,
        reasoning=False,
        # Ollama's default of 4096 tokens is too small for tool schemas plus a timetable.
        num_ctx=limits.context_tokens,
        client_kwargs={"timeout": limits.specialist_seconds},
    )


def _compact(data: Any, limit: int = 6000) -> str:
    """Serialize data for a prompt, cut to a safe length.

    Args:
        data: JSON-compatible data.
        limit: Maximum number of characters.

    Returns:
        Compact JSON text.
    """
    text = json.dumps(data, ensure_ascii=False, default=str, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "...(cut)"


def load_shared_context(context: ToolContext) -> str:
    """Run the orchestrator's read tools once and describe the student's situation.

    Args:
        context: Trusted tool context.

    Returns:
        Text with today's date, the profile, the current plan's timetable and
        recalled memory. Missing data is stated, never invented.
    """
    today = context.clock().astimezone(ZoneInfo(context.timezone))
    parts = [f"Today is {today:%A %Y-%m-%d}, time zone {context.timezone}."]

    profile = execute_tool(context, "get_user_profile", {})
    parts.append(
        "No profile is saved yet."
        if isinstance(profile, ToolFailure)
        else "Student profile: " + _compact(profile.profile.model_dump())
    )
    timetable = execute_tool(context, "get_timetable", {})
    if not isinstance(timetable, ToolFailure):
        parts.append(key_facts(today.strftime("%A"), timetable))
    if isinstance(timetable, ToolFailure) or not timetable.planned_workouts:
        parts.append("There is no active training plan yet.")
    else:
        current = execute_tool(context, "get_current_plan", {})
        if not isinstance(current, ToolFailure):
            parts.append(
                f'Current plan: "{current.plan.title}" (plan_id {current.plan_id}).'
            )
        parts.append(
            "Timetable of the current plan: "
            + _compact(
                [
                    {
                        "week_number": item.week_number,
                        "workout_index": item.workout_index,
                        "day": item.day,
                        "focus": item.focus,
                        "minutes": item.estimated_minutes,
                        "logged": (
                            item.feedback.completion_status if item.feedback else None
                        ),
                        "booked_at": item.booking.start if item.booking else None,
                    }
                    for item in timetable.planned_workouts
                ]
            )
        )
    recalled = memory.recall(context)
    if recalled:
        parts.append(recalled)
    return "\n".join(parts)


def key_facts(today: str, timetable: Any) -> str:
    """State the day facts every answer must respect, computed by code.

    A small model easily mixes up "available days" (when the student could
    train) with "workout days" (what the plan schedules). These lines spell
    both out, per week, so neither the agents nor the final answer have to
    work them out from JSON.

    Args:
        today: Today's weekday, such as ``"Saturday"``.
        timetable: A ``get_timetable`` result.

    Returns:
        A "Key facts" block of short lines.
    """
    available = list(timetable.available_days)
    lines = [
        f"- Today is {today}.",
        f"- Available days (when the student can train, from the profile): "
        f"{', '.join(available)}.",
    ]
    weeks: dict[int, list[Any]] = {}
    for item in timetable.planned_workouts:
        weeks.setdefault(item.week_number, []).append(item)
    if not weeks:
        lines.append("- There is no active plan, so no day has a planned workout.")
    for number, items in sorted(weeks.items()):
        workout_days = [item.day for item in items]
        planned = "; ".join(
            f"{item.day}: {item.focus}"
            + (f" (logged {item.feedback.completion_status})" if item.feedback else "")
            for item in items
        )
        free = [day for day in available if day not in workout_days]
        lines.append(
            f"- Week {number} of the current plan has {len(items)} workouts. "
            f"{planned}."
        )
        lines.append(
            f"- Available days without a workout in week {number}: "
            f"{', '.join(free) if free else 'none'}."
        )
    return "Key facts (computed by code, always right):\n" + "\n".join(lines)


def _history_text(history: list[dict[str, str]]) -> str:
    """Show the last turns of the conversation.

    Args:
        history: ``{"role", "text"}`` turns, oldest first.

    Returns:
        A short transcript, or ``""``.
    """
    turns = history[-2 * HISTORY_TURNS :]
    if not turns:
        return ""
    lines = [f"{turn['role']}: {turn['text'][:600]}" for turn in turns]
    return "Earlier in this conversation:\n" + "\n".join(lines)


def _report_digest(report: dict[str, Any]) -> dict[str, Any]:
    """Keep what later steps and the final answer need from a report.

    Args:
        report: A :class:`SpecialistReport` as a dictionary.

    Returns:
        Agent, task, status, summary, error, tool results and pending actions.
    """
    return {
        "agent": report["agent"],
        "task": report["task"],
        "status": report["status"],
        "summary": report["summary"],
        "error": report.get("error"),
        "results": report["results"],
        "pending_actions": [
            {"tool": item["tool_name"], "reason": item["reason"]}
            for item in report["pending_actions"]
        ],
    }


def slot_notice(report: dict[str, Any]) -> str | None:
    """Restate proposed training times exactly as code computed them.

    Models are unreliable at turning timestamps into weekdays and dates, so
    whenever slots were proposed, the code-written labels reach the student
    unchanged, whatever the answer above says.

    Args:
        report: A specialist report as a dictionary.

    Returns:
        One line per workout with its free times, or ``None`` when the report
        has no successful ``propose_training_slots`` result.
    """
    slots = report["results"].get("propose_training_slots")
    if not slots or not slots.get("ok"):
        return None
    lines = []
    for workout in slots["workouts"]:
        options = (
            ", ".join(slot["label"] for slot in workout["options"])
            or "no free time that week"
        )
        lines.append(
            f"{workout['focus']} (planned {workout['planned_day']}): {options}"
        )
    return "Free times checked in your calendar, nothing booked yet:\n- " + "\n- ".join(
        lines
    )


def places_notice(report: dict[str, Any]) -> str | None:
    """List the places found exactly as code sorted them, nearest first.

    The tool sorts by a distance computed by code and has already dropped
    unrelated places; the model is not trusted to keep that order, so the
    list reaches the student from here, whatever the answer above says.

    Args:
        report: A specialist report as a dictionary.

    Returns:
        One block per successful ``search_nearby_places`` call, or ``None``.
    """
    blocks = []
    for entry in report.get("all_results", []):
        data = entry["data"]
        if entry["name"] != "search_nearby_places" or not data.get("ok"):
            continue
        near = data["near"]["label"]
        if not data["places"]:
            blocks.append(
                f"No matching place was found within {data['radius_km']} km of {near}."
            )
            continue
        lines = [
            f"{place['name']}: {place['distance_km']:.2f} km"
            + (f", {place['address']}" if place.get("address") else "")
            + (f" ({place['map_url']})" if place.get("map_url") else "")
            for place in data["places"]
        ]
        label = data["category"].replace("_", " ")
        blocks.append(
            f"Places found ({label}) near {near}, nearest first:\n- "
            + "\n- ".join(lines)
            + f"\n{data['data_source']}"
        )
    return "\n\n".join(blocks) or None


def build_orchestrator(
    context: ToolContext, llm: Any, checkpointer: Any = None
) -> Orchestrator:
    """Compile the orchestrator graph for one authenticated student.

    Args:
        context: Trusted tool context.
        llm: LangChain chat model (``ChatOllama``, or a scripted fake in tests).
        checkpointer: LangGraph checkpointer keeping conversation history; an
            in-memory one is created when omitted.

    Returns:
        An :class:`Orchestrator` ready to :meth:`~Orchestrator.ask`.
    """
    limits = context.config.agent_limits
    recorder = Recorder()
    # include_raw keeps the model's reply, so its token counts can be recorded.
    planner = llm.with_structured_output(OrchestratorPlan, include_raw=True)
    chooser = llm.with_structured_output(SpecialistChoice, include_raw=True)

    def out_of_time(state: OrchestratorState) -> bool:
        """Whether the request's time budget is spent."""
        return monotonic() - state["started_at"] > limits.request_seconds

    def load_context(state: OrchestratorState) -> dict:
        """Run the orchestrator's read tools once (shared context)."""
        return {"shared_context": load_shared_context(context)}

    def ask_planner(
        state: OrchestratorState, note: str = ""
    ) -> OrchestratorPlan | None:
        """Ask the model for a plan; ``None`` when it cannot give a usable one."""
        specialists = "\n".join(
            f"- {s.name}: {s.purpose}" for s in SPECIALISTS.values()
        )
        prompt = PLANNER_PROMPT.format(
            specialists=specialists,
            max_steps=limits.max_delegations,
            context="\n\n".join(
                part
                for part in (state["shared_context"], _history_text(state["history"]))
                if part
            ),
        )
        try:
            return recorder.structured(
                planner,
                [SystemMessage(prompt + note), HumanMessage(state["request"])],
                "replan" if "failed" in note else "plan",
            )
        except (*MODEL_ERRORS, ValueError) as error:
            logger.warning("Planner gave no usable plan error=%s", error)
            return None

    def single_specialist_step(state: OrchestratorState) -> list[Delegation]:
        """Ask which one specialist should handle the request, as a fallback.

        The answer is constrained to the five specialist names, so even a small
        model gives a usable step. The task is the student's own request.

        Args:
            state: Working memory with the request.

        Returns:
            One step, or no step when the model cannot be reached.
        """
        prompt = SPECIALIST_CHOICE_PROMPT.format(
            specialists="\n".join(
                f"- {s.name}: {s.purpose}" for s in SPECIALISTS.values()
            ),
            context=state["shared_context"],
        )
        try:
            choice = recorder.structured(
                chooser,
                [SystemMessage(prompt), HumanMessage(state["request"])],
                "choose specialist",
            )
        except (*MODEL_ERRORS, ValueError) as error:
            logger.warning("Specialist choice failed error=%s", error)
            return []
        logger.info("Orchestrator fallback specialist=%s", choice.agent)
        if choice.agent == "none":
            return []
        return [Delegation(agent=choice.agent, task=state["request"][:400])]

    def plan(state: OrchestratorState) -> dict:
        """Write the plan, store facts the student asked to remember."""
        proposal = ask_planner(state)
        if (
            proposal
            and proposal.request_kind == "needs_specialists"
            and not proposal.steps
        ):
            # The model said specialists are needed but planned none: ask once more,
            # then fall back to the simplest question it can answer reliably.
            proposal = ask_planner(state, MISSING_STEPS_NOTE) or proposal
            if not proposal.steps:
                proposal = proposal.model_copy(
                    update={"steps": single_specialist_step(state)}
                )
        if proposal is None:
            # Fail closed: without a plan the request is not understood, so its
            # text is treated as possibly health-related for memory purposes.
            return {"planner_failed": True, "health_concern": True, "steps": []}
        notices = []
        for fact in proposal.remember:
            run = run_tool_with_approval(
                context, "remember_fact", {"fact": fact}, "orchestrator"
            )
            if not run.succeeded:
                notices.append(f'"{fact}" was not remembered: {run.result.error}')
        steps = [step.model_dump() for step in proposal.steps[: limits.max_delegations]]
        logger.info("Orchestrator plan: %s", [(s["agent"], s["task"]) for s in steps])
        return {
            "steps": steps,
            "health_concern": proposal.health_concern,
            "notices": notices,
        }

    def route_after_plan(state: OrchestratorState) -> str:
        """Delegate the first step, or answer directly."""
        return "delegate" if state["steps"] else "respond"

    def delegate(state: OrchestratorState) -> dict:
        """Run the current step's specialist with the shared context."""
        step = state["steps"][state["current"]]
        parts = [state["shared_context"]]
        if step.get("workout"):
            workout = execute_tool(context, "get_workout", step["workout"])
            parts.append(
                "The workout this task is about: "
                + (
                    workout.error
                    if isinstance(workout, ToolFailure)
                    else _compact(workout.model_dump(mode="json"))
                )
            )
        if state["reports"]:
            parts.append(
                "Results of earlier steps: "
                + _compact([_report_digest(r) for r in state["reports"]])
            )
        if state["health_concern"]:
            parts.append(
                "The request may mention a health issue: follow the medical-advice rule strictly."
            )
        deadline = min(
            monotonic() + limits.specialist_seconds,
            state["started_at"] + limits.request_seconds,
        )
        report = run_specialist(
            SPECIALISTS[step["agent"]],
            step["task"],
            "Context:\n" + "\n".join(parts),
            context,
            llm,
            deadline,
        )
        logger.info("Specialist done agent=%s status=%s", report.agent, report.status)
        steps = [dict(item) for item in state["steps"]]
        steps[state["current"]]["status"] = report.status
        return {
            "reports": [*state["reports"], report.model_dump(mode="json")],
            "steps": steps,
        }

    def check(state: OrchestratorState) -> dict:
        """Validate the step, then move on, revise the plan once, or stop."""
        report = state["reports"][-1]
        index = state["current"]
        if report["status"] == "completed":
            return {"current": index + 1}
        if out_of_time(state) or state["replans"] >= limits.max_replans:
            return {"current": len(state["steps"])}
        failure = f'{report["agent"]}: {report.get("error") or report["summary"][:300]}'
        done = ", ".join(
            f'{s["agent"]}: {s["task"]}' for s in state["steps"][: index + 1]
        )
        revised = ask_planner(state, REPLAN_NOTE.format(failure=failure, done=done))
        if revised is None:
            return {"current": len(state["steps"])}
        remaining_budget = max(0, limits.max_delegations - (index + 1))
        new_steps = [step.model_dump() for step in revised.steps[:remaining_budget]]
        logger.info(
            "Orchestrator replanned: %s", [(s["agent"], s["task"]) for s in new_steps]
        )
        return {
            "steps": state["steps"][: index + 1] + new_steps,
            "current": index + 1,
            "replans": state["replans"] + 1,
        }

    def route_after_check(state: OrchestratorState) -> str:
        """Run the next step, or write the answer."""
        if state["current"] < len(state["steps"]) and not out_of_time(state):
            return "delegate"
        return "respond"

    def respond(state: OrchestratorState) -> dict:
        """Combine the reports into one answer and add the code-written notices."""
        notices = list(state["notices"])
        if state["planner_failed"]:
            answer = NO_MODEL_ANSWER
        else:
            context_parts = [state["shared_context"], _history_text(state["history"])]
            if state["reports"]:
                context_parts.append(
                    "Specialist reports: "
                    + _compact([_report_digest(r) for r in state["reports"]], 12000)
                )
            else:
                context_parts.append(NO_REPORTS_NOTE)
            started = perf_counter()
            try:
                reply = llm.invoke(
                    [
                        SystemMessage(
                            RESPONDER_PROMPT.format(
                                context="\n\n".join(p for p in context_parts if p)
                            )
                        ),
                        HumanMessage(state["request"]),
                    ]
                )
                recorder.model_calls.append(record_call("respond", started, reply))
                answer = str(reply.content).strip()
            except MODEL_ERRORS as error:
                logger.warning("Responder model failed error=%s", error)
                answer = ""
            if not answer:
                # Without the model, still give the specialists' own words.
                answer = (
                    "\n\n".join(r["summary"] for r in state["reports"] if r["summary"])
                    or NO_MODEL_ANSWER
                )

        for report in state["reports"]:
            for notice in (places_notice(report), slot_notice(report)):
                if notice:
                    notices.append(notice)
            for action in report["pending_actions"]:
                notices.append(
                    f"Waiting for your approval: {action['reason']} "
                    f"(action {action['action_id']}, expires {action['expires_at']})."
                )
            if report["status"] != "completed":
                notices.append(
                    f"The {SPECIALISTS[report['agent']].title} could not finish: "
                    f"{report.get('error') or 'no result'}. Do not rely on anything above that says it did."
                )
        skipped = [s for s in state["steps"] if "status" not in s]
        if skipped:
            notices.append(
                "Not done, because of the time or step limit or an earlier failure: "
                + "; ".join(s["task"] for s in skipped)
            )
        if state["health_concern"] and not state["planner_failed"]:
            notices.append(HEALTH_NOTICE)
        return {
            "answer": answer,
            "notices": notices,
            "history": [
                {"role": "student", "text": state["request"]},
                {"role": "assistant", "text": answer},
            ],
        }

    def remember(state: OrchestratorState) -> dict:
        """Write one episode from what code observed (episodic memory)."""
        if not state["reports"]:
            return {}
        completed, waiting, unfinished = [], [], []
        for report in state["reports"]:
            label = SPECIALISTS[report["agent"]].title
            (completed if report["status"] == "completed" else unfinished).append(label)
            waiting += [action["tool_name"] for action in report["pending_actions"]]
        memory.save_episode(
            context,
            memory.describe_episode(
                state["request"],
                completed,
                waiting,
                unfinished,
                state["health_concern"],
            ),
        )
        return {}

    graph = StateGraph(OrchestratorState)

    def step_name(state: OrchestratorState) -> str:
        """Name the delegate phase after the specialist it runs."""
        return f"delegate: {state['steps'][state['current']]['agent']}"

    # Each node is timed, so the response shows where the time went.
    graph.add_node("load_context", recorder.timed("load_context", load_context))
    graph.add_node("plan", recorder.timed("plan", plan))
    graph.add_node("delegate", recorder.timed(step_name, delegate))
    graph.add_node("check", recorder.timed("check", check))
    graph.add_node("respond", recorder.timed("respond", respond))
    graph.add_node("remember", recorder.timed("remember", remember))
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "plan")
    graph.add_conditional_edges(
        "plan", route_after_plan, {"delegate": "delegate", "respond": "respond"}
    )
    graph.add_edge("delegate", "check")
    graph.add_conditional_edges(
        "check", route_after_check, {"delegate": "delegate", "respond": "respond"}
    )
    graph.add_edge("respond", "remember")
    graph.add_edge("remember", END)
    compiled = graph.compile(checkpointer=checkpointer or InMemorySaver())
    return Orchestrator(context=context, llm=llm, graph=compiled, recorder=recorder)
