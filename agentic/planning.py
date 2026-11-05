"""The Plan-and-Execute strategy's planning and checking, without the graph.

The model breaks a request into steps, and code checks each step's outcome from the
tool results. ``agent.py`` runs the steps. Each function here can be tested alone.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from ollama import ResponseError
from pydantic import Field

from src.schemas.models import Model

logger = logging.getLogger(__name__)

MAX_PLAN_STEPS = 5
# The latest messages the planner sees, enough for a follow-up like "the first slot".
PLANNER_HISTORY = 4
DECLINED_MESSAGE = "The student declined this action. It did not run."

PLANNER_PROMPT = """You plan the work of RepWise's workout assistant. Do not do the work.
Break the student's latest request into the fewest steps that complete it, in order.
Most requests need one step. Add a step only when the request asks for several actions, or when a tool needs another tool's result, such as a plan_id from get_timetable.
Each step has a short task, and a tool: the one tool that step must call. Use null if the step only needs a reply, such as answering from the background below.
Only use these tools:
{tools}"""

REVISION_PROMPT = """
The plan failed at this step: {failure}
Steps already done: {done}
Plan only the remaining work, including a different way to do the failed step if there is one."""


class PlanStep(Model):
    """One step of a plan."""

    task: str = Field(min_length=1, max_length=200)
    tool: str | None = None


class Plan(Model):
    """The steps the model proposes for one request."""

    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_PLAN_STEPS)


def one_step(task: str, tool: str | None = None) -> dict[str, Any]:
    """Build a plan step, in the plain form the graph's state keeps."""
    return {"task": task, "tool": tool}


def make_plan(
    planner: Any,
    messages: list,
    tools: dict[str, Any],
    memory: str = "",
    failure: str | None = None,
    done: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Ask the model for steps. If it can't give any, plan one step for the whole request.

    ``planner`` is the model with ``Plan`` as its structured output. Pass ``failure``
    to revise a plan: the new steps replace the ones not done yet.
    """
    tool_lines = "\n".join(
        f"- {name}: {tool.description}" for name, tool in tools.items()
    )
    prompt = PLANNER_PROMPT.format(tools=tool_lines)
    if memory:
        prompt += f"\n{memory}"
    if failure:
        prompt += REVISION_PROMPT.format(
            failure=failure, done=", ".join(done or []) or "none"
        )
    conversation = [
        message
        for message in messages
        if isinstance(message, HumanMessage)
        or (isinstance(message, AIMessage) and message.content)
    ]
    try:
        plan = planner.invoke(
            [SystemMessage(prompt), *conversation[-PLANNER_HISTORY:]]
        )
        steps = plan.steps
    # A plan is only a guide: without one, the executor handles the whole request.
    except (AttributeError, ConnectionError, httpx.HTTPError, ResponseError, ValueError):
        logger.warning("The model gave no usable plan", exc_info=True)
        return [one_step("Complete the student's request.")]
    return [
        # A made-up tool name could never pass the check, so it counts as no tool.
        one_step(step.task, step.tool if step.tool in tools else None)
        for step in steps
    ]


def latest_request_messages(messages: list) -> list:
    """Return the messages since the student's latest request, including it."""
    last = max(i for i, message in enumerate(messages) if isinstance(message, HumanMessage))
    return messages[last:]


def tool_result_failed(message: ToolMessage) -> bool:
    """Say whether a tool reported a failure, as an error or as ``"ok": false``."""
    if message.status == "error":
        return True
    content = str(message.content)
    return content.startswith("{") and json.loads(content).get("ok") is False


def was_declined(messages: list) -> bool:
    """Say whether the student declined an action during the latest request."""
    return any(
        isinstance(message, ToolMessage) and message.content == DECLINED_MESSAGE
        for message in latest_request_messages(messages)
    )


def step_failure(step: dict[str, Any], messages: list) -> str | None:
    """Check a step against the latest request's tool results. None means it passed.

    This is what catches an answer that claims work no tool did: a step that names a
    tool passes only when that tool really ran and succeeded.
    """
    tool = step["tool"]
    if tool is None:
        return None
    results = [
        message
        for message in latest_request_messages(messages)
        if isinstance(message, ToolMessage) and message.name == tool
    ]
    if not results:
        return f'"{step["task"]}": {tool} was not called.'
    if tool_result_failed(results[-1]):
        return f'"{step["task"]}": {tool} failed: {str(results[-1].content)[:300]}'
    return None


def messages_for_executor(messages: list) -> list:
    """Hide text replies from earlier steps of the latest request.

    Otherwise the conversation ends on the model's own reply, so it has nothing to
    respond to and returns nothing. It also stops rereading a claim the check rejected.
    Tool results stay, because they are the work done so far.
    """
    current = latest_request_messages(messages)
    earlier = messages[: len(messages) - len(current)]
    return earlier + [
        message
        for message in current
        if not (isinstance(message, AIMessage) and not message.tool_calls)
    ]


def step_instructions(plan: list[dict[str, Any]], index: int) -> str:
    """Tell the executor the whole plan and the step it is on."""
    steps = "\n".join(
        f"{number}. {step['task']}" for number, step in enumerate(plan, start=1)
    )
    step = plan[index]
    lines = [f"The plan for this request:\n{steps}", f"You are on step {index + 1}."]
    if step["tool"]:
        lines.append(
            f"This step must call {step['tool']}, after any tool it needs a value "
            "from, unless a tool result above already covers it."
        )
    if index == len(plan) - 1:
        lines.append(
            "This is the last step. Once it is done, write the student's complete "
            "answer to their whole request."
        )
    else:
        lines.append("Do only this step, then reply with one short line.")
    return "\n".join(lines)
