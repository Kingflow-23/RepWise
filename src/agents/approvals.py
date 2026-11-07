"""The single approval gate for consequential actions.

Every tool call from an agent or an MCP client goes through
:func:`run_tool_with_approval`:

1. the arguments are validated against the tool's arguments model;
2. the catalog's approval rule is applied (``never``, ``always`` or ``conditional``);
3. a call that needs no approval runs now; a call that needs approval is **not
   run**. It is stored as a *pending action* holding the exact tool name, the
   exact validated arguments (with their SHA-256 hash), the reason, the agent
   that proposed it, the account it belongs to and an expiry time.

Only :func:`approve` runs a pending action, and only for the account that owns
it, while it is still pending and not expired. The status change from
``pending`` to ``approved`` is a single conditional database update, so an
approval cannot be replayed or applied twice. A generic "yes" from the model
is never an approval: the student must approve a specific action ID through
the API (``POST /api/v1/students/{id}/agent/actions/{action_id}/approve``).

Action statuses: ``pending`` -> ``approved`` -> ``executed`` or ``failed``;
or ``pending`` -> ``declined``; or ``pending`` -> ``expired``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

from pydantic import ValidationError

from src.schemas.models import Model
from src.tools.context import ToolContext
from src.tools.planning import GenerateWorkoutPlanResult
from src.tools.registry import (
    approval_reason,
    execute_tool,
    prepare_arguments,
    validate_arguments,
)
from src.tools.results import ToolFailure, failure

logger = logging.getLogger(__name__)

ACTIVATION_REASON = (
    "The new plan was saved as a draft. Approve to make it your active plan; "
    "your current plan will be archived, not deleted."
)


class PendingAction(Model):
    """A consequential action and what happened to it."""

    id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    proposed_by: str
    status: Literal["pending", "approved", "executed", "failed", "declined", "expired"]
    created_at: str
    expires_at: str
    decided_at: str | None = None
    result: dict[str, Any] | None = None


class PendingActionResult(Model):
    """What a tool call returns when it was held for approval instead of run."""

    ok: Literal[True] = True
    status: Literal["pending_approval"] = "pending_approval"
    action_id: str
    tool_name: str
    reason: str
    expires_at: str
    message: str = (
        "Nothing has changed yet. The student must approve this action by its ID."
    )


class ApprovalError(Exception):
    """An approval or decline was refused.

    Attributes:
        code: ``not_found``, ``already_decided``, ``expired`` or ``tampered``.
        message: Explanation for the student.
    """

    def __init__(self, code: str, message: str):
        """Store the reason.

        Args:
            code: Stable reason code.
            message: Human-readable explanation.
        """
        super().__init__(message)
        self.code = code
        self.message = message


class ActionDecision(Model):
    """The outcome of approving or declining one action."""

    action: PendingAction
    executed: bool


@dataclass
class ToolRun:
    """What one gated tool call produced.

    Attributes:
        name: Tool name.
        result: The tool's result, a :class:`PendingActionResult`, or a failure.
        pending: Every action this call left waiting for approval (a held call,
            or the activation of a plan saved as a draft).
    """

    name: str
    result: Model
    pending: list[PendingActionResult] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """Whether the call ran (or was held) without a failure."""
        return not isinstance(self.result, ToolFailure)

    def as_json(self) -> str:
        """Serialize the result for the language model.

        Returns:
            The result as JSON, plus a ``pending_actions`` list when the call
            left actions waiting for approval.
        """
        data = self.result.model_dump(mode="json")
        if self.pending and not isinstance(self.result, PendingActionResult):
            data["pending_actions"] = [
                item.model_dump(mode="json") for item in self.pending
            ]
        return json.dumps(data, ensure_ascii=False)


def arguments_hash(tool_name: str, arguments: dict[str, Any]) -> str:
    """Hash a tool call, so the stored action can be checked before it runs.

    Args:
        tool_name: Tool name.
        arguments: JSON-compatible arguments.

    Returns:
        Hex SHA-256 of the canonical JSON of ``[tool_name, arguments]``.
    """
    canonical = json.dumps([tool_name, arguments], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def propose(
    context: ToolContext,
    tool_name: str,
    arguments: dict[str, Any],
    reason: str,
    proposed_by: str,
) -> PendingActionResult:
    """Store an action for the student to approve. Nothing runs.

    Args:
        context: Trusted tool context (the owner is ``context.student_id``).
        tool_name: Tool to run once approved.
        arguments: Validated arguments, in JSON form.
        reason: Why approval is needed.
        proposed_by: Agent (or ``mcp``) proposing the action.

    Returns:
        The held-call result to give back to the model.
    """
    expires_at = (
        context.clock() + timedelta(minutes=context.config.approval_minutes)
    ).isoformat()
    record = context.repository.create_pending_action(
        context.student_id,
        tool_name,
        arguments,
        arguments_hash(tool_name, arguments),
        reason,
        proposed_by,
        expires_at,
    )
    logger.info("Action proposed tool=%s by=%s", tool_name, proposed_by)
    return PendingActionResult(
        action_id=record["id"],
        tool_name=tool_name,
        reason=reason,
        expires_at=expires_at,
    )


def run_tool_with_approval(
    context: ToolContext, tool_name: str, raw_arguments: dict | None, proposed_by: str
) -> ToolRun:
    """Validate a tool call, then run it or hold it for approval.

    Args:
        context: Trusted tool context.
        tool_name: Tool the agent asked for.
        raw_arguments: Arguments as the model sent them.
        proposed_by: Name of the calling agent (for the audit trail).

    Returns:
        The run, including any actions left waiting for approval.
    """
    try:
        arguments = validate_arguments(tool_name, raw_arguments)
    except KeyError as error:
        return ToolRun(tool_name, failure("unknown_tool", str(error)))
    except ValidationError as error:
        return ToolRun(tool_name, failure("invalid_arguments", str(error)))
    # Shorthand arguments (such as a catch-up week) become the exact change
    # first, so the student approves, and the action stores, what will run.
    prepared = prepare_arguments(context, tool_name, arguments)
    if isinstance(prepared, ToolFailure):
        return ToolRun(tool_name, prepared)
    arguments = prepared

    reason = approval_reason(context, tool_name, arguments)
    if reason:
        held = propose(
            context, tool_name, arguments.model_dump(mode="json"), reason, proposed_by
        )
        return ToolRun(tool_name, held, [held])

    result = execute_tool(context, tool_name, arguments)
    run = ToolRun(tool_name, result)
    # A plan generated while another plan is active is only a draft: replacing
    # the plan the student follows is a consequential change, so its activation
    # is proposed here, in code, rather than left to the model to remember.
    if (
        isinstance(result, GenerateWorkoutPlanResult)
        and result.status == "completed"
        and result.plan_status == "draft"
    ):
        run.pending.append(
            propose(
                context,
                "activate_plan",
                {"plan_id": result.plan_id},
                ACTIVATION_REASON,
                proposed_by,
            )
        )
    return run


def _to_action(record: dict[str, Any]) -> PendingAction:
    """Turn a repository record into the public action model.

    Args:
        record: A row from ``pending_actions``.

    Returns:
        The action, without the owner ID and the internal hash.
    """
    hidden = {"student_id", "arguments_hash"}
    return PendingAction.model_validate(
        {key: value for key, value in record.items() if key not in hidden}
    )


def _load_owned(context: ToolContext, action_id: str) -> dict[str, Any]:
    """Load an action record of the authenticated student, expiring stale ones.

    Args:
        context: Trusted tool context.
        action_id: Action ID.

    Returns:
        The repository record.

    Raises:
        ApprovalError: ``not_found`` for an unknown or another student's action.
    """
    context.repository.expire_pending_actions(
        context.student_id, context.clock().isoformat()
    )
    record = context.repository.get_pending_action(context.student_id, action_id)
    if record is None:
        raise ApprovalError("not_found", "No such action for this account.")
    return record


def list_actions(
    context: ToolContext, status: str | None = "pending"
) -> list[PendingAction]:
    """List the authenticated student's actions, newest first.

    Args:
        context: Trusted tool context.
        status: Only this status, or ``None`` for every action.

    Returns:
        The actions; stale pending actions are reported as ``expired``.
    """
    context.repository.expire_pending_actions(
        context.student_id, context.clock().isoformat()
    )
    records = context.repository.list_pending_actions(context.student_id, status)
    return [_to_action(record) for record in records]


def approve(context: ToolContext, action_id: str) -> ActionDecision:
    """Run one pending action that the authenticated student approved.

    Args:
        context: Trusted tool context of the approving student.
        action_id: The action to run.

    Returns:
        The action after it ran, with its result (``executed`` or ``failed``).

    Raises:
        ApprovalError: ``not_found``, ``already_decided``, ``expired`` or
            ``tampered`` (the stored arguments no longer match their hash).
    """
    record = _load_owned(context, action_id)
    action = _to_action(record)
    _require_pending(action)
    if record["arguments_hash"] != arguments_hash(action.tool_name, action.arguments):
        raise ApprovalError(
            "tampered", "The stored action changed after it was proposed."
        )
    now = context.clock().isoformat()
    if not context.repository.decide_pending_action(
        context.student_id, action_id, "approved", now
    ):
        raise ApprovalError(
            "already_decided", "This action was already decided or expired."
        )

    result = execute_tool(context, action.tool_name, action.arguments)
    status = "failed" if isinstance(result, ToolFailure) else "executed"
    context.repository.finish_pending_action(
        context.student_id, action_id, status, result.model_dump(mode="json")
    )
    logger.info("Action approved tool=%s status=%s", action.tool_name, status)
    return ActionDecision(
        action=_to_action(_load_owned(context, action_id)),
        executed=status == "executed",
    )


def decline(context: ToolContext, action_id: str) -> ActionDecision:
    """Refuse one pending action. Nothing runs.

    Args:
        context: Trusted tool context of the student.
        action_id: The action to refuse.

    Returns:
        The declined action.

    Raises:
        ApprovalError: ``not_found``, ``already_decided`` or ``expired``.
    """
    action = _to_action(_load_owned(context, action_id))
    _require_pending(action)
    now = context.clock().isoformat()
    if not context.repository.decide_pending_action(
        context.student_id, action_id, "declined", now
    ):
        raise ApprovalError(
            "already_decided", "This action was already decided or expired."
        )
    logger.info("Action declined tool=%s", action.tool_name)
    return ActionDecision(
        action=_to_action(_load_owned(context, action_id)), executed=False
    )


def _require_pending(action: PendingAction) -> None:
    """Refuse to decide an action that is no longer pending.

    Args:
        action: The loaded action.

    Raises:
        ApprovalError: ``expired`` or ``already_decided``.
    """
    if action.status == "expired":
        raise ApprovalError("expired", "This action expired. Ask RepWise again.")
    if action.status != "pending":
        raise ApprovalError(
            "already_decided", f"This action is already {action.status}."
        )
