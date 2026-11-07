"""HTTP routes for the multi-agent assistant, approvals and the saved location.

All routes require the bearer token and only act on the authenticated account:
the account ID in the path must match the token, and every tool runs with a
:class:`~src.tools.context.ToolContext` built here, never from model output.

How a client (the future frontend) uses the approval flow:

1. ``POST .../agent/requests`` with the student's message. The answer may list
   ``pending_actions``: consequential changes the agents proposed but did not run.
2. Show each pending action's ``reason`` and ask the student.
3. ``POST .../agent/actions/{action_id}/approve`` runs exactly that stored
   action (same tool, same arguments) and returns its result;
   ``.../decline`` refuses it. An action can be decided once, before it expires.
4. ``GET .../agent/actions`` lists pending actions, e.g. after a page reload.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import Field, field_validator, model_validator

from src.agents import approvals
from src.agents.orchestrator import AgentResponse, build_orchestrator
from src.schemas.models import Model
from src.tools.context import Location, ToolContext
from src.tools.places import in_service_area, outside_area_failure, resolve_place
from src.tools.results import ToolFailure

logger = logging.getLogger(__name__)

APPROVAL_ERROR_STATUS = {
    "not_found": 404,
    "already_decided": 409,
    "expired": 410,
    "tampered": 409,
}
# HTTP status for each reason a place name or saved location is refused.
LOCATION_ERROR_STATUS = {
    "invalid_request": 422,
    "place_too_vague": 422,
    "outside_service_area": 422,
    "place_not_found": 404,
    "places_unavailable": 503,
}


class SharedLocation(Model):
    """A location the client shares for one request only (never stored)."""

    label: str = Field(default="Current location", min_length=1, max_length=120)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class AgentRequest(Model):
    """One message to the assistant.

    Attributes:
        message: The student's request.
        conversation_id: Returned by a previous answer, to continue it.
        location: Optional location shared for this request only.
        timezone: Optional IANA time zone, such as ``Europe/Paris``.
    """

    message: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = Field(default=None, max_length=100)
    location: SharedLocation | None = None
    timezone: str | None = Field(default=None, max_length=60)

    @field_validator("timezone")
    @classmethod
    def known_timezone(cls, value: str | None) -> str | None:
        """Reject a time zone name the system does not know.

        Args:
            value: Time zone name, or ``None``.

        Returns:
            The same value.

        Raises:
            ValueError: If it is not a known IANA time zone.
        """
        if value is not None:
            try:
                ZoneInfo(value)
            except (ZoneInfoNotFoundError, ValueError) as error:
                raise ValueError(f"Unknown time zone {value!r}") from error
        return value


class SavedLocationRequest(Model):
    """Opt in to saving a home location: coordinates, or a place name to look up."""

    label: str = Field(min_length=2, max_length=120)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def both_or_neither(self) -> "SavedLocationRequest":
        """Require both coordinates, or none (then ``label`` is geocoded).

        Returns:
            The validated request.

        Raises:
            ValueError: If only one coordinate is given.
        """
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Give both latitude and longitude, or neither.")
        return self


class SavedLocation(Model):
    """The saved location, as returned by the API."""

    label: str
    latitude: float
    longitude: float
    consented_at: str


def build_agent_router(
    *,
    require_account: Callable[..., str],
    check_owner: Callable[[str, str], None],
    make_context: Callable[..., ToolContext],
    make_llm: Callable[[], object],
    checkpointer: object,
) -> APIRouter:
    """Create the agent, approval and location routes.

    Args:
        require_account: FastAPI dependency returning the authenticated ID.
        check_owner: Raises 403 when the path account is not the caller.
        make_context: Builds a trusted ToolContext for an account
            (``make_context(student_id, session_location=None, timezone=None)``).
        make_llm: Returns the chat model the agents use.
        checkpointer: Shared LangGraph checkpointer for conversation history.

    Returns:
        A router to include in the FastAPI app.
    """
    router = APIRouter(prefix="/api/v1/students/{student_id}")

    @router.post("/agent/requests", response_model=AgentResponse)
    def ask_agent(
        student_id: str, request: AgentRequest, user_id: str = Depends(require_account)
    ) -> AgentResponse:
        """Run one request through the orchestrator and its specialists.

        Args:
            student_id: Account ID from the path.
            request: Message, optional conversation, location and time zone.
            user_id: Authenticated account ID.

        Returns:
            The answer, the executed plan and any pending actions.

        Raises:
            HTTPException: 403 when the path account is not the caller.
        """
        check_owner(student_id, user_id)
        location = (
            Location(**request.location.model_dump())
            if request.location is not None
            else None
        )
        context = make_context(
            student_id, session_location=location, timezone=request.timezone
        )
        conversation_id = request.conversation_id or str(uuid4())
        orchestrator = build_orchestrator(context, make_llm(), checkpointer)
        # Thread IDs include the account, so two accounts can never share history.
        response = orchestrator.ask(request.message, f"{student_id}:{conversation_id}")
        logger.info(
            "Agent request done steps=%s pending=%s seconds=%.1f",
            len(response.steps),
            len(response.pending_actions),
            response.seconds,
        )
        return response.model_copy(update={"conversation_id": conversation_id})

    @router.get("/agent/actions", response_model=list[approvals.PendingAction])
    def list_actions(
        student_id: str,
        status: Literal[
            "pending", "executed", "failed", "declined", "expired", "all"
        ] = Query("pending"),
        user_id: str = Depends(require_account),
    ) -> list[approvals.PendingAction]:
        """List the account's actions (pending by default), newest first.

        Raises:
            HTTPException: 403 when the path account is not the caller.
        """
        check_owner(student_id, user_id)
        return approvals.list_actions(
            make_context(student_id), None if status == "all" else status
        )

    @router.post(
        "/agent/actions/{action_id}/approve", response_model=approvals.ActionDecision
    )
    def approve_action(
        student_id: str, action_id: str, user_id: str = Depends(require_account)
    ) -> approvals.ActionDecision:
        """Run one pending action the student approved.

        Raises:
            HTTPException: 403 for another account, 404 unknown action,
                409 already decided or tampered, 410 expired.
        """
        check_owner(student_id, user_id)
        try:
            return approvals.approve(make_context(student_id), action_id)
        except approvals.ApprovalError as error:
            raise HTTPException(
                APPROVAL_ERROR_STATUS[error.code], error.message
            ) from error

    @router.post(
        "/agent/actions/{action_id}/decline", response_model=approvals.ActionDecision
    )
    def decline_action(
        student_id: str, action_id: str, user_id: str = Depends(require_account)
    ) -> approvals.ActionDecision:
        """Refuse one pending action; nothing runs.

        Raises:
            HTTPException: 403 for another account, 404 unknown action,
                409 already decided, 410 expired.
        """
        check_owner(student_id, user_id)
        try:
            return approvals.decline(make_context(student_id), action_id)
        except approvals.ApprovalError as error:
            raise HTTPException(
                APPROVAL_ERROR_STATUS[error.code], error.message
            ) from error

    @router.put("/location", response_model=SavedLocation)
    def save_location(
        student_id: str,
        body: SavedLocationRequest,
        user_id: str = Depends(require_account),
    ) -> SavedLocation:
        """Opt in to saving a home location for nearby searches.

        Without coordinates, ``label`` is looked up with the map service.

        Raises:
            HTTPException: 403 for another account, 404 when the place name
                is not found, 422 for a place that is too vague or outside the
                service area, 503 when the map service is unavailable.
        """
        check_owner(student_id, user_id)
        context = make_context(student_id)
        latitude, longitude, label = body.latitude, body.longitude, body.label
        if latitude is None or longitude is None:
            found = resolve_place(context.places, label)
            if isinstance(found, ToolFailure):
                raise HTTPException(LOCATION_ERROR_STATUS[found.code], found.error)
            latitude, longitude = found.latitude, found.longitude
        elif not in_service_area(latitude, longitude):
            refused = outside_area_failure(label)
            raise HTTPException(LOCATION_ERROR_STATUS[refused.code], refused.error)
        context.repository.save_location(student_id, label, latitude, longitude)
        logger.info("Saved location updated")
        return SavedLocation.model_validate(context.repository.get_location(student_id))

    @router.get("/location", response_model=SavedLocation)
    def get_location(
        student_id: str, user_id: str = Depends(require_account)
    ) -> SavedLocation:
        """Read the saved location.

        Raises:
            HTTPException: 403 for another account, 404 when none is saved.
        """
        check_owner(student_id, user_id)
        saved = make_context(student_id).repository.get_location(student_id)
        if saved is None:
            raise HTTPException(404, "No location is saved.")
        return SavedLocation.model_validate(saved)

    @router.delete("/location", status_code=204)
    def delete_location(
        student_id: str, user_id: str = Depends(require_account)
    ) -> None:
        """Forget the saved location.

        Raises:
            HTTPException: 403 for another account, 404 when none is saved.
        """
        check_owner(student_id, user_id)
        if not make_context(student_id).repository.delete_location(student_id):
            raise HTTPException(404, "No location is saved.")

    return router
