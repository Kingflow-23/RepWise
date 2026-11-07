"""Streamlit client for the RepWise API: plans, feedback, the multi-agent assistant,
approvals, saved location and a system view of the tools and the MCP server.

Run from the repository root, with the API already started:

    python -m streamlit run .\\genai_app\\streamlit_app.py

The app only talks to the HTTP API (``REPWISE_API_URL``). The one exception is
the System tab, which reads the local tool catalog ``config/tools.json`` to
display it. Screens fetch data when you click, so nothing is called behind
your back except the health check.
"""

from __future__ import annotations

import csv
import html
import io
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import requests
import streamlit as st

# Streamlit may run this file with ``genai_app/`` as the import root. Add the
# project root so local packages (config, src) resolve from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.app_config import get_api_url  # noqa: E402
from src.observability import configure_logging  # noqa: E402

configure_logging("streamlit")
logger = logging.getLogger(__name__)
st.set_page_config(page_title="RepWise", page_icon="🏋️", layout="wide")

API_BASE = get_api_url()
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
COMPLETION_LABELS = {
    "completed": "Completed",
    "partially_completed": "Partially completed",
    "not_completed": "Not completed",
}
DIFFICULTY_LABELS = {
    "much_too_easy": "Much too easy",
    "slightly_too_easy": "Slightly too easy",
    "just_right": "Just right",
    "slightly_too_hard": "Slightly too hard",
    "much_too_hard": "Much too hard",
}
AGENTS = {
    "orchestrator": ("🧭", "Orchestrator"),
    "planning": ("🗓️", "Workout Planning"),
    "performance": ("📈", "Performance & Adaptation"),
    "schedule": ("⏰", "Schedule & Calendar"),
    "location": ("📍", "Location"),
    "knowledge": ("📚", "Fitness Knowledge"),
}
STATUS_COLORS = {
    "completed": "green",
    "executed": "green",
    "active": "green",
    "pending": "orange",
    "draft": "orange",
    "limit_reached": "orange",
    "failed": "red",
    "declined": "gray",
    "expired": "gray",
    "archived": "gray",
    "not_run": "gray",
}
EXAMPLE_REQUESTS = [
    "Pourrais-tu me trouver la salle de sport la plus près de moi ?",
    "Dans quel magasin proche de moi puis-je trouver des haltères ?",
    "Dans quel magasin proche de moi puis-je trouver de la créatine ?",
    "Comment pourrais-je organiser mes séances de la semaine en fonction de mon programme ?",
    "Propose-moi un programme pour prendre du muscle au niveau de mes jambes.",
    "Find a gym near me and create a training programme.",
    "I missed two workouts this week; organise the rest of my week.",
    "How much rest should I take between sets for hypertrophy, according to your documents?",
    "Remember that I prefer training in the morning.",
]
SESSION_KEYS_PER_ACCOUNT = (
    "profile_form",
    "plan_history",
    "latest_plan",
    "pending_delete_plan_id",
    "plan_notice",
    "plan_generation_error",
    "plan_generation_request",
    "plan-focus",
    "plan-recording",
    "plan_voice_notice",
    "plan_voice_error",
    "agent_messages",
    "agent_conversation_id",
    "agent_actions",
    "action_notice",
    "saved_location",
    "request_log",
)


class ApiError(RuntimeError):
    """A readable API failure for display in the app."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: Any = None,
    ) -> None:
        """Store the message and optional response context for UI handling.

        Args:
            message: Human-readable error shown by the Streamlit client.
            status_code: HTTP status returned by the backend, if available.
            detail: Structured error detail returned by the backend, if any.
        """
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def api_request(method: str, path: str, **kwargs: Any) -> Any:
    """Send an authenticated backend request and translate failures for the UI.

    Args:
        method: HTTP verb used for the API call.
        path: API route appended to the configured backend URL.
        **kwargs: Additional request options passed to ``requests.request``.

    Returns:
        Decoded response JSON, or ``None`` for an empty response.

    Raises:
        ApiError: If the backend cannot be reached or returns an error status.
    """
    # The backend derives ownership from this signed token, not a client-sent ID.
    token = st.session_state.get("access_token")
    if token:
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {token}"
        kwargs["headers"] = headers
    started = perf_counter()
    try:
        response = requests.request(
            method, f"{API_BASE}{path}", timeout=(5, 950), **kwargs
        )
    except requests.RequestException as error:
        logger.error(
            "Backend request failed method=%s error_type=%s elapsed_ms=%.1f",
            method,
            type(error).__name__,
            (perf_counter() - started) * 1000,
        )
        raise ApiError(
            f"Cannot reach the RepWise API at {API_BASE}. Start it with "
            f"`python -m uvicorn src.api.main:app --reload`. Details: {error}"
        ) from error
    if not response.ok:
        logger.warning(
            "Backend returned error method=%s status=%s request_id=%s elapsed_ms=%.1f",
            method,
            response.status_code,
            response.headers.get("X-Request-ID", "-"),
            (perf_counter() - started) * 1000,
        )
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if isinstance(detail, dict) and isinstance(detail.get("message"), str):
            raise ApiError(
                detail["message"], status_code=response.status_code, detail=detail
            )
        raise ApiError(
            f"API returned {response.status_code}: {detail}",
            status_code=response.status_code,
            detail=detail,
        )
    logger.info(
        "Backend request completed method=%s status=%s request_id=%s elapsed_ms=%.1f",
        method,
        response.status_code,
        response.headers.get("X-Request-ID", "-"),
        (perf_counter() - started) * 1000,
    )
    return response.json() if response.content else None


def badge(text: str, status: str | None = None) -> str:
    """Return a Streamlit colour badge for a status word.

    Args:
        text: Label to show.
        status: Status used to choose the colour; defaults to ``text``.

    Returns:
        Markdown such as ``:green-badge[completed]``.
    """
    color = STATUS_COLORS.get(status or text, "blue")
    return f":{color}-badge[{text}]"


def generation_error_message(error: ApiError) -> str:
    """Translate a plan-generation API failure into actionable retry guidance.

    Args:
        error: API failure, including any structured backend status and detail.

    Returns:
        A user-facing explanation that does not imply saved plans were changed.
    """
    detail = error.detail if isinstance(error.detail, dict) else {}
    status = detail.get("status")
    reason = str(detail.get("message", str(error))).casefold()

    if status == "insufficient_evidence":
        cause = "there wasn't enough relevant training material for this request."
    elif status == "retrieval_error":
        cause = "the training library couldn't be reached."
    elif "equipment" in reason:
        cause = "the generated plan used equipment outside your saved profile."
    elif "not an available day" in reason:
        cause = "the generated schedule used a day you marked unavailable."
    elif "more workouts than available days" in reason:
        cause = "you requested more workouts than your available days allow."
    elif "duration" in reason or "longer than" in reason:
        cause = "a workout exceeded your session time limit."
    elif "workout" in reason and (
        "exactly" in reason or "count" in reason or "available days" in reason
    ):
        cause = "the plan didn't match the number of workouts you requested."
    elif "week" in reason:
        cause = "the plan didn't match the requested week structure."
    elif "source" in reason or "citation" in reason or "quote" in reason:
        cause = "the generated plan's source references didn't match the retrieved material."
    elif "timeout" in reason or "timed out" in reason:
        cause = "the model took too long to respond."
    elif "cannot reach the repwise api" in reason:
        cause = "the RepWise API couldn't be reached."
    elif "connection" in reason or "ollama" in reason:
        cause = "the language model service couldn't be reached."
    else:
        cause = "the model couldn't return a plan that met the request requirements."

    return (
        f"We couldn't generate this plan because {cause} "
        "Your saved plans were not changed. Please try generating again; a fresh "
        "attempt may produce a plan that fits your request."
    )


def save_generated_plan(
    student_id: str, payload: dict[str, Any], focus: str, workouts_per_week: int
) -> None:
    """Generate a plan and update the UI's latest-plan and history state.

    Args:
        student_id: Authenticated account ID used by the backend route.
        payload: Plan options submitted to the generation endpoint.
        focus: Display label stored with the generated plan record.
        workouts_per_week: Requested weekly workout count stored for display.

    API failures and out-of-scope requests are represented in session state;
    successful results are added to both the latest-plan and history state.
    """
    logger.info("Plan generation requested through Streamlit")
    st.session_state.pop("plan_scope_message", None)
    started = perf_counter()
    try:
        result = api_request(
            "POST", f"/api/v1/students/{student_id}/plans", json=payload
        )
    except ApiError as error:
        log_request("plan", focus, perf_counter() - started, error=str(error))
        logger.warning(
            "Plan generation failed through Streamlit status=%s",
            error.status_code or "unavailable",
        )
        st.session_state.plan_generation_error = generation_error_message(error)
        return

    log_request(
        "plan",
        focus,
        perf_counter() - started,
        status=result.get("status"),
        sources=len(result.get("sources", [])),
    )
    if result.get("status") == "out_of_scope":
        st.session_state.plan_scope_message = result.get(
            "message",
            "RepWise can't generate a plan for this request. Please revise it to fit "
            "general fitness planning.",
        )
        st.session_state.pop("plan_generation_error", None)
        st.session_state.pop("latest_plan", None)
        logger.info("Plan request declined by scope gate")
        return

    record = {
        "id": result.get("plan_id"),
        "plan": result.get("plan"),
        "sources": result.get("sources", []),
        "status": result.get("plan_status") or "active",
        "request": {
            "focus": focus,
            "weeks": 1,
            "workouts_per_week": workouts_per_week,
        },
        "created_at": datetime.now().astimezone().isoformat(),
    }
    st.session_state.latest_plan = record
    st.session_state.plan_history = [
        record,
        *[
            saved
            for saved in st.session_state.get("plan_history", [])
            if saved.get("id") != record["id"]
        ],
    ]
    st.session_state.pop("plan_generation_error", None)
    st.session_state.pop("plan_scope_message", None)
    logger.info("Plan generation completed through Streamlit")
    st.success("Your one-week plan was generated and saved.")


def highlighted_evidence(text: str, matched_terms: list[str]) -> str:
    """Escape source text and mark exact query-term matches for review.

    Args:
        text: Full retrieved extract to render without interpreting its HTML.
        matched_terms: Literal terms from the request, profile, and effective
            retrieval query; these are escaped and regex-quoted before matching.

    Returns:
        HTML with exact case-insensitive matches wrapped in ``<mark>`` tags.
        It is a lexical review aid, not an explanation of model reasoning.
    """
    terms = sorted(set(matched_terms), key=len, reverse=True)
    if not terms:
        return html.escape(text)

    pattern = re.compile(
        rf"(?<!\w)({'|'.join(re.escape(term) for term in terms)})(?!\w)",
        flags=re.IGNORECASE | re.UNICODE,
    )
    rendered: list[str] = []
    previous_end = 0
    for match in pattern.finditer(text):
        rendered.append(html.escape(text[previous_end : match.start()]))
        rendered.append(f"<mark>{html.escape(match.group(0))}</mark>")
        previous_end = match.end()
    rendered.append(html.escape(text[previous_end:]))
    return "".join(rendered)


def submit_workout_feedback(
    record: dict[str, Any],
    week_number: int,
    workout_index: int,
    completion_status: str,
    difficulty: str,
    comment: str,
) -> None:
    """Send feedback for one workout and refresh the displayed plan.

    Args:
        record: The plan record being displayed; updated in place.
        week_number: Week of the workout.
        workout_index: Position of the workout in its week.
        completion_status: Completion value from the form.
        difficulty: Difficulty value from the form.
        comment: Free-text comment (may be empty).
    """
    student_id = st.session_state.get("user_id")
    try:
        result = api_request(
            "POST",
            f"/api/v1/students/{student_id}/plans/{record['id']}"
            f"/weeks/{week_number}/workouts/{workout_index}/feedback",
            json={
                "completion_status": completion_status,
                "difficulty": difficulty,
                "comment": comment.strip() or None,
            },
        )
    except ApiError as error:
        st.error(str(error))
        return
    record["plan"] = result["plan"]
    record["workout_feedback"] = result["workout_feedback"]
    if result["adapted"]:
        st.success(result["message"])
    else:
        st.info(result["message"])
    st.rerun()


def show_workout(
    record: dict[str, Any],
    week: dict[str, Any],
    workout_index: int,
    feedback_by_workout: dict,
    adapted_workouts: dict,
    form_prefix: str,
) -> None:
    """Render one workout, its exercises and its feedback form or saved feedback.

    Args:
        record: The plan record.
        week: The week holding the workout.
        workout_index: Position of the workout in the week.
        feedback_by_workout: Saved feedback keyed by ``(week, index)``.
        adapted_workouts: Feedback that adapted a workout, keyed by its target.
        form_prefix: Makes widget keys unique when a plan is shown twice.
    """
    workout = week["workouts"][workout_index]
    workout_key = (week["number"], workout_index)
    adaptation_note = (
        " · :violet-badge[adapted from your feedback]"
        if workout_key in adapted_workouts
        else ""
    )
    st.markdown(
        f"**{workout['day']} · {workout['focus']}** — "
        f"about {workout['estimated_minutes']} min{adaptation_note}"
    )
    for exercise in workout.get("exercises", []):
        exercise_minutes = exercise.get("estimated_minutes")
        time_estimate = (
            f"~{exercise_minutes} min"
            if isinstance(exercise_minutes, int)
            else "time estimate unavailable"
        )
        st.markdown(
            f"- **{exercise['name']}** · {exercise['sets']} × "
            f"{exercise['repetitions']} · rest {exercise['rest_seconds']}s · {time_estimate}"
        )
        st.caption(exercise.get("coaching_note", ""))
        for evidence in exercise.get("supporting_evidence", []):
            st.caption(f"Evidence `{evidence['source_id']}`: “{evidence['quote']}”")
    saved_feedback = feedback_by_workout.get(workout_key)
    if saved_feedback:
        st.success("Workout logged")
        concern = (
            " · ⚠️ may mention pain or injury (no automatic change)"
            if saved_feedback.get("health_concern")
            else ""
        )
        st.caption(
            f"{COMPLETION_LABELS.get(saved_feedback.get('completion_status'), saved_feedback.get('completion_status'))} · "
            f"{DIFFICULTY_LABELS.get(saved_feedback.get('difficulty'), saved_feedback.get('difficulty'))}{concern}"
        )
        if saved_feedback.get("comment"):
            st.write(saved_feedback["comment"])
    elif record.get("id") and st.session_state.get("user_id"):
        with st.form(
            f"log-workout-{form_prefix}-{record['id']}-{week['number']}-{workout_index}"
        ):
            left, right = st.columns(2)
            completion_status = left.selectbox(
                "Completion", list(COMPLETION_LABELS), format_func=COMPLETION_LABELS.get
            )
            difficulty = right.selectbox(
                "Overall difficulty",
                list(DIFFICULTY_LABELS),
                format_func=DIFFICULTY_LABELS.get,
            )
            comment = st.text_area(
                "Comment (optional)",
                max_chars=500,
                key=f"workout-comment-{form_prefix}-{record['id']}-{week['number']}-{workout_index}",
            )
            submitted = st.form_submit_button("Log workout")
        if submitted:
            submit_workout_feedback(
                record,
                week["number"],
                workout_index,
                completion_status,
                difficulty,
                comment,
            )


def show_sources(sources: list[dict[str, Any]]) -> None:
    """Render a plan's sources with their full extracts.

    Args:
        sources: Source records saved with the plan.
    """
    st.markdown("**Sources used**")
    for source in sources:
        st.markdown(f"- [{source['title']}]({source['url']}) · `{source['source_id']}`")
        evidence_text = source.get("evidence_text")
        if evidence_text:
            with st.expander("Full retrieved extract"):
                matched_terms = source.get("matched_terms", [])
                if matched_terms:
                    st.caption(
                        f"Search and profile terms found in this extract ({len(matched_terms)}): "
                        + ", ".join(matched_terms)
                    )
                else:
                    st.caption(
                        "No exact intent-term matches found. The semantic retriever "
                        "may still have selected this related passage."
                    )
                st.html(
                    '<div style="white-space: pre-wrap; overflow-wrap: anywhere">'
                    + highlighted_evidence(evidence_text, matched_terms)
                    + "</div>"
                )
        elif source.get("excerpt"):
            st.caption(source["excerpt"])
            st.caption("Full evidence is unavailable for this saved plan.")


def show_plan(record: dict[str, Any], *, form_prefix: str = "plan") -> None:
    """Render a saved plan, its supporting evidence, and its source details.

    Args:
        record: Plan history record containing the plan, request, and sources.
        form_prefix: Makes widget keys unique when a plan is shown twice.
    """
    plan = record.get("plan") or {}
    options = record.get("request") or {}
    status = record.get("status")
    st.subheader(plan.get("title", "Training plan"))
    st.caption(
        f"{options.get('focus', 'Plan')} · "
        f"{options.get('weeks', len(plan.get('weeks', [])))} weeks · "
        f"Created {record.get('created_at', 'just now')}"
    )
    if status:
        st.markdown(badge(status))
    if status == "draft":
        st.warning(
            "This plan is a draft: it replaces your current plan only after you "
            "approve its activation (Approvals tab)."
        )
    st.write(plan.get("summary", ""))
    feedback_records = record.get("workout_feedback", [])
    feedback_by_workout = {
        (item.get("week_number"), item.get("workout_index")): item
        for item in feedback_records
    }
    adapted_workouts = {
        (item.get("adapted_week_number"), item.get("adapted_workout_index")): item
        for item in feedback_records
        if item.get("adapted_week_number") is not None
    }
    for week in plan.get("weeks", []):
        with st.expander(
            f"Week {week['number']} — {week['objective']}",
            expanded=len(plan.get("weeks", [])) == 1,
        ):
            for workout_index in range(len(week.get("workouts", []))):
                show_workout(
                    record,
                    week,
                    workout_index,
                    feedback_by_workout,
                    adapted_workouts,
                    form_prefix,
                )
    if plan.get("assumptions"):
        st.markdown("**Assumptions**")
        for assumption in plan["assumptions"]:
            st.markdown(f"- {assumption}")
    if plan.get("safety_note"):
        st.info(plan["safety_note"])
    if record.get("sources"):
        show_sources(record["sources"])


def use_plan_recording() -> None:
    """Replace only the editable focus draft; never submit or generate a plan."""
    st.session_state.pop("plan_voice_error", None)
    st.session_state.pop("plan_voice_notice", None)
    recording = st.session_state.get("plan-recording")
    if recording is None:
        st.session_state.plan_voice_error = "Record your request first."
        return
    try:
        with st.spinner("Transcribing your recording…"):
            result = api_request(
                "POST",
                "/api/v1/speech/transcribe",
                data=recording.getvalue(),
                headers={"Content-Type": "audio/wav"},
            )
        text = result.get("text") if isinstance(result, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "No speech was detected. Please record your request again."
            )
        st.session_state["plan-focus"] = text.strip()
        st.session_state.plan_voice_notice = "Recording transcribed. Review and edit your request before generating a plan."
    except (ApiError, ValueError) as error:
        st.session_state.plan_voice_error = str(error)


def store_account(account: dict[str, Any]) -> None:
    """Keep the authenticated account identity in Streamlit session state.

    Args:
        account: API account response with ``user_id`` and ``full_name``. A
            changed owner clears cached state, so data from one account never
            appears under another.
    """
    if st.session_state.get("user_id") != account["user_id"]:
        for key in SESSION_KEYS_PER_ACCOUNT:
            st.session_state.pop(key, None)
    st.session_state.user_id = account["user_id"]
    st.session_state.full_name = account["full_name"]


def sign_out() -> None:
    """Forget the token and every account-specific value of this session."""
    for key in (*SESSION_KEYS_PER_ACCOUNT, "access_token", "user_id", "full_name"):
        st.session_state.pop(key, None)


# --- Assistant and approvals ------------------------------------------------


def decide_action(action_id: str, decision: str) -> None:
    """Approve or decline one pending action and remember the outcome.

    Args:
        action_id: The action to decide.
        decision: ``"approve"`` or ``"decline"``.
    """
    student_id = st.session_state.get("user_id")
    try:
        result = api_request(
            "POST",
            f"/api/v1/students/{student_id}/agent/actions/{action_id}/{decision}",
        )
    except ApiError as error:
        reasons = {
            404: "it no longer exists for this account",
            409: "it was already decided",
            410: "it expired",
        }
        why = reasons.get(error.status_code, str(error))
        st.session_state.action_notice = (
            "error",
            f"Could not {decision} this action: {why}.",
        )
        return
    action = result["action"]
    st.session_state.setdefault("decided_actions", {})[action_id] = action
    if decision == "approve" and action["status"] == "executed":
        message = f"Done: {action['tool_name']} ran after your approval."
        # Plans may have changed (activation, schedule, adaptation): reload them.
        st.session_state.pop("plan_history", None)
        st.session_state.pop("latest_plan", None)
        st.session_state.action_notice = ("success", message)
    elif decision == "approve":
        error = (action.get("result") or {}).get("error", "the tool failed")
        st.session_state.action_notice = (
            "error",
            f"Approved, but {action['tool_name']} failed: {error}",
        )
    else:
        st.session_state.action_notice = (
            "info",
            f"Declined: {action['tool_name']} did not run.",
        )
    st.session_state.agent_actions = [
        item
        for item in st.session_state.get("agent_actions", [])
        if item.get("id") != action_id
    ]


def show_action(action: dict[str, Any], key_prefix: str) -> None:
    """Render one pending action with its reason and decision buttons.

    Args:
        action: A pending action (``action_id`` or ``id``, ``tool_name``,
            ``reason``, ``expires_at``).
        key_prefix: Makes the button keys unique on the page.
    """
    action_id = action.get("action_id") or action.get("id")
    decided = st.session_state.get("decided_actions", {}).get(action_id)
    with st.container(border=True):
        st.markdown(
            f"**{action['tool_name']}** · {badge(decided['status'] if decided else 'pending')}"
        )
        st.write(action.get("reason", ""))
        if action.get("arguments"):
            with st.expander("Exact arguments that will run"):
                st.json(action["arguments"])
        st.caption(
            f"Action `{action_id}` · expires {str(action.get('expires_at', ''))[:19]}"
        )
        if decided:
            return
        approve, decline = st.columns(2)
        approve.button(
            "Approve",
            key=f"approve-{key_prefix}-{action_id}",
            type="primary",
            icon=":material/check:",
            width="stretch",
            on_click=decide_action,
            args=(action_id, "approve"),
        )
        decline.button(
            "Decline",
            key=f"decline-{key_prefix}-{action_id}",
            icon=":material/close:",
            width="stretch",
            on_click=decide_action,
            args=(action_id, "decline"),
        )


def show_agent_response(response: dict[str, Any], key_prefix: str) -> None:
    """Render the orchestrator's plan, specialists' work and pending actions.

    Args:
        response: An ``AgentResponse`` from the API.
        key_prefix: Makes widget keys unique for this message.
    """
    steps = response.get("steps", [])
    if steps:
        timeline = "  →  ".join(
            f"{AGENTS.get(step['agent'], ('•', step['agent']))[0]} "
            f"{AGENTS.get(step['agent'], ('', step['agent']))[1]} {badge(step['status'])}"
            for step in steps
        )
        st.markdown(f"🧭 **Plan:** {timeline}")
    else:
        st.caption(
            "🧭 Answered by the orchestrator from your saved data, without specialists."
        )
    meta = [f"⏱ {response.get('seconds', 0):.0f} s"]
    if response.get("replans"):
        meta.append(f"🔁 replanned {response['replans']}×")
    if response.get("health_concern"):
        meta.append("⚕️ health concern detected")
    st.caption(" · ".join(meta))
    reports = response.get("reports", [])
    if reports:
        with st.expander("What each agent did"):
            for report in reports:
                icon, title = AGENTS.get(report["agent"], ("•", report["agent"]))
                st.markdown(
                    f"**{icon} {title}** {badge(report['status'])} · "
                    f"{report.get('seconds', 0):.1f} s"
                )
                st.caption(f"Task: {report['task']}")
                for call in report.get("tools_used", []):
                    outcome = "✅" if call.get("ok") else f"❌ `{call.get('code')}`"
                    held = (
                        " · ⏸ waits for approval"
                        if call.get("pending_approval")
                        else ""
                    )
                    st.markdown(
                        f"- `{call['name']}` {outcome} · {call.get('ms', 0)} ms{held}"
                    )
                if report.get("summary"):
                    st.caption(f"Report: {report['summary'][:600]}")
                if report.get("error"):
                    st.caption(f"Error: {report['error']}")
                if report.get("results"):
                    st.json(report["results"], expanded=False)
    if response.get("metrics"):
        with st.expander("📊 Performance details (time, tokens, model and tool calls)"):
            show_metrics(response, key_prefix)
    for action in response.get("pending_actions", []):
        show_action(action, key_prefix)


def log_request(kind: str, text: str, seconds: float, **details: Any) -> None:
    """Add one request to the session's evaluation log (Metrics tab).

    Args:
        kind: ``agent`` or ``plan``.
        text: The request text.
        seconds: Round-trip time measured by the app.
        **details: ``response`` (an AgentResponse), ``error``, ``status``, ...
    """
    response = details.pop("response", None)
    entry: dict[str, Any] = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "kind": kind,
        "request": text,
        "round_trip_s": round(seconds, 2),
        **details,
    }
    if response:
        metrics = response.get("metrics") or {}
        steps = response.get("steps", [])
        entry.update(
            server_s=metrics.get("total_seconds"),
            agents=" → ".join(step["agent"] for step in steps) or "none",
            step_statuses=", ".join(step["status"] for step in steps),
            replans=response.get("replans", 0),
            model_calls=len(metrics.get("model_calls", [])),
            input_tokens=metrics.get("input_tokens"),
            output_tokens=metrics.get("output_tokens"),
            tool_calls=metrics.get("tool_calls"),
            tool_s=metrics.get("tool_seconds"),
            pending_actions=len(response.get("pending_actions", [])),
            health_concern=response.get("health_concern", False),
            model=metrics.get("model"),
        )
    st.session_state.setdefault("request_log", []).append(entry)


def show_metrics(response: dict[str, Any], key_prefix: str) -> None:
    """Render every measurement of one agent request.

    Args:
        response: An ``AgentResponse`` with ``metrics``.
        key_prefix: Makes widget keys unique for this message.
    """
    metrics = response["metrics"]
    calls = metrics.get("model_calls", [])
    generated = sum(call.get("output_tokens") or 0 for call in calls)
    model_time = sum(call["seconds"] for call in calls) or 1
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Server time", f"{metrics['total_seconds']:.1f} s")
    k2.metric("Round trip", f"{response.get('client_seconds', 0):.1f} s")
    k3.metric("Model calls", len(calls))
    k4.metric(
        "Tokens in / out", f"{metrics['input_tokens']} / {metrics['output_tokens']}"
    )
    k5.metric(
        "Tool calls", f"{metrics['tool_calls']} ({metrics['tool_seconds']:.2f} s)"
    )
    st.caption(
        f"Model `{metrics['model']}` · about {generated / model_time:.1f} generated "
        "tokens per second of model time."
    )
    phases = metrics.get("phases", [])
    if phases:
        st.markdown("**Time per orchestrator phase (seconds)**")
        st.bar_chart(
            {
                "seconds": {
                    f"{index + 1}. {phase['name']}": phase["seconds"]
                    for index, phase in enumerate(phases)
                }
            },
            horizontal=True,
        )
    if calls:
        st.markdown("**Model calls**")
        st.dataframe(
            [
                {
                    "purpose": call["purpose"],
                    "seconds": call["seconds"],
                    "input tokens": call.get("input_tokens"),
                    "output tokens": call.get("output_tokens"),
                }
                for call in calls
            ],
            hide_index=True,
            width="stretch",
        )
    tools = [
        {"agent": report["agent"], **call}
        for report in response.get("reports", [])
        for call in report.get("tools_used", [])
    ]
    if tools:
        st.markdown("**Tool calls**")
        st.dataframe(tools, hide_index=True, width="stretch")
    st.download_button(
        "Download this response as JSON",
        data=json.dumps(response, indent=2, ensure_ascii=False),
        file_name=f"repwise-agent-{key_prefix}.json",
        mime="application/json",
        key=f"download-{key_prefix}",
    )


def metrics_csv(rows: list[dict[str, Any]]) -> str:
    """Turn the session log into CSV text.

    Args:
        rows: Entries written by :func:`log_request`.

    Returns:
        CSV with one column per key seen in any row.
    """
    columns = sorted({key for row in rows for key in row})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def send_agent_message(message: str) -> None:
    """Send one message to the multi-agent assistant and store the reply.

    Args:
        message: The student's request.
    """
    student_id = st.session_state.get("user_id")
    payload: dict[str, Any] = {
        "message": message,
        "timezone": st.session_state.get("agent-timezone") or None,
        "conversation_id": st.session_state.get("agent_conversation_id"),
    }
    if st.session_state.get("agent-share-location"):
        payload["location"] = {
            "label": st.session_state.get("agent-location-label") or "Current location",
            "latitude": st.session_state.get("agent-latitude"),
            "longitude": st.session_state.get("agent-longitude"),
        }
    messages = st.session_state.setdefault("agent_messages", [])
    messages.append({"role": "user", "content": message})
    started = perf_counter()
    try:
        with st.spinner("The agents are working… a plan can take one to two minutes."):
            response = api_request(
                "POST", f"/api/v1/students/{student_id}/agent/requests", json=payload
            )
    except ApiError as error:
        log_request("agent", message, perf_counter() - started, error=str(error))
        messages.append(
            {"role": "assistant", "content": f"⚠️ {error}", "response": None}
        )
        return
    response["client_seconds"] = round(perf_counter() - started, 3)
    log_request("agent", message, response["client_seconds"], response=response)
    st.session_state.agent_conversation_id = response.get("conversation_id")
    messages.append(
        {"role": "assistant", "content": response["answer"], "response": response}
    )


def use_example(text: str) -> None:
    """Put an example request in the assistant's message box.

    Args:
        text: Example request.
    """
    st.session_state["agent-draft"] = text


def take_agent_draft() -> None:
    """Queue the typed message for sending, and empty the message box.

    Runs as the Send button's callback, before the page reruns, so the box
    can be cleared without a Streamlit error.
    """
    text = st.session_state.get("agent-draft", "").strip()
    if not text:
        st.session_state["agent_draft_empty"] = True
        return
    st.session_state["agent_queued_message"] = text
    st.session_state["agent-draft"] = ""


def start_new_conversation() -> None:
    """Forget the assistant conversation shown on the page."""
    st.session_state.pop("agent_messages", None)
    st.session_state.pop("agent_conversation_id", None)


def load_actions(status: str = "pending") -> None:
    """Fetch the account's actions into session state.

    Args:
        status: Filter passed to the API (``pending`` or ``all``...).
    """
    student_id = st.session_state.get("user_id")
    try:
        st.session_state.agent_actions = api_request(
            "GET",
            f"/api/v1/students/{student_id}/agent/actions",
            params={"status": status},
        )
    except ApiError as error:
        st.session_state.action_notice = ("error", str(error))


def show_action_notice() -> None:
    """Display (once) the outcome of the last approval decision."""
    notice = st.session_state.pop("action_notice", None)
    if notice:
        kind, text = notice
        {"success": st.success, "error": st.error}.get(kind, st.info)(text)


def load_tool_catalog() -> list[dict[str, Any]]:
    """Read the local tool catalog for the System tab.

    Returns:
        Catalog entries, or an empty list when the file cannot be read.
    """
    try:
        data = json.loads(
            (PROJECT_ROOT / "config" / "tools.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return []
    return data.get("tools", [])


# --- Page --------------------------------------------------------------------

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.rw-hero {padding: 1.4rem 1.6rem; border-radius: 18px; color: white;
  background: linear-gradient(120deg, #0f766e 0%, #155e75 55%, #1e3a8a 100%);
  box-shadow: 0 8px 24px rgba(15, 118, 110, .25); margin-bottom: 1rem;}
.rw-hero h1 {margin: 0; font-size: 2.1rem; color: white;}
.rw-hero p {margin: .35rem 0 0; opacity: .92; font-size: 1.02rem;}
.rw-agents {display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .9rem;}
.rw-agents span {background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
  padding: .25rem .65rem; border-radius: 999px; font-size: .85rem;}
[data-testid="stMetric"] {background: rgba(15,118,110,.07); padding: .8rem 1rem; border-radius: 14px;
  border: 1px solid rgba(15,118,110,.18);}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class="rw-hero">
  <h1>🏋️ RepWise</h1>
  <p>Evidence-grounded training plans and a multi-agent assistant that never changes
  your plan or calendar without your approval.</p>
  <div class="rw-agents">
    <span>🧭 Orchestrator</span><span>🗓️ Planning</span><span>📈 Performance</span>
    <span>⏰ Schedule</span><span>📍 Location</span><span>📚 Knowledge</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Account")
    if st.session_state.get("full_name"):
        st.success(f"Signed in as **{st.session_state.full_name}**")
        st.button("Sign out", width="stretch", on_click=sign_out)
    account_name = st.text_input(
        "Full name", value=st.session_state.get("full_name", "")
    )
    account_password = st.text_input("Password", type="password")
    sign_in, register = st.columns(2)
    if sign_in.button("Sign in", width="stretch"):
        try:
            token = api_request(
                "POST",
                "/api/v1/auth/login",
                json={"full_name": account_name, "password": account_password},
            )
            st.session_state.access_token = token["access_token"]
            store_account(token)
            logger.info("User signed in through Streamlit")
            st.success("Signed in.")
        except ApiError as error:
            st.error(str(error))
    if register.button("Create account", width="stretch"):
        try:
            token = api_request(
                "POST",
                "/api/v1/auth/register",
                json={"full_name": account_name, "password": account_password},
            )
            st.session_state.access_token = token["access_token"]
            store_account(token)
            logger.info("Account registered through Streamlit")
            st.success("Account created and signed in.")
        except ApiError as error:
            st.error(str(error))

    st.header("Backend")
    api_base = st.text_input(
        "Backend API",
        value=API_BASE,
        help="Set REPWISE_API_URL before launching Streamlit, or edit it here.",
    )
    if api_base.rstrip("/") != API_BASE:
        API_BASE = api_base.rstrip("/")
    if st.button("Check connection", width="stretch"):
        logger.info("Connection health refresh requested")
        st.session_state.pop("health", None)
        st.session_state.pop("health_error", None)
    if "health" not in st.session_state and "health_error" not in st.session_state:
        try:
            st.session_state.health = api_request("GET", "/api/v1/health")
        except ApiError as error:
            st.session_state.health_error = str(error)
    if st.session_state.get("health"):
        health = st.session_state.health
        st.success("API connected")
        st.caption(
            f"Retriever: {health.get('retrieval', 'unknown')} · "
            f"Strategy: {health.get('retriever', 'unknown')} · "
            f"Model: {health.get('model', 'unknown')}"
        )
    elif st.session_state.get("health_error"):
        st.error(st.session_state.health_error)
    st.divider()
    st.caption(
        "Plans are general fitness suggestions, not medical advice. Stop if an "
        "exercise causes pain and talk to a health professional."
    )

signed_in = bool(st.session_state.get("user_id"))
if signed_in:
    history_count = len(st.session_state.get("plan_history", []))
    pending_count = len(st.session_state.get("agent_actions", []))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Account", st.session_state.get("full_name", ""))
    m2.metric("Saved plans", history_count)
    m3.metric("Pending approvals", pending_count)
    m4.metric("Agent messages", len(st.session_state.get("agent_messages", [])))

(
    profile_tab,
    plan_tab,
    history_tab,
    agent_tab,
    approvals_tab,
    metrics_tab,
    location_tab,
    system_tab,
) = st.tabs(
    [
        "👤 Profile",
        "🗓️ Generate a plan",
        "📚 Saved plans",
        "🤖 Assistant",
        "✅ Approvals",
        "📊 Metrics",
        "📍 Location & calendar",
        "⚙️ System",
    ]
)

with profile_tab:
    st.subheader("Your training profile")
    st.write(
        "Save or update the profile every plan and agent uses to personalize advice."
    )
    student_id = st.session_state.get("user_id", "")
    if not student_id:
        st.info("Sign in or create an account to save your profile.")
    if st.button("Load saved profile"):
        if not student_id:
            st.error("Sign in or create an account first.")
        else:
            try:
                saved = api_request("GET", f"/api/v1/students/{student_id}/profile")
                st.session_state.profile_form = saved
                logger.info("Saved profile loaded into profile editor")
                st.success("Profile loaded.")
            except ApiError as error:
                st.error(str(error))
    profile = st.session_state.get("profile_form", {})
    with st.form("profile_editor_form"):
        left, right = st.columns(2)
        with left:
            experience = st.selectbox(
                "Experience",
                ["beginner", "intermediate", "advanced"],
                index=(
                    ["beginner", "intermediate", "advanced"].index(
                        profile.get("experience", "beginner")
                    )
                    if profile.get("experience")
                    in ["beginner", "intermediate", "advanced"]
                    else 0
                ),
            )
            goal = st.text_input(
                "Main training goal",
                value=profile.get("goal", "Build general strength"),
            )
            session_minutes = st.slider(
                "Available minutes per session",
                15,
                180,
                int(profile.get("session_minutes", 45)),
                step=5,
            )
        with right:
            defaults = [
                day
                for day in profile.get(
                    "available_days", ["Monday", "Wednesday", "Friday"]
                )
                if day in DAYS
            ]
            available_days = st.multiselect("Days available", DAYS, default=defaults)
            equipment_text = st.text_input(
                "Equipment (comma separated)",
                value=", ".join(profile.get("equipment", ["bodyweight"])),
            )
            preferences = st.text_area(
                "Preferences (optional)",
                value=profile.get("preferences") or "",
                placeholder="For example: shorter warm-up, low impact",
            )
        save = st.form_submit_button("Save profile", type="primary", width="stretch")
    if save:
        if not student_id:
            st.error("Sign in or create an account first.")
        elif not goal.strip() or not available_days:
            st.error("Add a goal and choose at least one available day.")
        else:
            payload = {
                "experience": experience,
                "goal": goal,
                "available_days": available_days,
                "session_minutes": session_minutes,
                "equipment": [
                    item.strip() for item in equipment_text.split(",") if item.strip()
                ],
                "preferences": preferences.strip() or None,
            }
            try:
                st.session_state.profile_form = api_request(
                    "PUT", f"/api/v1/students/{student_id}/profile", json=payload
                )
                logger.info("Profile saved through Streamlit")
                st.success("Profile saved to the backend database.")
            except ApiError as error:
                st.error(str(error))

with plan_tab:
    st.subheader("Create a personalized plan")
    st.write(
        "Each run creates one week of workouts on your available days, validated "
        "against your profile and the retrieved evidence. Sources are shown below the plan."
    )
    recording = st.audio_input(
        "Record your request (optional)",
        sample_rate=16000,
        key="plan-recording",
        disabled=not bool(st.session_state.get("access_token")),
        help="Record up to two minutes, then use the recording to fill the text field below.",
    )
    st.button(
        "Use recording",
        key="use-plan-recording",
        on_click=use_plan_recording,
        disabled=recording is None or not bool(st.session_state.get("access_token")),
        help="Replaces the request text below with your transcription. It does not generate a plan.",
    )
    if st.session_state.get("plan_voice_error"):
        st.error(st.session_state.plan_voice_error)
    if st.session_state.get("plan_voice_notice"):
        st.info(st.session_state.plan_voice_notice)
    # Keep the draft outside the form so recorder reruns preserve manual edits.
    focus = st.text_input(
        "What do you want to focus on?",
        placeholder="For example: a beginner core routine",
        key="plan-focus",
    )
    if len(focus.strip()) > 300:
        st.warning("Shorten your request to 300 characters before generating a plan.")
    with st.form("plan_form"):
        saved_profile = st.session_state.get("profile_form") or {}
        profile_days = [
            day for day in saved_profile.get("available_days", []) if day in DAYS
        ]
        default_workouts = min(3, max(1, len(profile_days)))
        workouts_per_week = st.number_input(
            "Workouts per week",
            min_value=1,
            max_value=max(1, len(profile_days)),
            value=min(default_workouts, max(1, len(profile_days))),
            step=1,
            help="Choose how many of your available days should have a workout.",
        )
        constraints = st.text_input(
            "Additional constraints (comma separated)",
            placeholder="For example: no jumping, minimal equipment",
        )
        refinement_instructions = st.text_area(
            "What should change from your previous plan? (optional)",
            max_chars=500,
            placeholder="For example: make it easier, avoid floor exercises, or add more variety",
            key="plan-refinement",
        )
        generate = st.form_submit_button(
            "Generate plan", type="primary", width="stretch"
        )
    if generate:
        sid = st.session_state.get("user_id", "")
        if not sid:
            st.error("Sign in or create an account first.")
        elif len(focus.strip()) < 3:
            st.error("Describe the plan focus in at least three characters.")
        elif len(focus.strip()) > 300:
            st.error("Shorten your request to 300 characters before generating a plan.")
        else:
            payload = {
                "focus": focus.strip(),
                "workouts_per_week": workouts_per_week,
                "extra_constraints": [
                    item.strip() for item in constraints.split(",") if item.strip()
                ],
                "refinement_instructions": refinement_instructions.strip() or None,
            }
            st.session_state.plan_generation_request = {
                "student_id": sid,
                "payload": payload,
                "focus": focus.strip(),
                "workouts_per_week": workouts_per_week,
            }
            with st.spinner("Generating your plan…"):
                save_generated_plan(sid, payload, focus.strip(), workouts_per_week)
    if st.session_state.get("plan_scope_message"):
        st.info(st.session_state.plan_scope_message)
    elif st.session_state.get("plan_generation_error"):
        st.error(st.session_state.plan_generation_error, title="Plan generation failed")
        if st.button("Try generating again", type="primary"):
            previous_request = st.session_state.get("plan_generation_request")
            if previous_request:
                with st.spinner("Trying again to generate your plan…"):
                    save_generated_plan(
                        previous_request["student_id"],
                        previous_request["payload"],
                        previous_request["focus"],
                        previous_request["workouts_per_week"],
                    )
            else:
                st.error(
                    "Submit your plan request again before retrying.",
                    title="Request unavailable",
                )
    if st.session_state.get("latest_plan"):
        show_plan(st.session_state.latest_plan, form_prefix="latest")

with history_tab:
    st.subheader("Your saved plans")
    st.write(
        "Every plan is stored with its request and sources. The **active** plan is the "
        "one you follow; **drafts** wait for your approval; **archived** plans were replaced."
    )
    notice = st.session_state.pop("plan_notice", None)
    if notice:
        st.success(notice)
    if st.button("Refresh plan history"):
        user_id = st.session_state.get("user_id")
        if not user_id:
            st.error("Sign in or create an account first.")
        else:
            try:
                logger.info("Plan history refresh requested")
                st.session_state.plan_history = api_request(
                    "GET", f"/api/v1/students/{user_id}/plans", params={"limit": 100}
                )
            except ApiError as error:
                st.error(str(error))
    records = st.session_state.get("plan_history", [])
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.info("Sign in to view your saved plans.")
    elif "plan_history" not in st.session_state:
        try:
            records = api_request(
                "GET", f"/api/v1/students/{user_id}/plans", params={"limit": 100}
            )
            st.session_state.plan_history = records
        except ApiError as error:
            records = []
            st.info(f"History will be available when the backend is connected. {error}")
    if records:
        for index, record in enumerate(records):
            plan_id = record.get("id")
            plan_title = (record.get("plan") or {}).get(
                "title", record.get("request", {}).get("focus", "Training plan")
            )
            with st.container(border=True):
                summary_column, action_column = st.columns(
                    [5, 1], vertical_alignment="center"
                )
                status_badge = (
                    f" {badge(record['status'])}" if record.get("status") else ""
                )
                summary_column.markdown(f"**{plan_title}**{status_badge}")
                summary_column.caption(
                    f"{record.get('request', {}).get('focus', 'Training plan')} · "
                    f"{record.get('created_at', '')[:10]}"
                )
                if plan_id:
                    if action_column.button(
                        "Delete plan",
                        key=f"delete-plan-{plan_id}",
                        icon=":material/delete:",
                    ):
                        st.session_state.pending_delete_plan_id = plan_id
                        st.rerun()
                    if st.session_state.get("pending_delete_plan_id") == plan_id:
                        st.warning(
                            f"Delete {plan_title}? This action cannot be undone.",
                            title="Confirm deletion",
                        )
                        confirm_column, cancel_column = st.columns(2)
                        if confirm_column.button(
                            "Confirm delete",
                            type="primary",
                            key=f"confirm-delete-{plan_id}",
                        ):
                            current_user_id = st.session_state.get("user_id")
                            if not current_user_id:
                                st.error("Sign in again before deleting a saved plan.")
                            else:
                                try:
                                    api_request(
                                        "DELETE",
                                        f"/api/v1/students/{current_user_id}/plans/{plan_id}",
                                    )
                                except ApiError as error:
                                    st.error(str(error))
                                else:
                                    st.session_state.plan_history = [
                                        saved
                                        for saved in st.session_state.get(
                                            "plan_history", []
                                        )
                                        if saved.get("id") != plan_id
                                    ]
                                    if (st.session_state.get("latest_plan") or {}).get(
                                        "id"
                                    ) == plan_id:
                                        st.session_state.pop("latest_plan", None)
                                    st.session_state.pop("pending_delete_plan_id", None)
                                    st.session_state.plan_notice = "Saved plan deleted."
                                    st.rerun()
                        if cancel_column.button(
                            "Keep plan", key=f"cancel-delete-{plan_id}"
                        ):
                            st.session_state.pop("pending_delete_plan_id", None)
                            st.rerun()
                else:
                    st.error("This saved plan has no ID and cannot be deleted.")
                with st.expander("View plan details", expanded=index == 0):
                    show_plan(record, form_prefix=f"history-{index}")
    elif not st.session_state.get("health_error"):
        st.info("No saved plans yet. Generate a plan to see it here.")

with agent_tab:
    st.subheader("🤖 Ask the RepWise assistant")
    st.write(
        "The orchestrator reads your profile and plan, plans which specialists to use, "
        "and combines their work. Changes to your plan or calendar are only **proposed**: "
        "approve them below or in the Approvals tab."
    )
    if not signed_in:
        st.info("Sign in to talk to the assistant.")
    else:
        show_action_notice()
        settings, examples = st.columns([1, 1])
        with settings.expander("Request settings", expanded=False):
            st.text_input("Time zone", value="Europe/Paris", key="agent-timezone")
            st.toggle(
                "Share a location for this request",
                key="agent-share-location",
                help="Sent with this request only and never stored. Use the Location tab to save one.",
            )
            if st.session_state.get("agent-share-location"):
                st.text_input(
                    "Place label", value="Current location", key="agent-location-label"
                )
                lat, lon = st.columns(2)
                lat.number_input(
                    "Latitude", value=48.7908, format="%.4f", key="agent-latitude"
                )
                lon.number_input(
                    "Longitude", value=2.3317, format="%.4f", key="agent-longitude"
                )
        with examples.expander(
            "Example requests (agent sheet and multi-agent)", expanded=False
        ):
            for number, example in enumerate(EXAMPLE_REQUESTS):
                st.button(
                    example,
                    key=f"agent-example-{number}",
                    on_click=use_example,
                    args=(example,),
                    width="stretch",
                )

        for number, message in enumerate(st.session_state.get("agent_messages", [])):
            with st.chat_message(
                message["role"], avatar="🧑" if message["role"] == "user" else "🏋️"
            ):
                st.markdown(message["content"])
                if message.get("response"):
                    show_agent_response(message["response"], key_prefix=f"msg{number}")

        # A plain text box, not a form: a form with clear_on_submit could hand
        # back the cleared box instead of the typed message. The box is
        # emptied in the button's callback, the only safe place to do it.
        with st.container(border=True):
            st.text_area(
                "Your message",
                key="agent-draft",
                max_chars=1000,
                placeholder="For example: Find a gym near me and create a training programme.",
            )
            ask, reset = st.columns([3, 1])
            ask.button(
                "Send to the agents",
                type="primary",
                width="stretch",
                on_click=take_agent_draft,
            )
            reset.button(
                "New conversation", width="stretch", on_click=start_new_conversation
            )
        if st.session_state.pop("agent_draft_empty", False):
            st.error("Write a message first.")
        queued = st.session_state.pop("agent_queued_message", None)
        if queued:
            send_agent_message(queued)
            st.rerun()

with approvals_tab:
    st.subheader("✅ Actions waiting for your approval")
    st.write(
        "Agents never run a consequential change by themselves. Each action below runs "
        "exactly as proposed, only once, and only if you approve it before it expires."
    )
    if not signed_in:
        st.info("Sign in to see your pending actions.")
    else:
        show_action_notice()
        refresh, history = st.columns(2)
        if refresh.button(
            "Load pending actions", width="stretch", icon=":material/refresh:"
        ):
            load_actions("pending")
        if history.button(
            "Show all decisions", width="stretch", icon=":material/history:"
        ):
            load_actions("all")
        actions = st.session_state.get("agent_actions")
        if actions is None:
            st.caption("Click **Load pending actions** to fetch them.")
        elif not actions:
            st.success("Nothing is waiting for your approval.")
        for action in actions or []:
            if action.get("status", "pending") == "pending":
                show_action(action, "inbox")
            else:
                with st.container(border=True):
                    st.markdown(
                        f"**{action['tool_name']}** · {badge(action['status'])}"
                    )
                    st.caption(
                        f"{action.get('reason', '')} · proposed by {action.get('proposed_by')}"
                    )
                    if action.get("result"):
                        st.json(action["result"], expanded=False)

with metrics_tab:
    st.subheader("📊 Session metrics")
    st.write(
        "Every assistant request and plan generation of this session, with its "
        "timing, tokens and outcome. Download it to compare models or settings later."
    )
    log = st.session_state.get("request_log", [])
    if not log:
        st.info(
            "Send a request to the assistant or generate a plan to collect metrics."
        )
    else:
        agent_rows = [
            row for row in log if row["kind"] == "agent" and "error" not in row
        ]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Requests", len(log))
        if agent_rows:
            times = sorted(row.get("server_s") or 0 for row in agent_rows)
            tokens_in = sum(row.get("input_tokens") or 0 for row in agent_rows)
            tokens_out = sum(row.get("output_tokens") or 0 for row in agent_rows)
            completed = sum(
                all(
                    status == "completed"
                    for status in row["step_statuses"].split(", ")
                    if status
                )
                for row in agent_rows
            )
            c2.metric("Median agent time", f"{times[len(times) // 2]:.1f} s")
            c3.metric("Tokens in / out", f"{tokens_in} / {tokens_out}")
            c4.metric("All steps completed", f"{completed}/{len(agent_rows)}")
        st.dataframe(log, hide_index=True, width="stretch")
        if agent_rows:
            st.markdown("**Agent request time (server, seconds)**")
            st.bar_chart(
                {
                    "seconds": {
                        f"{index + 1}. {row['request'][:40]}": row.get("server_s") or 0
                        for index, row in enumerate(agent_rows)
                    }
                },
                horizontal=True,
            )
        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "Download CSV", metrics_csv(log), "repwise-metrics.csv", "text/csv"
        )
        d2.download_button(
            "Download JSON",
            json.dumps(log, indent=2, ensure_ascii=False),
            "repwise-metrics.json",
            "application/json",
        )
        if d3.button("Clear metrics"):
            st.session_state.pop("request_log", None)
            st.rerun()

with location_tab:
    st.subheader("📍 Saved location")
    st.write(
        "Optional. Saving a location lets the Location Agent search near you without "
        "asking each time. It is stored only after you save it here, and you can delete it."
    )
    if not signed_in:
        st.info("Sign in to manage your saved location.")
    else:
        student_id = st.session_state.get("user_id")
        if st.button("Load saved location", icon=":material/my_location:"):
            try:
                st.session_state.saved_location = api_request(
                    "GET", f"/api/v1/students/{student_id}/location"
                )
            except ApiError as error:
                st.session_state.saved_location = None
                st.info(
                    "No location saved yet." if error.status_code == 404 else str(error)
                )
        saved = st.session_state.get("saved_location")
        if saved:
            st.success(
                f"Saved: **{saved['label']}** ({saved['latitude']:.4f}, {saved['longitude']:.4f}) "
                f"· consent {saved['consented_at'][:10]}"
            )
            st.map([{"lat": saved["latitude"], "lon": saved["longitude"]}], zoom=12)
        with st.form("location_form"):
            label = st.text_input(
                "Place name", placeholder="For example: Cachan, France"
            )
            use_coordinates = st.checkbox(
                "I'll enter coordinates instead of looking up the name"
            )
            lat, lon = st.columns(2)
            latitude = lat.number_input("Latitude", value=48.7908, format="%.4f")
            longitude = lon.number_input("Longitude", value=2.3317, format="%.4f")
            save_location = st.form_submit_button("Save this location", type="primary")
        if save_location:
            body: dict[str, Any] = {"label": label.strip() or "Home"}
            if use_coordinates:
                body.update(latitude=latitude, longitude=longitude)
            try:
                st.session_state.saved_location = api_request(
                    "PUT", f"/api/v1/students/{student_id}/location", json=body
                )
                st.success("Location saved.")
                st.rerun()
            except ApiError as error:
                st.error(str(error))
        if saved and st.button("Forget my saved location", icon=":material/delete:"):
            try:
                api_request("DELETE", f"/api/v1/students/{student_id}/location")
                st.session_state.pop("saved_location", None)
                st.success("Saved location deleted.")
                st.rerun()
            except ApiError as error:
                st.error(str(error))

    st.divider()
    st.subheader("⏰ Calendar")
    st.write(
        "Free/busy times and bookings use your Google Calendar once connected. Without "
        "it, the Schedule Agent says no calendar is connected and can still move "
        "workouts between your available days (with your approval)."
    )
    st.code(
        'python -m scripts.connect_google_calendar connect "Your full name"',
        language="powershell",
    )
    st.caption(
        "Run it once from the repository root; it opens Google's consent page in your browser."
    )

with system_tab:
    st.subheader("⚙️ System")
    health = st.session_state.get("health")
    if health:
        h1, h2, h3 = st.columns(3)
        h1.metric("Database", health.get("database", "?"))
        h2.metric(
            "Retrieval",
            f"{health.get('retrieval', '?')} ({health.get('retriever', '?')})",
        )
        h3.metric("Plan model", health.get("model", "?"))
        if health.get("retrieval_error"):
            st.error(health["retrieval_error"])
    else:
        st.warning("The API is not connected. Start it, then use **Check connection**.")

    st.markdown("#### Tool catalog (`config/tools.json`)")
    catalog = load_tool_catalog()
    if catalog:
        st.dataframe(
            [
                {
                    "Tool": tool["name"],
                    "Owner": AGENTS.get(tool["agents"][0], ("", tool["agents"][0]))[1],
                    "Also": ", ".join(
                        AGENTS.get(a, ("", a))[1] for a in tool["agents"][1:]
                    ),
                    "Access": tool["access"],
                    "Approval": tool["approval"],
                    "Services": ", ".join(tool.get("external_services", [])),
                    "Description": tool["description"],
                }
                for tool in catalog
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"{len(catalog)} tools. The agents and the MCP server read this same file."
        )
    else:
        st.info("config/tools.json could not be read.")

    st.markdown("#### MCP server")
    st.write(
        "`python -m src.mcp_server.server` exposes the same tools to an MCP host, bound to one account "
        "by its token. Generate a host configuration for the signed-in account:"
    )
    if signed_in and st.toggle("Show an MCP host configuration (contains your token)"):
        st.warning(
            "Keep this private: it contains your access token. Replace the secret placeholder."
        )
        st.code(
            json.dumps(
                {
                    "mcpServers": {
                        "repwise": {
                            "command": str(Path(sys.executable)),
                            "args": ["-m", "src.mcp_server.server"],
                            "cwd": str(PROJECT_ROOT),
                            "env": {
                                "REPWISE_AUTH_SECRET": "<the API's REPWISE_AUTH_SECRET>",
                                "REPWISE_MCP_ACCESS_TOKEN": st.session_state.get(
                                    "access_token", ""
                                ),
                                "PYTHONIOENCODING": "utf-8",
                            },
                        }
                    }
                },
                indent=2,
            ),
            language="json",
        )
    st.code(
        "python -m scripts.mcp_smoke   # self-check on a temporary account",
        language="powershell",
    )
