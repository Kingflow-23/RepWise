"""Streamlit demo client for the RepWise HTTP API."""

from __future__ import annotations

import html
import os
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import requests
import streamlit as st

# Streamlit may execute this file with ``app/`` as the import root. Add the
# project root explicitly so local packages (config, src) resolve consistently
# whether launched with a relative path, an absolute path, or another cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.app_config import get_api_url
from src.observability import configure_logging

configure_logging("streamlit")
logger = logging.getLogger(__name__)
st.set_page_config(page_title="RepWise | Training plans", page_icon="🏋️", layout="wide")

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


class ApiError(RuntimeError):
    """A readable API failure for display in the demo."""

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
            f"Cannot reach the RepWise API at {API_BASE}. Start it with `uvicorn src.api.main:app --reload`. Details: {error}"
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
                detail["message"],
                status_code=response.status_code,
                detail=detail,
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
    try:
        result = api_request(
            "POST",
            f"/api/v1/students/{student_id}/plans",
            json=payload,
        )
    except ApiError as error:
        logger.warning(
            "Plan generation failed through Streamlit status=%s",
            error.status_code or "unavailable",
        )
        st.session_state.plan_generation_error = generation_error_message(error)
        return

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


def show_plan(record: dict[str, Any], *, form_prefix: str = "plan") -> None:
    """Render a saved plan, its supporting evidence, and its source details.

    Args:
        record: Plan history record containing the plan, request, and sources.
    """
    plan = record.get("plan") or {}
    options = record.get("request") or {}
    st.subheader(plan.get("title", "Training plan"))
    st.caption(
        f"{options.get('focus', 'Plan')} · {options.get('weeks', len(plan.get('weeks', [])))} weeks · Created {record.get('created_at', 'just now')}"
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
            for workout_index, workout in enumerate(week.get("workouts", [])):
                workout_key = (week["number"], workout_index)
                adapted_from = adapted_workouts.get(workout_key)
                adaptation_note = (
                    " · Adapted from your workout feedback" if adapted_from else ""
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
                        f"- **{exercise['name']}** · {exercise['sets']} × {exercise['repetitions']} · rest {exercise['rest_seconds']}s · {time_estimate}"
                    )
                    st.caption(exercise.get("coaching_note", ""))
                    for evidence in exercise.get("supporting_evidence", []):
                        st.caption(
                            f"Evidence `{evidence['source_id']}`: "
                            f"“{evidence['quote']}”"
                        )
                saved_feedback = feedback_by_workout.get(workout_key)
                if saved_feedback:
                    st.success("Workout logged")
                    st.caption(
                        f"{COMPLETION_LABELS.get(saved_feedback.get('completion_status'), saved_feedback.get('completion_status'))} · "
                        f"{DIFFICULTY_LABELS.get(saved_feedback.get('difficulty'), saved_feedback.get('difficulty'))}"
                    )
                    if saved_feedback.get("comment"):
                        st.write(saved_feedback["comment"])
                elif record.get("id") and st.session_state.get("user_id"):
                    with st.form(
                        f"log-workout-{form_prefix}-{record['id']}-"
                        f"{week['number']}-{workout_index}"
                    ):
                        completion_status = st.selectbox(
                            "Completion",
                            list(COMPLETION_LABELS),
                            format_func=COMPLETION_LABELS.get,
                        )
                        difficulty = st.selectbox(
                            "Overall difficulty",
                            list(DIFFICULTY_LABELS),
                            format_func=DIFFICULTY_LABELS.get,
                        )
                        comment = st.text_area(
                            "Comment (optional)",
                            max_chars=500,
                            key=(
                                f"workout-comment-{form_prefix}-{record['id']}-"
                                f"{week['number']}-{workout_index}"
                            ),
                        )
                        submitted = st.form_submit_button("Log workout")
                    if submitted:
                        student_id = st.session_state.get("user_id")
                        try:
                            result = api_request(
                                "POST",
                                f"/api/v1/students/{student_id}/plans/{record['id']}"
                                f"/weeks/{week['number']}/workouts/{workout_index}"
                                "/feedback",
                                json={
                                    "completion_status": completion_status,
                                    "difficulty": difficulty,
                                    "comment": comment.strip() or None,
                                },
                            )
                        except ApiError as error:
                            st.error(str(error))
                        else:
                            record["plan"] = result["plan"]
                            record["workout_feedback"] = result["workout_feedback"]
                            if result["adapted"]:
                                st.success(result["message"])
                            else:
                                st.info(result["message"])
                            st.rerun()
    if plan.get("assumptions"):
        st.markdown("**Assumptions**")
        for assumption in plan["assumptions"]:
            st.markdown(f"- {assumption}")
    if plan.get("safety_note"):
        st.info(plan["safety_note"])
    sources = record.get("sources", [])
    if sources:
        st.markdown("**Sources used**")
        for source in sources:
            st.markdown(
                f"- [{source['title']}]({source['url']}) · `{source['source_id']}`"
            )
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
            changed owner clears cached profile and plan state to prevent data
            from one signed-in account appearing under another.
    """
    if st.session_state.get("user_id") != account["user_id"]:
        for key in (
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
        ):
            st.session_state.pop(key, None)
    st.session_state.user_id = account["user_id"]
    st.session_state.full_name = account["full_name"]


st.markdown(
    "<style> .block-container{padding-top:2rem;max-width:1200px} [data-testid='stMetric']{background:#f4f7f5;padding:1rem;border-radius:12px} </style>",
    unsafe_allow_html=True,
)
st.title("🏋️ RepWise")
st.markdown("### Training plans grounded in your knowledge base")
st.write(
    "Build a plan around your schedule, experience, equipment, and current goal. Each completed plan keeps links to the passages used as evidence."
)

with st.sidebar:
    st.header("Demo connection")
    st.subheader("Account")
    account_name = st.text_input(
        "Full name", value=st.session_state.get("full_name", "")
    )
    account_password = st.text_input("Password", type="password")
    if st.session_state.get("full_name"):
        st.caption(f"Signed in as {st.session_state.full_name}")
    sign_in, register = st.columns(2)
    if sign_in.button("Sign in", use_container_width=True):
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
    if register.button("Create account", use_container_width=True):
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
    api_base = st.text_input(
        "Backend API",
        value=API_BASE,
        help="Set REPWISE_API_URL before launching Streamlit, or edit it here.",
    )
    if api_base.rstrip("/") != API_BASE:
        API_BASE = api_base.rstrip("/")
    if st.button("Check connection", use_container_width=True):
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
        "Plans are training suggestions, not medical advice. Stop if an exercise causes pain."
    )

profile_tab, plan_tab, history_tab = st.tabs(
    ["1 · Student profile", "2 · Generate a plan", "3 · Saved plans"]
)

with profile_tab:
    st.subheader("Your training profile")
    st.write("Save or update the profile the backend will use to personalize plans.")
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
        save = st.form_submit_button(
            "Save profile", type="primary", use_container_width=True
        )
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
                    "PUT",
                    f"/api/v1/students/{student_id}/profile",
                    json=payload,
                )
                logger.info("Profile saved through Streamlit")
                st.success("Profile saved to the backend database.")
            except ApiError as error:
                st.error(str(error))

with plan_tab:
    st.subheader("Create a personalized plan")
    st.write(
        "Each run creates one week of workouts using your available days. Generate again when you need a fresh week. Retrieved sources are shown for context."
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
        available_days = [
            day for day in saved_profile.get("available_days", []) if day in DAYS
        ]
        default_workouts = min(3, max(1, len(available_days)))
        workouts_per_week = st.number_input(
            "Workouts per week",
            min_value=1,
            max_value=max(1, len(available_days)),
            value=min(default_workouts, max(1, len(available_days))),
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
            "Generate plan", type="primary", use_container_width=True
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
        st.error(
            st.session_state.plan_generation_error,
            title="Plan generation failed",
        )
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
        "Completed plans are stored in SQLite with their request and source links."
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
                    "GET",
                    f"/api/v1/students/{user_id}/plans",
                    params={"limit": 100},
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
                "GET",
                f"/api/v1/students/{user_id}/plans",
                params={"limit": 100},
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
                summary_column.markdown(f"**{plan_title}**")
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
                            "Keep plan",
                            key=f"cancel-delete-{plan_id}",
                        ):
                            st.session_state.pop("pending_delete_plan_id", None)
                            st.rerun()
                else:
                    st.error("This saved plan has no ID and cannot be deleted.")
                with st.expander(
                    "View plan details",
                    expanded=index == 0,
                ):
                    show_plan(record, form_prefix=f"history-{index}")
    elif not st.session_state.get("health_error"):
        st.info("No saved plans yet. Generate a plan to see it here.")
