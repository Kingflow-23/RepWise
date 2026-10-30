"""Streamlit demo client for the RepWise HTTP API."""

from __future__ import annotations

import os
import sys
from pathlib import Path
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

st.set_page_config(page_title="RepWise | Training plans", page_icon="🏋️", layout="wide")

API_BASE = get_api_url()
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class ApiError(RuntimeError):
    """A readable API failure for display in the demo."""


def api_request(method: str, path: str, **kwargs: Any) -> Any:
    # The backend derives ownership from this signed token, not a client-sent ID.
    token = st.session_state.get("access_token")
    if token:
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {token}"
        kwargs["headers"] = headers
    try:
        response = requests.request(method, f"{API_BASE}{path}", timeout=130, **kwargs)
    except requests.RequestException as error:
        raise ApiError(
            f"Cannot reach the RepWise API at {API_BASE}. Start it with `uvicorn src.api.main:app --reload`. Details: {error}"
        ) from error
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"API returned {response.status_code}: {detail}")
    return response.json() if response.content else None


def show_plan(record: dict[str, Any]) -> None:
    plan = record.get("plan") or {}
    options = record.get("request") or {}
    st.subheader(plan.get("title", "Training plan"))
    st.caption(
        f"{options.get('focus', 'Plan')} · {options.get('weeks', len(plan.get('weeks', [])))} weeks · Created {record.get('created_at', 'just now')}"
    )
    st.write(plan.get("summary", ""))
    for week in plan.get("weeks", []):
        with st.expander(
            f"Week {week['number']} — {week['objective']}",
            expanded=len(plan.get("weeks", [])) == 1,
        ):
            for workout in week.get("workouts", []):
                st.markdown(
                    f"**{workout['day']} · {workout['focus']}** — about {workout['estimated_minutes']} min"
                )
                for exercise in workout.get("exercises", []):
                    st.markdown(
                        f"- **{exercise['name']}** · {exercise['sets']} × {exercise['repetitions']} · rest {exercise['rest_seconds']}s"
                    )
                    st.caption(exercise.get("coaching_note", ""))
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
            if source.get("excerpt"):
                st.caption(source["excerpt"])


def store_account(account: dict[str, Any]) -> None:
    """Keep the internal owner ID in session state without asking the user to enter it."""
    if st.session_state.get("user_id") != account["user_id"]:
        for key in ("profile_form", "plan_history", "latest_plan"):
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
            f"Retrieval: {health.get('retrieval', 'unknown')} · Model: {health.get('model', 'unknown')}"
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
                saved = api_request(
                    "GET", f"/api/v1/students/{student_id}/profile"
                )
                st.session_state.profile_form = saved
                st.success("Profile loaded.")
            except ApiError as error:
                st.error(str(error))
    profile = st.session_state.get("profile_form", {})
    with st.form("profile_form"):
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
                st.success("Profile saved to the backend database.")
            except ApiError as error:
                st.error(str(error))

with plan_tab:
    st.subheader("Create a personalized plan")
    st.write(
        "The backend retrieves evidence, builds the prompt, calls Ollama, validates the response, and stores completed plans."
    )
    with st.form("plan_form"):
        focus = st.text_input(
            "What do you want to focus on?",
            placeholder="For example: a beginner core routine",
        )
        weeks = st.slider("Plan length", 1, 12, 4)
        constraints = st.text_input(
            "Additional constraints (comma separated)",
            placeholder="For example: no jumping, minimal equipment",
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
        else:
            with st.spinner(
                "Retrieving evidence and generating your plan with the backend…"
            ):
                try:
                    result = api_request(
                        "POST",
                        f"/api/v1/students/{sid}/plans",
                        json={
                            "focus": focus,
                            "weeks": weeks,
                            "extra_constraints": [
                                item.strip()
                                for item in constraints.split(",")
                                if item.strip()
                            ],
                        },
                    )
                    st.session_state.latest_plan = {
                        "id": result.get("plan_id"),
                        "plan": result.get("plan"),
                        "sources": result.get("sources", []),
                        "request": {"focus": focus, "weeks": weeks},
                    }
                    st.success("Plan generated, validated, and saved.")
                except ApiError as error:
                    st.error(str(error))
    if st.session_state.get("latest_plan"):
        show_plan(st.session_state.latest_plan)

with history_tab:
    st.subheader("Your saved plans")
    st.write(
        "Completed plans are stored in SQLite with their request and source links."
    )
    if st.button("Refresh plan history"):
        user_id = st.session_state.get("user_id")
        if not user_id:
            st.error("Sign in or create an account first.")
        else:
            try:
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
            with st.expander(
                f"{record.get('request', {}).get('focus', 'Training plan')} · {record.get('created_at', '')[:10]}",
                expanded=index == 0,
            ):
                show_plan(record)
    elif not st.session_state.get("health_error"):
        st.info("No saved plans yet. Generate a plan to see it here.")
