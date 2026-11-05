"""Session 5 deliverable: run the LangGraph agent on the 5 Agent Sheet requests.

Uses a temporary database with one synthetic account and a demo location. The
location is sent to OpenStreetMap services, and plan generation uses the local
corpus. Your real account and data are never touched.

Needs Ollama running. From the repository root:

    python -m agentic.run_agent            # asks before risky actions
    python -m agentic.run_agent --approve  # approves them automatically
    python -m agentic.run_agent --all-tools --approve  # each tool and memory, in sequence

Uses a fake calendar with synthetic busy times, so no real calendar is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime, time, timedelta
from functools import partial
from pathlib import Path
from zoneinfo import ZoneInfo

from agentic import tools
from agentic.agent import (
    MAX_SECONDS,
    AgentReply,
    ask,
    build_agent,
    build_chat_model,
)
from agentic.agent_tools import AgentContext, build_plan_pipeline
from agentic.calendar_tools import BusyPeriod, FakeCalendar, default_week_start
from agentic.mcp_demo_client import seed_demo_account
from agentic.places import Location
from config.config import AppConfig

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT = ROOT / "outputs" / "agent_runs" / "s5_five_requests.md"
ALL_TOOLS_TRANSCRIPT = ROOT / "outputs" / "agent_runs" / "s5_all_tools.md"
TIMEZONE = "Europe/Paris"
DEMO_LOCATION = Location(label="Cachan, France", latitude=48.7908, longitude=2.3317)
LEG_WORK = re.compile(
    r"\b(?:legs?|lower|squats?|lunges?|calf|calves|deadlifts?|jambes?|fentes?)\b",
    re.IGNORECASE,
)

# The Agent Sheet's requests, with spelling fixed and each "X" filled in.
CASES = [
    {
        "request": "Pourrais-tu me trouver la salle de sport la plus près de moi ?",
        "decision": "Agent",
        "tool": "find_nearby_places",
        "category": "gym",
        "must_name_nearest": True,
    },
    {
        "request": "Dans quel magasin proche de moi puis-je trouver des haltères ?",
        "decision": "Agent",
        "tool": "find_nearby_places",
        "category": "sports_store",
    },
    {
        "request": "Dans quel magasin proche de moi puis-je trouver de la créatine ?",
        "decision": "Agent",
        "tool": "find_nearby_places",
        "category": "supplement_store",
    },
    {
        "request": "Comment pourrais-je organiser mes séances de la semaine en "
        "fonction de mon programme ?",
        "decision": "Agent",
        "tool": "propose_training_slots",
    },
    {
        "request": "Propose-moi un programme pour prendre du muscle au niveau de "
        "mes jambes.",
        "decision": "Workflow",
        "tool": "generate_workout_plan",
    },
]


# Run in this order, on one account: later requests use what earlier ones saved.
# Requests sharing a "thread" are one conversation.
ALL_TOOLS_CASES = [
    {
        "request": "J'ai fini la séance de lundi : goblet squat, 3 séries de 10 "
        "répétitions avec 12 kg. Enregistre ces charges.",
        "tool": "log_workout_performance",
    },
    {
        "request": "La séance de lundi est terminée et elle était un peu trop dure. "
        "Enregistre-le, puis adapte la prochaine séance.",
        "tool": "update_upcoming_workout",
    },
    {
        "request": "Comment se passent mes entraînements jusqu'ici ?",
        "tool": "get_workout_history",
    },
    {
        "request": "Que disent tes sources sur le temps de repos entre les séries "
        "pour l'hypertrophie ?",
        "tool": "search_documents",
    },
    {
        "request": "Garde pour plus tard cette idée : ajouter une série de squats la "
        "semaine prochaine, pour progresser. Ne change pas mon programme.",
        "tool": "save_recommendation",
    },
    {
        "request": "Souviens-toi que je préfère m'entraîner le matin.",
        "tool": "remember_fact",
    },
    # A new conversation, so the answer can only come from semantic memory.
    {
        "request": "À quel moment de la journée est-ce que je préfère m'entraîner ?",
        "tool": None,
        "answer_contains": "matin",
    },
    {
        "request": "Suis-je occupé la semaine prochaine ?",
        "tool": "get_calendar_busy",
    },
    {
        "request": "Propose-moi des créneaux dans mon agenda pour mes séances.",
        "tool": "propose_training_slots",
        "thread": "booking",
    },
    {
        "request": "Réserve le premier créneau proposé pour chaque séance.",
        "tool": "add_workouts_to_calendar",
        "thread": "booking",
    },
    {
        "request": "Échange les jours de mes séances de lundi et de vendredi dans "
        "mon programme.",
        "tool": "organize_schedule",
    },
]


def demo_calendar(now: datetime, plan_days: list[str]) -> FakeCalendar:
    """Build synthetic busy times, so Wednesday is full and others partly busy."""
    zone = ZoneInfo(TIMEZONE)
    monday = default_week_start(plan_days, now.astimezone(zone).date())

    def busy(day_offset: int, start_hour: int, end_hour: int) -> BusyPeriod:
        day = monday + timedelta(days=day_offset)
        return BusyPeriod(
            start=datetime.combine(day, time(start_hour), zone),
            end=datetime.combine(day, time(end_hour), zone),
        )

    return FakeCalendar(
        [busy(0, 8, 18), busy(2, 6, 22), busy(4, 12, 14), busy(5, 9, 11)]
    )


def check_case(case: dict, reply: AgentReply) -> list[str]:
    """Return what went wrong in one case, or an empty list if it passed."""
    problems = []
    # Notes come from code and can repeat the plan, so they don't count as the answer.
    model_answer = reply.answer.split("**Note:**")[0].strip()
    if not model_answer:
        return ["the answer is empty"]
    if case["tool"] is None:
        expected = case["answer_contains"]
        return [] if expected in model_answer.lower() else [f'no "{expected}" in the answer']
    calls = [args for name, args in reply.tool_calls if name == case["tool"]]
    if not calls:
        return [f"did not call {case['tool']}"]
    if "category" in case and all(
        args.get("category") != case["category"] for args in calls
    ):
        problems.append(f"did not search category {case['category']}")
    if reply.seconds > MAX_SECONDS:
        problems.append(f"took {reply.seconds:.0f} s, over the {MAX_SECONDS} s limit")

    outputs = [
        json.loads(content)
        for name, content in reply.tool_results
        if name == case["tool"] and content.startswith("{")
    ]
    if outputs and not any(output.get("ok", True) for output in outputs):
        problems.append(f"the tool failed: {outputs[-1].get('code')}")
    if case["tool"] == "add_workouts_to_calendar" and not any(
        outcome["status"] == "booked"
        for output in outputs
        for outcome in output.get("outcomes", [])
    ):
        problems.append("no workout was booked")
    if case["tool"] == "find_nearby_places":
        names = [
            place["name"] for output in outputs for place in output.get("places", [])
        ]
        # An empty or failed search is an acceptable tool result, but any place
        # the agent found must appear in its answer.
        if names and not any(name in reply.answer for name in names):
            problems.append("the answer names none of the places found")
        if names and case.get("must_name_nearest") and names[0] not in reply.answer:
            problems.append(f"the answer does not name the nearest place: {names[0]}")
    if case["tool"] == "generate_workout_plan" and not any(
        output.get("status") == "completed" for output in outputs
    ):
        status = outputs[-1].get("status") if outputs else "did not run"
        problems.append(f"no plan was generated: {status}")
    for output in outputs:
        if case["tool"] == "generate_workout_plan" and output.get("plan"):
            workouts = [w for week in output["plan"]["weeks"] for w in week["workouts"]]
            leg_days = sum(bool(LEG_WORK.search(json.dumps(w))) for w in workouts)
            if leg_days * 2 < len(workouts):
                problems.append(f"only {leg_days}/{len(workouts)} workouts train legs")
    return problems


def prompt_for_approval(request: dict) -> bool:
    """Ask at the terminal. Without a terminal to answer, decline."""
    print(f"  Approval needed: {request['reason']}")
    try:
        return input("  Approve? [y/N] ").strip().casefold() in {"y", "yes", "o", "oui"}
    except EOFError:
        return False


def transcript_entry(number: int, case: dict, reply: AgentReply, problems: list) -> str:
    """Format one case for the saved transcript."""
    lines = [
        f"## {number}. {case['request']}",
        "",
        f"Sheet decision: {case.get('decision', 'Agent')}. "
        f"Time: {reply.seconds:.0f} s. "
        f"Result: **{'PASS' if not problems else 'FAIL: ' + '; '.join(problems)}**",
        "",
        "Plan:",
        "",
    ]
    lines += [f"{step}. {task}" for step, task in enumerate(reply.plan, start=1)]
    if reply.revisions:
        lines += ["", f"The plan was revised {reply.revisions} time(s) after a failed step."]
    lines += ["", "Tool calls:", ""]
    lines += [f"- `{name}({json.dumps(args)})`" for name, args in reply.tool_calls]
    lines += [
        f"- Approval requested: {reason} **{'Approved' if ok else 'Declined'}**"
        for reason, ok in reply.approvals
    ]
    for name, content in reply.tool_results:
        if name == "find_nearby_places" and content.startswith("{"):
            output = json.loads(content)
            found = ", ".join(
                f"{place['name']} ({place['distance_km']} km)"
                for place in output.get("places", [])
            )
            lines.append(f"- Places returned: {found or output.get('error') or 'none'}")
    lines += ["", "Answer:", "", reply.answer, ""]
    return "\n".join(lines)


def main() -> int:
    """Run all five requests on one temporary account and save a transcript."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approve",
        action="store_true",
        help="approve risky actions automatically, and say so in the transcript",
    )
    parser.add_argument(
        "--all-tools",
        action="store_true",
        help="exercise the history, adaptation, search, and calendar tools instead",
    )
    args = parser.parse_args()

    cases = ALL_TOOLS_CASES if args.all_tools else CASES
    transcript = ALL_TOOLS_TRANSCRIPT if args.all_tools else TRANSCRIPT
    config = AppConfig.from_environment()
    llm = build_chat_model(config)
    approve = (lambda _request: True) if args.approve else prompt_for_approval
    entries = []
    passed = 0
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        database = Path(directory) / "agent.sqlite3"
        os.environ["REPWISE_DATABASE_PATH"] = str(database)
        student_id, _plan_id = seed_demo_account(database)
        # Give the synthetic student a spare day, so a workout has somewhere to move.
        repository = tools.open_repository()
        profile = repository.get_profile(student_id)
        repository.save_profile(
            student_id,
            profile.model_copy(
                update={"available_days": [*profile.available_days, "Saturday"]}
            ),
        )
        plan_days = [
            workout.day for workout in tools.get_timetable(student_id).planned_workouts
        ]
        graph = build_agent(
            AgentContext(
                student_id=student_id,
                location=DEMO_LOCATION,
                pipeline_factory=partial(build_plan_pipeline, config),
                calendar=demo_calendar(datetime.now(UTC), plan_days),
                timezone=TIMEZONE,
            ),
            llm,
        )
        for number, case in enumerate(cases, start=1):
            print(f"\n[{number}/{len(cases)}] {case['request']}", flush=True)
            reply = ask(
                graph,
                case["request"],
                thread_id=case.get("thread", f"request-{number}"),
                approve=approve,
            )
            for name, arguments in reply.tool_calls:
                print(f"  called {name}({json.dumps(arguments)})")
            print(f"  answered in {reply.seconds:.0f} s:\n{reply.answer}")
            problems = check_case(case, reply)
            print("  PASS" if not problems else f"  FAIL: {'; '.join(problems)}")
            passed += not problems
            entries.append(transcript_entry(number, case, reply, problems))

    header = [
        "# Session 5: the LangGraph agent on "
        + ("each tool and memory" if args.all_tools else "the 5 requests"),
        "",
        f"Model: `{config.ollama_model}`. Passed: **{passed}/{len(cases)}**. "
        f"Account: synthetic. Calendar: fake, with synthetic busy times. "
        f"Location: {DEMO_LOCATION.label}. Approvals: "
        f"{'automatic (--approve)' if args.approve else 'answered at the terminal'}.",
        "",
        "Generated by `python -m agentic.run_agent`. PASS is a mechanical check of "
        "tool choice, time, and results, not a review of answer quality.",
        "",
    ]
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("\n".join(header + entries), encoding="utf-8")
    print(f"\n{passed}/{len(cases)} requests passed. Transcript: {transcript}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
