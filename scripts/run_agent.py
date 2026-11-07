"""Run the multi-agent assistant on the agent sheet's requests, with the real model.

It uses a temporary database with the synthetic account, a fake calendar with
synthetic busy times, and a demo location sent to the Photon map service. Plan
generation and document search use the local corpus; every model call goes to
the local Ollama. Real accounts and calendars are never touched.

From the repository root (Ollama running, REPWISE_CHUNK_FILE set):

    python -m scripts.run_agent              # the agent sheet's 5 requests
    python -m scripts.run_agent --multi      # the multi-agent scenarios
    python -m scripts.run_agent --approve    # also approve every pending action

Each case checks mechanically which specialists ran and which tools were
called; it is not a review of answer quality. A transcript is written to
``outputs/agent_runs/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from config.config import REPOSITORY_ROOT, AppConfig
from scripts.demo_context import demo_context
from src.agents import approvals
from src.agents.orchestrator import AgentResponse, build_chat_model, build_orchestrator

OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "agent_runs"

# The agent sheet's five initial requests, with each "X" filled in.
SHEET_CASES = [
    {
        "request": "Pourrais-tu me trouver la salle de sport la plus près de moi ?",
        "agents": {"location"},
        "tools": {"search_nearby_places"},
    },
    {
        "request": "Dans quel magasin proche de moi puis-je trouver des haltères ?",
        "agents": {"location"},
        "tools": {"search_nearby_places"},
    },
    {
        "request": "Dans quel magasin proche de moi puis-je trouver de la créatine ?",
        "agents": {"location"},
        "tools": {"search_nearby_places"},
    },
    {
        "request": "Comment pourrais-je organiser mes séances de la semaine en fonction de mon programme ?",
        "agents": {"schedule"},
        "tools": {"propose_training_slots"},
    },
    {
        "request": "Propose-moi un programme pour prendre du muscle au niveau de mes jambes.",
        "agents": {"planning"},
        "tools": {"generate_workout_plan"},
    },
]

# Requests that need several specialists, run in order on one account.
MULTI_CASES = [
    {
        "request": "Find a gym near me and create a training programme to build muscle.",
        "agents": {"location", "planning"},
        "tools": {"search_nearby_places", "generate_workout_plan"},
    },
    {
        "request": "I missed my Monday and Wednesday workouts this week. Record them and organise the rest of my week.",
        "agents": {"performance", "schedule"},
        "tools": {"save_workout_feedback"},
    },
    {
        "request": "How much rest should I take between sets for hypertrophy, according to your documents?",
        "agents": {"knowledge"},
        "tools": {"search_documents"},
    },
]


def check(case: dict, response: AgentResponse) -> list[str]:
    """List what went wrong in one case.

    Args:
        case: Expected specialists and tools.
        response: The orchestrator's answer.

    Returns:
        Problems; empty when the case passed.
    """
    problems = []
    if not response.answer.split("**Note:**")[0].strip():
        problems.append("the answer is empty")
    ran = {report.agent for report in response.reports}
    missing_agents = case["agents"] - ran
    if missing_agents:
        problems.append(f"specialists not used: {', '.join(sorted(missing_agents))}")
    called = {item["name"] for report in response.reports for item in report.tools_used}
    missing_tools = case["tools"] - called
    if missing_tools:
        problems.append(f"tools not called: {', '.join(sorted(missing_tools))}")
    return problems


def transcript_entry(
    number: int, case: dict, response: AgentResponse, problems: list[str]
) -> str:
    """Format one case for the transcript.

    Args:
        number: Case number.
        case: The case.
        response: The answer.
        problems: Result of :func:`check`.

    Returns:
        Markdown text.
    """
    lines = [
        f"## {number}. {case['request']}",
        "",
        f"Time: {response.seconds:.0f} s. Replans: {response.replans}. "
        f"Result: **{'PASS' if not problems else 'FAIL: ' + '; '.join(problems)}**",
        "",
        "Plan:",
        "",
    ]
    lines += [
        f"{i}. [{step.agent}] {step.task} -> {step.status}"
        for i, step in enumerate(response.steps, 1)
    ]
    lines += ["", "Tool calls:", ""]
    for report in response.reports:
        lines += [
            f"- {report.agent}: `{item['name']}` {'ok' if item['ok'] else 'failed: ' + str(item['code'])}"
            for item in report.tools_used
        ]
    lines += ["", "Answer:", "", response.answer, ""]
    return "\n".join(lines)


def main() -> int:
    """Run the cases and save a transcript.

    Returns:
        0 when every case passed, 1 otherwise.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--multi", action="store_true", help="run the multi-agent scenarios"
    )
    parser.add_argument(
        "--approve", action="store_true", help="approve every pending action"
    )
    args = parser.parse_args()

    cases = MULTI_CASES if args.multi else SHEET_CASES
    config = AppConfig.from_environment()
    llm = build_chat_model(config)
    entries, passed = [], 0
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        context, _plan_id = demo_context(config, Path(directory) / "agent.sqlite3")
        orchestrator = build_orchestrator(context, llm)
        for number, case in enumerate(cases, start=1):
            print(f"\n[{number}/{len(cases)}] {case['request']}", flush=True)
            response = orchestrator.ask(case["request"], f"case-{number}")
            for step in response.steps:
                print(f"  [{step.agent}] {step.task} -> {step.status}")
            print(f"  answered in {response.seconds:.0f} s:\n{response.answer}")
            if args.approve:
                for action in response.pending_actions:
                    decision = approvals.approve(context, action.action_id)
                    print(f"  approved {action.tool_name}: {decision.action.status}")
            problems = check(case, response)
            print("  PASS" if not problems else f"  FAIL: {'; '.join(problems)}")
            passed += not problems
            entries.append(transcript_entry(number, case, response, problems))

    name = "multi_agent_scenarios.md" if args.multi else "agent_sheet_requests.md"
    header = [
        f"# RepWise multi-agent run: {name.removesuffix('.md').replace('_', ' ')}",
        "",
        f"Model: `{config.agent_model}`. Passed: **{passed}/{len(cases)}**. Account: "
        "synthetic. Calendar: fake, with synthetic busy times. Approvals: "
        f"{'approved automatically' if args.approve else 'left pending'}.",
        "",
        "Generated by `python -m scripts.run_agent`. PASS checks which specialists and "
        "tools ran, not answer quality.",
        "",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / name).write_text("\n".join(header + entries), encoding="utf-8")
    print(f"\n{passed}/{len(cases)} cases passed. Transcript: {OUTPUT_DIR / name}")
    print(json.dumps({"passed": passed, "cases": len(cases)}))
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
