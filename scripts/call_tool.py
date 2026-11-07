"""Call one agent tool directly, without a language model, to see what it returns.

Runs on a temporary database with the synthetic account and a fake calendar,
so real data is never touched. The approval gate still applies: a tool that
needs approval returns ``pending_approval``; add ``--approve`` to approve it.
From the repository root:

    python -m scripts.call_tool --list
    python -m scripts.call_tool get_timetable
    python -m scripts.call_tool search_nearby_places "{\\"category\\": \\"gym\\"}"
    python -m scripts.call_tool organize_schedule "{\\"moves\\": [{\\"week_number\\": 1, \\"workout_index\\": 0, \\"new_day\\": \\"Saturday\\"}]}" --approve
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from config.config import AppConfig
from scripts.demo_context import demo_context
from src.agents import approvals
from src.tools.catalog import load_catalog


def main() -> int:
    """Run the command line.

    Returns:
        0 on success, 2 for an unknown tool.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tool", nargs="?")
    parser.add_argument("arguments", nargs="?", default="{}", help="JSON arguments")
    parser.add_argument("--list", action="store_true", help="show the tools")
    parser.add_argument("--approve", action="store_true", help="approve a held call")
    args = parser.parse_args()

    catalog = {tool.name: tool for tool in load_catalog().tools}
    if args.list or args.tool is None:
        for tool in catalog.values():
            print(
                f"{tool.name} [{tool.owner}, {tool.access}, approval: {tool.approval}]"
            )
            print(f"    {tool.description}")
        return 0
    if args.tool not in catalog:
        print(f"Unknown tool {args.tool!r}. Use --list to see the tools.")
        return 2

    config = AppConfig.from_environment()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        context, plan_id = demo_context(config, Path(directory) / "debug.sqlite3")
        print(f"Demo plan_id for this run: {plan_id}")
        run = approvals.run_tool_with_approval(
            context, args.tool, json.loads(args.arguments), "call_tool"
        )
        print(json.dumps(json.loads(run.as_json()), indent=2, ensure_ascii=False))
        if args.approve:
            for action in run.pending:
                decision = approvals.approve(context, action.action_id)
                print(f"\nApproved {action.tool_name}:")
                print(
                    json.dumps(
                        decision.model_dump(mode="json"), indent=2, ensure_ascii=False
                    )
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
