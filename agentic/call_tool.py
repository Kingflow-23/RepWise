"""Call any agent tool directly, without a model, to debug it.

It uses a temporary database with the synthetic account and a fake calendar, so your
real data is never touched. Approval is skipped, because this calls the tool itself.
From the repository root:

    python -m agentic.call_tool --list
    python -m agentic.call_tool get_timetable
    python -m agentic.call_tool propose_training_slots
    python -m agentic.call_tool find_nearby_places "{\"category\": \"gym\"}"

``--logged`` saves feedback on the first workout first, for the tools that need it.
Add ``--debug`` to see each tool's log lines.

After changing a tool, ``--write-schemas`` updates docs/agent/s5_agent_tools.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from functools import partial
from pathlib import Path

from agentic import tools
from agentic.agent_tools import AgentContext, build_plan_pipeline, build_tools
from agentic.calendar_tools import FakeCalendar
from agentic.mcp_demo_client import seed_demo_account
from agentic.places import Location
from config.config import AppConfig


SCHEMA_FILE = Path(__file__).resolve().parents[1] / "docs" / "agent" / "s5_agent_tools.json"


def agent_tool_schemas() -> dict:
    """Describe each tool exactly as the model sees it."""
    return {
        "tools": [
            {
                "name": item.name,
                "description": item.description,
                "parameters": item.tool_call_schema.model_json_schema(),
            }
            for item in build_tools(AgentContext(student_id="schema-only"))
        ]
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", nargs="?")
    parser.add_argument("arguments", nargs="?", default="{}", help="JSON arguments")
    parser.add_argument("--list", action="store_true", help="show tools and arguments")
    parser.add_argument("--logged", action="store_true", help="log the first workout")
    parser.add_argument("--debug", action="store_true", help="show log lines")
    parser.add_argument(
        "--write-schemas", action="store_true", help="update the published tool schemas"
    )
    args = parser.parse_args()
    if args.write_schemas:
        SCHEMA_FILE.write_text(
            json.dumps(agent_tool_schemas(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"Updated {SCHEMA_FILE}")
        return 0
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    config = AppConfig.from_environment()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        database = Path(directory) / "debug.sqlite3"
        os.environ["REPWISE_DATABASE_PATH"] = str(database)
        student_id, plan_id = seed_demo_account(database)
        if args.logged:
            tools.save_workout_feedback(
                student_id, plan_id, 1, 0, "completed", "slightly_too_hard"
            )
        available = {
            item.name: item
            for item in build_tools(
                AgentContext(
                    student_id=student_id,
                    location=Location(label="Cachan", latitude=48.7908, longitude=2.3317),
                    pipeline_factory=partial(build_plan_pipeline, config),
                    calendar=FakeCalendar(),
                )
            )
        }
        if args.list or args.tool is None:
            print(f"The demo plan's ID changes each run. This run: {plan_id}\n")
            for name, item in available.items():
                print(f"{name}({', '.join(item.tool_call_schema.model_fields)})")
            return 0
        if args.tool not in available:
            print(f"Unknown tool {args.tool!r}. Use --list to see the tools.")
            return 2
        arguments = json.loads(args.arguments)
        # The plan ID changes each run, so it can be left out of the arguments.
        if "plan_id" in available[args.tool].tool_call_schema.model_fields:
            arguments.setdefault("plan_id", plan_id)
        if args.tool == "generate_workout_plan":
            arguments.setdefault("focus", "Build leg muscles")
        output = available[args.tool].invoke(arguments)
        print(json.dumps(json.loads(output), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
