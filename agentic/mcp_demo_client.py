"""Demonstrate the three tools over MCP, without an LLM, an API key, or real data.

Run from the repository root with ``python -m agentic.mcp_demo_client``. It uses a
temporary database holding one synthetic account.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError

from src.db.repository import StudentRepository
from src.schemas.models import PlanRequest, PlanResult, TrainingPlan, UserProfile

ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA = Path(__file__).with_name("data") / "demo_data.json"


def seed_demo_account(database: Path) -> tuple[str, str]:
    """Create the synthetic account and plan, and return their IDs."""
    data = json.loads(DEMO_DATA.read_text(encoding="utf-8"))
    profile = UserProfile.model_validate(data["profile"])
    repository = StudentRepository(database)
    repository.initialize()
    student_id = repository.create_user("Demo User", "!synthetic-account-without-login")
    repository.save_profile(student_id, profile)
    plan_id = repository.save_plan(
        student_id,
        PlanRequest(profile=profile, **data["plan_options"]),
        PlanResult(
            status="completed",
            message="Synthetic demo plan.",
            plan=TrainingPlan.model_validate(data["plan"]),
        ),
    )
    return student_id, plan_id


def start_server(database: Path) -> Client:
    """Create a client that launches ``agentic.mcp_server`` on this database."""
    return Client(
        StdioTransport(
            command=sys.executable,
            args=["-m", "agentic.mcp_server"],
            cwd=str(ROOT),
            env={**os.environ, "REPWISE_DATABASE_PATH": str(database)},
        )
    )


async def call(client: Client, tool: str, **arguments: object) -> dict:
    """Call a tool, print its output, and return it as a dictionary."""
    output = (await client.call_tool(tool, arguments)).structured_content["result"]
    print(f"\n--- {tool}\n{json.dumps(output, indent=2)}")
    return output


async def run_demo(database: Path) -> None:
    """Discover and call the tools, then reload the feedback in a new server."""
    student_id, plan_id = seed_demo_account(database)
    workout = {"plan_id": plan_id, "week_number": 1, "workout_index": 0}
    feedback = {
        "completion_status": "completed",
        "difficulty": "slightly_too_hard",
        "comment": "Last set was difficult",
    }

    async with start_server(database) as client:
        print("Tools found:", [tool.name for tool in await client.list_tools()])
        await call(client, "get_user_profile", student_id=student_id)
        await call(client, "get_timetable", student_id=student_id)
        saved = await call(
            client,
            "save_workout_feedback",
            student_id=student_id,
            **workout,
            **feedback,
        )

        print("\nThese must be rejected:")
        await call(client, "get_user_profile", student_id="unknown")
        await call(
            client,
            "save_workout_feedback",
            student_id=student_id,
            **{**workout, "workout_index": 9},
            **feedback,
        )
        try:
            await client.call_tool(
                "save_workout_feedback",
                {"student_id": student_id, **workout, **feedback, "difficulty": "hard"},
            )
            raise RuntimeError("An invalid difficulty was accepted.")
        except ToolError:
            print("\n--- save_workout_feedback\nInvalid difficulty rejected.")

    async with start_server(database) as client:
        timetable = (
            await client.call_tool("get_timetable", {"student_id": student_id})
        ).structured_content["result"]
    if timetable["planned_workouts"][0]["feedback"] != saved["feedback"]:
        raise RuntimeError("The saved feedback did not persist.")
    print("\nA new server process reloaded the saved feedback.")
    print("Demonstration passed.")


def main() -> None:
    """Run the demonstration on a temporary database."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        asyncio.run(run_demo(Path(directory) / "demo.sqlite3"))


if __name__ == "__main__":
    main()
