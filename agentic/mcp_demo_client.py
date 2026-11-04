"""Demonstrate the three tools over real MCP stdio, without an LLM or API key.

Run from the repository root with ``python -m agentic.mcp_demo_client``.

Feedback goes to a temporary directory unless ``REPWISE_DATA_DIR`` is set, so
the demonstration never writes to RepWise's data.
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLS = {"get_user_profile", "get_timetable", "save_workout_feedback"}


def server_client(data_dir: str) -> Client:
    """Create a client that launches its own ``agentic.mcp_server`` process."""
    return Client(
        StdioTransport(
            command=sys.executable,
            args=["-m", "agentic.mcp_server"],
            cwd=str(REPOSITORY_ROOT),
            env={**os.environ, "REPWISE_DATA_DIR": data_dir},
        )
    )


def show(title: str, value: object) -> None:
    """Print one labeled result as readable JSON."""
    print(f"\n--- {title}\n{json.dumps(value, indent=2, ensure_ascii=False)}")


def require(condition: bool, message: str) -> None:
    """Stop the demonstration if a result is not what the tools promise."""
    if not condition:
        raise RuntimeError(f"Demonstration failed: {message}")


async def run_demo(data_dir: str) -> None:
    """Discover the tools, call each one, and prove that feedback persists."""
    print(f"Feedback directory: {data_dir}")
    async with server_client(data_dir) as client:
        discovered = await client.list_tools()
        require(
            {tool.name for tool in discovered} == EXPECTED_TOOLS,
            "the server must expose exactly the three tools",
        )
        show(
            "Discovered tool schemas",
            {tool.name: tool.input_schema for tool in discovered},
        )

        profile = (
            await client.call_tool("get_user_profile", {"user_id": "demo_user"})
        ).data
        show("get_user_profile(demo_user)", profile)
        require(profile["ok"] is True, "the demo profile must load")

        timetable = (
            await client.call_tool("get_timetable", {"user_id": "demo_user"})
        ).data
        show(
            "get_timetable(demo_user), without exercise details",
            {
                **timetable,
                "planned_workouts": [
                    {key: value for key, value in item.items() if key != "exercises"}
                    for item in timetable["planned_workouts"]
                ],
            },
        )
        require(timetable["ok"] is True, "the demo timetable must load")

        saved = (
            await client.call_tool(
                "save_workout_feedback",
                {
                    "user_id": "demo_user",
                    "workout_id": "leg_day",
                    "difficulty": "hard",
                    "notes": "Last set was difficult",
                },
            )
        ).data
        show("save_workout_feedback(demo_user, leg_day, hard, ...)", saved)
        require(saved["saved"] is True, "the feedback must be saved")

        rejected = {
            "unknown user": ("get_user_profile", {"user_id": "nobody"}),
            "unknown workout": (
                "save_workout_feedback",
                {"user_id": "demo_user", "workout_id": "nope", "difficulty": "hard"},
            ),
        }
        for label, (name, arguments) in rejected.items():
            result = (await client.call_tool(name, arguments)).data
            show(f"Rejected: {label}", result)
            require(result["ok"] is False, f"{label} must be rejected")

        try:
            await client.call_tool(
                "save_workout_feedback",
                {
                    "user_id": "demo_user",
                    "workout_id": "leg_day",
                    "difficulty": "impossible",
                },
            )
        except ToolError as error:
            print(f"\n--- Rejected: invalid difficulty\n{str(error).splitlines()[0]}")
        else:
            raise RuntimeError("Demonstration failed: invalid difficulty was accepted")

    # A second server process can only see this record if it reached the disk.
    async with server_client(data_dir) as client:
        reloaded = (
            await client.call_tool("get_timetable", {"user_id": "demo_user"})
        ).data
    stored = next(
        item["feedback"]
        for item in reloaded["planned_workouts"]
        if item["workout_id"] == "leg_day"
    )
    show("Reloaded by a new server process", stored)
    require(stored == saved["feedback"], "the reloaded record must match the save")
    print("\nDemonstration passed.")


def main() -> None:
    """Run in ``REPWISE_DATA_DIR``, or in a temporary directory by default."""
    configured = os.getenv("REPWISE_DATA_DIR")
    if configured:
        asyncio.run(run_demo(configured))
        return
    with tempfile.TemporaryDirectory(
        prefix="repwise-demo-", ignore_cleanup_errors=True
    ) as data_dir:
        asyncio.run(run_demo(data_dir))


if __name__ == "__main__":
    main()
