"""Start ``server.py`` over stdio and check it end to end, without a model.

It creates a temporary database with the synthetic account, signs a token for
it with a throwaway secret (as the API's login would), launches the MCP server
as a subprocess, and checks that:

- every tool of ``config/tools.json`` is listed, and no schema has a ``student_id``;
- a read tool returns the account's data;
- a consequential write is held as ``pending_approval``;
- an invalid call returns a structured error;
- a wrong token makes the server refuse to start.

From the repository root: ``python -m scripts.mcp_smoke`` (exit code 0 = passed).
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from config.config import REPOSITORY_ROOT
from src.api.auth import create_access_token
from src.db.demo import seed_demo_account
from src.tools.catalog import load_catalog


def server_environment(folder: Path) -> dict[str, str]:
    """Seed a demo account and build the server's environment.

    Args:
        folder: Temporary folder for the database and logs.

    Returns:
        Environment variables for ``server.py``.
    """
    database = folder / "mcp.sqlite3"
    student_id, _plan_id = seed_demo_account(database)
    secret = secrets.token_hex(32)
    return {
        **os.environ,
        "REPWISE_DATABASE_PATH": str(database),
        "REPWISE_AUTH_SECRET": secret,
        "REPWISE_MCP_ACCESS_TOKEN": create_access_token(student_id, secret),
        "REPWISE_LOG_DIR": str(folder / "logs"),
        "PYTHONIOENCODING": "utf-8",
    }


async def check_server(environment: dict[str, str]) -> list[str]:
    """Talk to a running server and collect problems.

    Args:
        environment: Variables from :func:`server_environment`.

    Returns:
        Problems; empty when every check passed.
    """
    problems = []
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "src.mcp_server.server"],
        cwd=str(REPOSITORY_ROOT),
        env=environment,
    )
    async with Client(transport) as client:
        tools = await client.list_tools()
        listed = {tool.name for tool in tools}
        expected = {tool.name for tool in load_catalog().tools}
        print(f"Tools listed: {len(listed)}")
        if listed != expected:
            problems.append(
                f"tools differ from the catalog: {sorted(listed ^ expected)}"
            )
        if any("student_id" in json.dumps(tool.input_schema) for tool in tools):
            problems.append("a tool schema accepts student_id")

        profile = (await client.call_tool("get_user_profile", {})).structured_content
        print("get_user_profile ->", profile.get("profile", {}).get("goal"))
        if not profile.get("ok"):
            problems.append("get_user_profile failed")

        held = (
            await client.call_tool(
                "organize_schedule",
                {
                    "moves": [
                        {"week_number": 1, "workout_index": 0, "new_day": "Friday"}
                    ]
                },
            )
        ).structured_content
        print("organize_schedule ->", held.get("status"))
        if held.get("status") != "pending_approval":
            problems.append("organize_schedule ran without approval")

        missing = await client.call_tool(
            "get_workout", {"week_number": 9, "workout_index": 0}, raise_on_error=False
        )
        print("get_workout(week 9) ->", missing.structured_content.get("code"))
        if (
            not missing.is_error
            or missing.structured_content.get("code") != "workout_not_found"
        ):
            problems.append("an unknown workout did not return workout_not_found")
    return problems


def main() -> int:
    """Run every check.

    Returns:
        0 when they all passed, 1 otherwise.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        environment = server_environment(Path(directory))
        problems = asyncio.run(check_server(environment))
        refused = subprocess.run(
            [sys.executable, "-m", "src.mcp_server.server"],
            cwd=REPOSITORY_ROOT,
            env={**environment, "REPWISE_MCP_ACCESS_TOKEN": "not-a-token"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        print("Wrong token -> exit code", refused.returncode)
        # Exit code 1 with the refusal message: the server ran and refused,
        # rather than failing to start for another reason.
        if refused.returncode != 1 or "invalid or expired" not in refused.stderr:
            problems.append("the server did not refuse the wrong token as expected")
    for problem in problems:
        print(f"FAIL: {problem}")
    print("MCP smoke test passed." if not problems else "MCP smoke test failed.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
