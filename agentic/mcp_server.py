"""FastMCP stdio server exposing the three RepWise tools.

Run from the repository root with ``python -m agentic.mcp_server``.
"""

import sys
from pathlib import Path
from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field

# Also allow MCP hosts to launch this file by path from another directory.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from agentic import tools
from src.observability import configure_logging

UserId = Annotated[
    str,
    Field(
        min_length=1,
        description="Account ID: the API's user_id / student_id, e.g. demo_user.",
    ),
]

mcp = FastMCP("RepWise Agentic Tools")


@mcp.tool
def get_user_profile(user_id: UserId) -> dict[str, object]:
    """Read a user's experience, goal, available_days, session_minutes, equipment, and preferences."""
    return tools.get_user_profile(user_id)


@mcp.tool
def get_timetable(user_id: UserId) -> dict[str, object]:
    """Read a user's weekly availability and planned workouts, each with its workout_id and saved feedback."""
    return tools.get_timetable(user_id)


@mcp.tool
def save_workout_feedback(
    user_id: UserId,
    workout_id: Annotated[
        str,
        Field(min_length=1, description="A workout_id returned by get_timetable."),
    ],
    difficulty: Annotated[
        Literal["easy", "moderate", "hard"],
        Field(
            description=(
                "How the workout felt. Stored as the API values slightly_too_easy, "
                "just_right, and slightly_too_hard respectively."
            )
        ),
    ],
    notes: Annotated[
        str,
        Field(
            max_length=tools.MAX_NOTES_LENGTH,
            description="Optional user comment, stored as the API's comment field.",
        ),
    ] = "",
) -> dict[str, object]:
    """Save feedback for a completed workout. Call only when the user explicitly asks to log it."""
    return tools.save_workout_feedback(user_id, workout_id, difficulty, notes)


if __name__ == "__main__":
    configure_logging("agentic")
    mcp.run(show_banner=False)
