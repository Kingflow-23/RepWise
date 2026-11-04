"""FastMCP stdio server exposing the three RepWise tools.

Run from the repository root with ``python -m agentic.mcp_server``.
"""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from agentic import tools
from src.observability import configure_logging

StudentId = Annotated[
    str,
    Field(min_length=1, description="Account ID, as in the API's student routes."),
]

mcp = FastMCP("RepWise Agentic Tools")


@mcp.tool
def get_user_profile(student_id: StudentId) -> tools.ProfileResult | tools.ToolFailure:
    """Read a student's experience, goal, available_days, session_minutes, equipment, and preferences."""
    return tools.get_user_profile(student_id)


@mcp.tool
def get_timetable(student_id: StudentId) -> tools.TimetableResult | tools.ToolFailure:
    """Read a student's weekly availability and planned workouts, each with its plan_id, week_number, workout_index, and saved feedback."""
    return tools.get_timetable(student_id)


@mcp.tool
def save_workout_feedback(
    student_id: StudentId,
    plan_id: Annotated[
        str, Field(min_length=1, description="A plan_id returned by get_timetable.")
    ],
    week_number: Annotated[
        int, Field(ge=1, description="A week_number returned by get_timetable.")
    ],
    workout_index: Annotated[
        int,
        Field(
            ge=0, description="A zero-based workout_index returned by get_timetable."
        ),
    ],
    completion_status: Annotated[
        tools.CompletionStatus,
        Field(description="Whether the student finished the workout."),
    ],
    difficulty: Annotated[
        tools.WorkoutDifficulty,
        Field(description="How the workout felt to the student."),
    ],
    comment: Annotated[
        str | None,
        Field(max_length=500, description="Optional comment from the student."),
    ] = None,
) -> tools.FeedbackResult | tools.ToolFailure:
    """Save feedback for one workout. Call only when the student explicitly asks to log it."""
    return tools.save_workout_feedback(
        student_id,
        plan_id,
        week_number,
        workout_index,
        completion_status,
        difficulty,
        comment,
    )


if __name__ == "__main__":
    configure_logging("agentic")
    mcp.run(show_banner=False)
