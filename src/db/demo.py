"""Create a synthetic demo account, for tests, scripts and the MCP lab.

The data in ``demo_data.json`` describes no real person. Use this only on a
temporary database: it never touches the configured application database
unless you pass that path yourself.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.db.repository import StudentRepository
from src.schemas.models import PlanRequest, PlanResult, TrainingPlan, UserProfile

DEMO_DATA = Path(__file__).with_name("demo_data.json")
DEMO_NAME = "Demo User"


def seed_demo_account(
    database: Path | str, full_name: str = DEMO_NAME
) -> tuple[str, str]:
    """Create one synthetic account with a profile and an active plan.

    The account has an unusable password hash, so nobody can sign in to it.

    Args:
        database: SQLite file to create or extend.
        full_name: Account name (must be unique in the database).

    Returns:
        ``(student_id, plan_id)``.
    """
    data = json.loads(DEMO_DATA.read_text(encoding="utf-8"))
    profile = UserProfile.model_validate(data["profile"])
    repository = StudentRepository(database)
    repository.initialize()
    student_id = repository.create_user(full_name, "!synthetic-account-without-login")
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
