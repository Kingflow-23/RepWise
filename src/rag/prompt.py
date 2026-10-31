"""Prompt text for the first RepWise version."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas.models import PlanRequest, TrainingPlan

PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "system_prompt_v1.txt"


def load_system_prompt() -> str:
    """Read the versioned system prompt stored outside application code."""
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


def make_user_prompt(request: PlanRequest, evidence: str) -> str:
    """Keep request data and retrieved passages clearly separated."""
    return (
        "STUDENT REQUEST\n"
        + json.dumps(request.model_dump(), ensure_ascii=False, indent=2)
        + "\n\nPLAN LENGTH REQUIREMENT\n"
        + f"Return exactly {request.weeks} weeks, numbered consecutively from 1 to "
        + f"{request.weeks}. Include every week; do not return a sample or omit weeks.\n"
        + "Use only equipment listed in the student profile (bodyweight requires no equipment)."
        + "\n\nRETRIEVED SOURCES\n"
        + evidence
        + "\n\nRETURN THIS JSON SHAPE\n"
        + json.dumps(plan_response_schema(request), ensure_ascii=False)
    )


def plan_response_schema(request: PlanRequest) -> dict[str, Any]:
    """Constrain structured generation to the exact number of requested weeks."""
    schema = TrainingPlan.model_json_schema()
    schema["properties"]["weeks"]["minItems"] = request.weeks
    schema["properties"]["weeks"]["maxItems"] = request.weeks
    return schema
