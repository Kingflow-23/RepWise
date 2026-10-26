"""Prompt text for the first RepWise version."""

from __future__ import annotations

import json
from pathlib import Path

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
        + "\n\nRETRIEVED SOURCES\n"
        + evidence
        + "\n\nRETURN THIS JSON SHAPE\n"
        + json.dumps(TrainingPlan.model_json_schema(), ensure_ascii=False)
    )
