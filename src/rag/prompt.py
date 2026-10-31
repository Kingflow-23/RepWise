"""Prompt text for the first RepWise version."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas.models import PlanRequest, TrainingPlan

PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "system_prompt_v1.txt"


EVIDENCE_POLICY_MARKER = "{{EVIDENCE_POLICY}}"
RAG_EVIDENCE_POLICY = """EVIDENCE POLICY (RAG MODE)
- Use only the retrieved sources as factual support for exercise choices and specific prescriptions. Do not invent study findings, source contents, or citations.
- Every exercise must cite at least one source ID supplied in the evidence block. Copy the ID exactly.
- For every cited source, include one exact, contiguous supporting quote in ``supporting_evidence``. The quote must support that specific exercise or prescription, not merely mention the same broad topic.
- ``source_ids`` must contain exactly the source IDs represented in ``supporting_evidence``. Never cite a source that was not supplied.
- If retrieved evidence does not support a requested detail, do not invent that detail. Omit it, state a reasonable labeled assumption when appropriate, or explain the evidence limitation in the plan.
"""
NO_RAG_EVIDENCE_POLICY = """EVIDENCE POLICY (NO-RAG BASELINE)
- No reference sources are supplied in this comparison. Do not claim that a recommendation comes from research or retrieved evidence.
- Leave every exercise's ``source_ids`` and ``supporting_evidence`` arrays empty. Do not invent citations or quotes.
- Use general fitness knowledge to answer the request, while following all system rules, profile limits, and safety requirements.
"""


def load_system_prompt(*, require_evidence: bool = True) -> str:
    """Load the shared system prompt with the appropriate evidence policy.

    Args:
        require_evidence: Select the RAG policy, which requires source-backed
            exercise prescriptions, or the no-RAG evaluation policy, which
            forbids fabricated citations but permits uncited general guidance.

    Returns:
        The UTF-8 prompt with the evidence-policy marker replaced and
        surrounding whitespace removed.

    Raises:
        OSError: If the prompt file cannot be read.
        ValueError: If the prompt template does not contain exactly one
            evidence-policy marker, preventing a silent policy omission.
    """
    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    if prompt.count(EVIDENCE_POLICY_MARKER) != 1:
        raise ValueError(
            f"System prompt must contain exactly one {EVIDENCE_POLICY_MARKER} marker."
        )
    evidence_policy = RAG_EVIDENCE_POLICY if require_evidence else NO_RAG_EVIDENCE_POLICY
    return prompt.replace(EVIDENCE_POLICY_MARKER, evidence_policy).strip()


def make_user_prompt(
    request: PlanRequest, evidence: str, *, require_evidence: bool = True
) -> str:
    """Combine request data, citation instructions, evidence, and output schema.

    Args:
        request: Validated student request to serialize into the prompt.
        evidence: Retrieved source block to provide to the model.
        require_evidence: Whether each exercise must cite exact supporting quotes;
            when false, instructs the model to leave citations empty.

    Returns:
        The complete user prompt, with request and retrieved passages in separate
        sections.
    """
    evidence_instructions = (
        "For every exercise, include at least one supporting_evidence item with a source_id and an exact, contiguous quote copied from that source. Each supporting quote must directly support the exercise or its prescription. source_ids must exactly match the source IDs in supporting_evidence.\n"
        if require_evidence
        else "No reference evidence is supplied. Leave source_ids and supporting_evidence empty; do not invent citations or quotes.\n"
    )
    return (
        "STUDENT REQUEST\n"
        + json.dumps(request.model_dump(), ensure_ascii=False, indent=2)
        + "\n\nPLAN LENGTH REQUIREMENT\n"
        + f"Return exactly {request.weeks} weeks, numbered consecutively from 1 to "
        + f"{request.weeks}. Include every week; do not return a sample or omit weeks.\n"
        + f"Schedule exactly {request.target_workouts_per_week} workouts in every week, on distinct days from the available days.\n"
        + "Write every generated plan field in English, regardless of the language of the request or sources.\n"
        + "Name specific exercises, never use a body-part or workout-split label as an exercise. Do not repeat an exercise within a workout. Keep workouts to 120 minutes or less and rest periods to 300 seconds or less.\n"
        + evidence_instructions
        + "Use only equipment listed in the student profile (bodyweight requires no equipment)."
        + "\n\nRETRIEVED SOURCES\n"
        + evidence
        + "\n\nRETURN THIS JSON SHAPE\n"
        + json.dumps(plan_response_schema(request), ensure_ascii=False)
    )


def plan_response_schema(request: PlanRequest) -> dict[str, Any]:
    """Constrain the plan schema to the request's week and workout counts.

    Args:
        request: Request whose exact week and per-week workout counts are required.

    Returns:
        A JSON schema for :class:`TrainingPlan` with exact ``weeks`` and
        ``workouts`` array lengths.
    """
    schema = TrainingPlan.model_json_schema()
    schema["properties"]["weeks"]["minItems"] = request.weeks
    schema["properties"]["weeks"]["maxItems"] = request.weeks
    workouts_schema = schema["$defs"]["Week"]["properties"]["workouts"]
    workouts_schema["minItems"] = request.target_workouts_per_week
    workouts_schema["maxItems"] = request.target_workouts_per_week
    return schema
