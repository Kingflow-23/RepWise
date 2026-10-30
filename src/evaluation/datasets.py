"""Load the labelled retrieval and generation cases used by the benchmark."""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .metrics.retrieval import Span
from .metrics.grounding import normalise
from ..schemas.models import UserProfile

REPO = Path(__file__).resolve().parents[2]
EVALS = REPO / "evals"
RETRIEVAL_CASES = EVALS / "retrieval" / "retrieval_test_cases.jsonl"
RELEVANCE_JUDGMENTS = EVALS / "expected_evidence" / "relevance_judgments.jsonl"
GENERATION_CASES = EVALS / "benchmark_cases.jsonl"
RED_TEAM_CASES = EVALS / "red_team" / "red_team_cases.jsonl"
STATUSES = {"ok", "abstain", "clarify", "blocked"}
MOJIBAKE_MARKERS = ("\ufffd", "\u00c3", "\u00c2", "\u00e2\u20ac")


def _has_mojibake(text: str) -> bool:
    return any(marker in text for marker in MOJIBAKE_MARKERS)


class DatasetError(ValueError):
    """An evaluation case is invalid or cannot be linked to its labels."""


def read_jsonl(path: Path) -> list[tuple[int, dict]]:
    rows = []
    try:
        with Path(path).open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append((line_number, json.loads(line)))
                except json.JSONDecodeError as error:
                    raise DatasetError(
                        f"{path}:{line_number}: invalid JSON ({error.msg})"
                    ) from error
    except OSError as error:
        raise DatasetError(f"cannot read {path}: {error}") from error
    return rows


def _require(row: dict, keys: tuple[str, ...], where: str) -> None:
    if not isinstance(row, dict):
        raise DatasetError(f"{where}: each JSONL line must be an object")
    missing = [key for key in keys if key not in row]
    if missing:
        raise DatasetError(f"{where}: missing field(s) {', '.join(missing)}")


def _unique(ids: list[str], kind: str) -> None:
    if len(ids) != len(set(ids)):
        seen = set()
        duplicate = next(item for item in ids if item in seen or seen.add(item))
        raise DatasetError(f"duplicate {kind} id {duplicate!r}")


@dataclass
class RetrievalCase:
    id: str
    query: str
    answerable: bool
    topic: str = ""
    spans: list[Span] = field(default_factory=list)
    answerability_note: str = ""


def load_retrieval_cases(
    cases_path: Path = RETRIEVAL_CASES,
    qrels_path: Path = RELEVANCE_JUDGMENTS,
) -> list[RetrievalCase]:
    cases = {}
    normalized_queries = set()
    for line_number, row in read_jsonl(cases_path):
        where = f"{cases_path.name}:{line_number}"
        _require(row, ("id", "query", "answerable"), where)
        if not isinstance(row["id"], str) or not row["id"].strip():
            raise DatasetError(f"{where}: id must be a non-empty string")
        if not isinstance(row["query"], str) or not row["query"].strip():
            raise DatasetError(f"{where}: query must be a non-empty string")
        if not isinstance(row["answerable"], bool):
            raise DatasetError(f"{where}: answerable must be a boolean")
        topic = row.get("topic")
        if not isinstance(topic, str) or not topic.strip():
            raise DatasetError(f"{where}: topic must be a non-empty string")
        answerability_note = row.get("answerability_note", "")
        if not isinstance(answerability_note, str):
            raise DatasetError(f"{where}: answerability_note must be a string")
        if _has_mojibake(answerability_note):
            raise DatasetError(f"{where}: answerability_note has invalid text encoding")
        if not row["answerable"] and not answerability_note.strip():
            raise DatasetError(
                f"{where}: unanswerable cases need an answerability_note"
            )
        if row["id"] in cases:
            raise DatasetError(f"{where}: duplicate retrieval case id {row['id']!r}")
        if _has_mojibake(row["query"]):
            raise DatasetError(f"{where}: query contains invalid text encoding")
        normalized_query = normalise(row["query"])
        if normalized_query in normalized_queries:
            raise DatasetError(
                f"{where}: duplicate normalized retrieval query (case/accent/punctuation insensitive)"
            )
        normalized_queries.add(normalized_query)
        cases[row["id"]] = RetrievalCase(
            id=row["id"],
            query=row["query"],
            answerable=row["answerable"],
            topic=topic.strip(),
            answerability_note=answerability_note.strip(),
        )

    seen_judgments = set()
    for line_number, row in read_jsonl(qrels_path):
        where = f"{qrels_path.name}:{line_number}"
        _require(
            row,
            (
                "query_id",
                "video_id",
                "source_url",
                "start",
                "end",
                "evidence_text",
                "relevance_rationale",
            ),
            where,
        )
        for field in ("evidence_text", "relevance_rationale"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise DatasetError(f"{where}: {field} must be a non-empty string")
            if _has_mojibake(row[field]):
                raise DatasetError(f"{where}: {field} contains invalid text encoding")
        if not isinstance(row["query_id"], str) or not row["query_id"].strip():
            raise DatasetError(f"{where}: query_id must be a non-empty string")
        if row["query_id"] not in cases:
            raise DatasetError(f"{where}: unknown query_id {row['query_id']!r}")
        if (
            not isinstance(row["video_id"], str)
            or not row["video_id"].strip()
            or not isinstance(row["source_url"], str)
            or isinstance(row["start"], bool)
            or isinstance(row["end"], bool)
            or not isinstance(row["start"], (int, float))
            or not isinstance(row["end"], (int, float))
            or not math.isfinite(row["start"])
            or not math.isfinite(row["end"])
            or row["start"] < 0
            or row["end"] <= row["start"]
        ):
            raise DatasetError(f"{where}: invalid video_id or evidence time range")
        expected_source_url = (
            f"https://youtu.be/{row['video_id']}?t={row['start']}"
        )
        if row["source_url"] != expected_source_url:
            raise DatasetError(
                f"{where}: source_url must link to the cited video and start time"
            )
        judgment = (
            row["query_id"],
            row["video_id"],
            row["start"],
            row["end"],
        )
        if judgment in seen_judgments:
            raise DatasetError(f"{where}: duplicate relevance judgment")
        seen_judgments.add(judgment)
        cases[row["query_id"]].spans.append(
            Span(row["video_id"], row["start"], row["end"])
        )

    for case in cases.values():
        if case.answerable and not case.spans:
            raise DatasetError(
                f"retrieval case {case.id}: answerable but has no relevance judgment"
            )
        if not case.answerable and case.spans:
            raise DatasetError(
                f"retrieval case {case.id}: unanswerable but has relevance judgments"
            )
    return list(cases.values())


@dataclass
class GenerationCase:
    id: str
    request: str
    expected_status: list[str]
    profile: dict = field(default_factory=dict)
    weeks: int = 4
    extra_constraints: list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)
    required_patterns: list[str] = field(default_factory=list)
    notes: str = ""
    evidence_query_ids: list[str] = field(default_factory=list)


def load_generation_cases(
    path: Path = GENERATION_CASES,
) -> list[GenerationCase]:
    is_nominal_dataset = Path(path).resolve() == GENERATION_CASES.resolve()
    answerable_ids = (
        {case.id for case in load_retrieval_cases() if case.answerable}
        if is_nominal_dataset
        else None
    )
    cases = []
    for line_number, row in read_jsonl(path):
        where = f"{path.name}:{line_number}"
        _require(row, ("id", "request", "expected_status"), where)
        if not isinstance(row["id"], str) or not row["id"].strip():
            raise DatasetError(f"{where}: id must be a non-empty string")
        if not isinstance(row["request"], str) or not row["request"].strip():
            raise DatasetError(f"{where}: request must not be empty")
        if _has_mojibake(row["request"]):
            raise DatasetError(f"{where}: request contains invalid text encoding")
        expected = row["expected_status"]
        expected = [expected] if isinstance(expected, str) else expected
        if not isinstance(expected, list) or not expected:
            raise DatasetError(
                f"{where}: expected_status must be a status or non-empty list"
            )
        for status in expected:
            if not isinstance(status, str) or status not in STATUSES:
                raise DatasetError(
                    f"{where}: expected_status={status!r} not in {sorted(STATUSES)}"
                )
        profile = row.get("profile", {})
        if not isinstance(profile, dict):
            raise DatasetError(f"{where}: profile must be an object when provided")
        if profile:
            try:
                UserProfile.model_validate(profile)
            except ValueError as error:
                raise DatasetError(f"{where}: invalid profile: {error}") from error
        weeks = row.get("weeks", 4)
        if isinstance(weeks, bool) or not isinstance(weeks, int) or not 1 <= weeks <= 12:
            raise DatasetError(f"{where}: weeks must be an integer from 1 to 12")
        constraints = row.get("extra_constraints", [])
        if not isinstance(constraints, list) or not all(
            isinstance(item, str) and item.strip() for item in constraints
        ):
            raise DatasetError(f"{where}: extra_constraints must be non-empty strings")
        forbidden = row.get("forbidden_patterns", [])
        required = row.get("required_patterns", [])
        notes = row.get("notes", "")
        evidence_query_ids = row.get("evidence_query_ids", [])
        if not isinstance(notes, str) or not notes.strip():
            raise DatasetError(f"{where}: notes must explain why the case is in scope")
        if _has_mojibake(notes):
            raise DatasetError(f"{where}: notes contain invalid text encoding")
        if not isinstance(evidence_query_ids, list) or not all(
            isinstance(item, str) and item.strip() for item in evidence_query_ids
        ):
            raise DatasetError(f"{where}: evidence_query_ids must be non-empty strings")
        if len(evidence_query_ids) != len(set(evidence_query_ids)):
            raise DatasetError(f"{where}: evidence_query_ids must be unique")
        if (
            is_nominal_dataset and not evidence_query_ids
        ):
            raise DatasetError(
                f"{where}: nominal benchmark cases need answerable evidence_query_ids"
            )
        if evidence_query_ids:
            valid_ids = answerable_ids or {
                case.id for case in load_retrieval_cases() if case.answerable
            }
            unknown = set(evidence_query_ids) - valid_ids
            if unknown:
                raise DatasetError(
                    f"{where}: evidence_query_ids must reference answerable retrieval cases; "
                    f"invalid: {', '.join(sorted(unknown))}"
                )
        if not isinstance(expected, list) or not all(
            isinstance(status, str) for status in expected
        ):
            raise DatasetError(f"{where}: expected_status values must be strings")
        if not isinstance(forbidden, list) or not all(
            isinstance(item, str) for item in forbidden
        ):
            raise DatasetError(f"{where}: forbidden_patterns must be strings")
        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise DatasetError(f"{where}: required_patterns must be strings")
        for pattern in forbidden + required:
            try:
                re.compile(pattern)
            except re.error as error:
                raise DatasetError(
                    f"{where}: invalid output pattern {pattern!r}: {error}"
                ) from error
        cases.append(
            GenerationCase(
                id=row["id"],
                request=row["request"],
                expected_status=expected,
                profile=profile,
                weeks=weeks,
                extra_constraints=constraints,
                forbidden_patterns=forbidden,
                required_patterns=required,
                notes=notes.strip(),
                evidence_query_ids=evidence_query_ids,
            )
        )
    _unique([case.id for case in cases], "generation case")
    if is_nominal_dataset:
        if len(cases) < 30:
            raise DatasetError(
                f"{path.name}: nominal generation benchmark needs at least 30 cases"
            )
        if any(case.expected_status != ["ok"] for case in cases):
            raise DatasetError(
                f"{path.name}: nominal planning cases must expect status 'ok'"
            )
        normalized_requests = [normalise(case.request) for case in cases]
        if len(normalized_requests) != len(set(normalized_requests)):
            raise DatasetError(f"{path.name}: duplicate normalized generation request")
    return cases


def load_red_team_cases(path: Path = RED_TEAM_CASES) -> list[GenerationCase]:
    """Load robustness cases independently from nominal generation benchmark cases."""
    cases = load_generation_cases(path)
    if not cases:
        raise DatasetError(f"{path.name}: red-team dataset must not be empty")
    return cases
