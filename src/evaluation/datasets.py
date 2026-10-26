"""Load the labelled retrieval and generation cases used by the benchmark."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .metrics.retrieval import Span

REPO = Path(__file__).resolve().parents[2]
EVALS = REPO / "evals"
RETRIEVAL_CASES = EVALS / "retrieval" / "retrieval_test_cases.jsonl"
RELEVANCE_JUDGMENTS = EVALS / "expected_evidence" / "relevance_judgments.jsonl"
GENERATION_CASES = EVALS / "benchmark_cases.jsonl"
STATUSES = {"ok", "abstain", "clarify", "blocked"}


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


def load_retrieval_cases(
    cases_path: Path = RETRIEVAL_CASES,
    qrels_path: Path = RELEVANCE_JUDGMENTS,
) -> list[RetrievalCase]:
    cases = {}
    for line_number, row in read_jsonl(cases_path):
        where = f"{cases_path.name}:{line_number}"
        _require(row, ("id", "query", "answerable"), where)
        if row["id"] in cases:
            raise DatasetError(f"{where}: duplicate retrieval case id {row['id']!r}")
        cases[row["id"]] = RetrievalCase(
            id=row["id"],
            query=row["query"],
            answerable=bool(row["answerable"]),
            topic=row.get("topic", ""),
        )

    for line_number, row in read_jsonl(qrels_path):
        where = f"{qrels_path.name}:{line_number}"
        _require(row, ("query_id", "video_id", "start", "end"), where)
        if row["query_id"] not in cases:
            raise DatasetError(f"{where}: unknown query_id {row['query_id']!r}")
        if row["end"] <= row["start"]:
            raise DatasetError(f"{where}: end must be greater than start")
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
    forbidden_patterns: list[str] = field(default_factory=list)
    required_patterns: list[str] = field(default_factory=list)
    notes: str = ""


def load_generation_cases(path: Path = GENERATION_CASES) -> list[GenerationCase]:
    cases = []
    for line_number, row in read_jsonl(path):
        where = f"{path.name}:{line_number}"
        _require(row, ("id", "request", "expected_status"), where)
        if not row["request"]:
            raise DatasetError(f"{where}: request must not be empty")
        expected = row["expected_status"]
        expected = [expected] if isinstance(expected, str) else expected
        if not isinstance(expected, list) or not expected:
            raise DatasetError(
                f"{where}: expected_status must be a status or non-empty list"
            )
        for status in expected:
            if status not in STATUSES:
                raise DatasetError(
                    f"{where}: expected_status={status!r} not in {sorted(STATUSES)}"
                )
        profile = row.get("profile", {})
        if not isinstance(profile, dict):
            raise DatasetError(f"{where}: profile must be an object when provided")
        for pattern in row.get("forbidden_patterns", []) + row.get(
            "required_patterns", []
        ):
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
                forbidden_patterns=row.get("forbidden_patterns", []),
                required_patterns=row.get("required_patterns", []),
                notes=row.get("notes", ""),
            )
        )
    _unique([case.id for case in cases], "generation case")
    return cases
