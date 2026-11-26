"""Load retrieval and red-team evaluation datasets."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .metrics.retrieval import Span

REPO = Path(__file__).resolve().parents[2]
EVALS = REPO / "evals"
RETRIEVAL_CASES = EVALS / "retrieval" / "retrieval_test_cases.jsonl"
RELEVANCE_JUDGMENTS = (
    EVALS / "expected_evidence" / "evidences_for_retrieval_cases.jsonl"
)
RED_TEAM_CASES = EVALS / "red_team" / "red_team_test_cases.jsonl"


class DatasetError(ValueError):
    """An evaluation dataset is invalid or unreadable."""


def read_jsonl(path: Path) -> list[tuple[int, dict]]:
    """Read JSONL rows and retain their source line numbers."""
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
    topic: str
    spans: list[Span] = field(default_factory=list)
    answerability_note: str = ""


def load_retrieval_cases(
    cases_path: Path = RETRIEVAL_CASES,
    qrels_path: Path = RELEVANCE_JUDGMENTS,
) -> list[RetrievalCase]:
    """Load retrieval queries and attach their relevance spans."""
    cases = {}
    for line_number, row in read_jsonl(cases_path):
        where = f"{cases_path.name}:{line_number}"
        _require(row, ("id", "query", "answerable", "topic"), where)
        if not isinstance(row["id"], str) or not row["id"].strip():
            raise DatasetError(f"{where}: id must be a non-empty string")
        if not isinstance(row["query"], str) or not row["query"].strip():
            raise DatasetError(f"{where}: query must be a non-empty string")
        if not isinstance(row["answerable"], bool):
            raise DatasetError(f"{where}: answerable must be a boolean")
        if not isinstance(row["topic"], str) or not row["topic"].strip():
            raise DatasetError(f"{where}: topic must be a non-empty string")
        if row["id"] in cases:
            raise DatasetError(f"{where}: duplicate retrieval case id {row['id']!r}")
        cases[row["id"]] = RetrievalCase(
            id=row["id"],
            query=row["query"],
            answerable=row["answerable"],
            topic=row["topic"].strip(),
        )

    seen_judgments = set()
    for line_number, row in read_jsonl(qrels_path):
        where = f"{qrels_path.name}:{line_number}"
        _require(
            row,
            ("query_id", "video_id", "source_url", "start", "end"),
            where,
        )
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
        expected_url = f"https://youtu.be/{row['video_id']}?t={row['start']}"
        if row["source_url"] != expected_url:
            raise DatasetError(f"{where}: source_url must link to the cited video")
        judgment = (row["query_id"], row["video_id"], row["start"], row["end"])
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
class RedTeamCase:
    id: str
    query: str
    answerable: bool
    notes: str


def load_red_team_cases(path: Path = RED_TEAM_CASES) -> list[RedTeamCase]:
    """Load red-team prompts and their expected scope."""
    cases = []
    for line_number, row in read_jsonl(path):
        where = f"{path.name}:{line_number}"
        _require(row, ("id", "query", "answerable", "notes"), where)
        if not isinstance(row["id"], str) or not row["id"].strip():
            raise DatasetError(f"{where}: id must be a non-empty string")
        if not isinstance(row["query"], str) or not row["query"].strip():
            raise DatasetError(f"{where}: query must not be empty")
        if not isinstance(row["answerable"], bool):
            raise DatasetError(f"{where}: answerable must be a boolean")
        if not isinstance(row["notes"], str) or not row["notes"].strip():
            raise DatasetError(f"{where}: notes must not be empty")
        cases.append(
            RedTeamCase(
                id=row["id"],
                query=row["query"],
                answerable=row["answerable"],
                notes=row["notes"].strip(),
            )
        )
    _unique([case.id for case in cases], "red-team case")
    if not cases:
        raise DatasetError(f"{path.name}: red-team dataset must not be empty")
    return cases
