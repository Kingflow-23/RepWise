"""Load and validate the JSONL evaluation datasets under evals/.

Every loader fails loudly with the file name and line number, so a broken test
case is caught before a run instead of silently skewing the metrics.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .metrics.retrieval import Span

REPO = Path(__file__).resolve().parents[2]
EVALS = REPO / "evals"

RETRIEVAL_CASES = EVALS / "retrieval" / "retrieval_test_cases.jsonl"
RELEVANCE_JUDGMENTS = EVALS / "expected_evidence" / "relevance_judgments.jsonl"
PROFILES = EVALS / "profiles" / "fixed_profiles.jsonl"
PROGRAM_SCENARIOS = EVALS / "scenarios" / "program_scenarios.jsonl"
SAFETY_CASES = EVALS / "red_team" / "safety_cases.jsonl"

STATUSES = {"ok", "abstain", "clarify", "blocked", "hold"}
SPLITS = {"dev", "test"}
LAYERS = {"retrieval", "generation", "adaptation"}


class DatasetError(ValueError):
    pass


def read_jsonl(path: Path) -> list[tuple[int, dict]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append((lineno, json.loads(line)))
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}:{lineno}: invalid JSON ({e.msg})") from e
    return rows


def _require(row: dict, keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in row]
    if missing:
        raise DatasetError(f"{where}: missing field(s) {', '.join(missing)}")


def _check_choice(value, allowed: set, field_name: str, where: str) -> None:
    if value not in allowed:
        raise DatasetError(f"{where}: {field_name}={value!r} not in {sorted(allowed)}")


def _unique(ids: list[str], what: str) -> None:
    seen = set()
    for i in ids:
        if i in seen:
            raise DatasetError(f"duplicate {what} id {i!r}")
        seen.add(i)


# ---------------------------------------------------------------- retrieval

@dataclass
class RetrievalCase:
    id: str
    query: str
    split: str
    answerable: bool
    topic: str = ""
    spans: list[Span] = field(default_factory=list)


def load_retrieval_cases(cases_path: Path = RETRIEVAL_CASES, qrels_path: Path = RELEVANCE_JUDGMENTS) -> list[RetrievalCase]:
    cases = {}
    for lineno, row in read_jsonl(cases_path):
        where = f"{cases_path.name}:{lineno}"
        _require(row, ("id", "query", "split", "answerable"), where)
        _check_choice(row["split"], SPLITS, "split", where)
        if row["id"] in cases:
            raise DatasetError(f"{where}: duplicate retrieval case id {row['id']!r}")
        cases[row["id"]] = RetrievalCase(
        id=row["id"], query=row["query"], split=row["split"],
        answerable=bool(row["answerable"]), topic=row.get("topic", ""))


    for lineno, row in read_jsonl(qrels_path):
        where = f"{qrels_path.name}:{lineno}"
        _require(row, ("query_id", "video_id", "start", "end"), where)
        if row["query_id"] not in cases:
            raise DatasetError(f"{where}: unknown query_id {row['query_id']!r}")
        if row["end"] <= row["start"]:
            raise DatasetError(f"{where}: end must be greater than start")
        cases[row["query_id"]].spans.append(Span(row["video_id"], row["start"], row["end"]))

    for case in cases.values():
        if case.answerable and not case.spans:
            raise DatasetError(f"retrieval case {case.id}: answerable but has no relevance judgment")
        if not case.answerable and case.spans:
            raise DatasetError(f"retrieval case {case.id}: unanswerable but has relevance judgments")
    return list(cases.values())


# ---------------------------------------------------------------- profiles

def load_profiles(path: Path = PROFILES) -> dict[str, dict]:
    profiles = {}
    for lineno, row in read_jsonl(path):
        where = f"{path.name}:{lineno}"
        _require(row, ("profile_id",), where)
        if row["profile_id"] in profiles:
            raise DatasetError(f"{where}: duplicate profile_id {row['profile_id']!r}")
        profiles[row["profile_id"]] = row
    return profiles


# ------------------------------------------------- generation and red team

@dataclass
class GenerationCase:
    id: str
    source: str  # "scenario" or "red_team"
    layer: str
    request: str
    split: str
    profile_id: str | None
    expected_status: list[str]
    rt: str | None = None
    forbidden_patterns: list[str] = field(default_factory=list)
    required_patterns: list[str] = field(default_factory=list)
    profile_overrides: dict = field(default_factory=dict)
    trajectory: list[dict] = field(default_factory=list)
    expected_state: dict = field(default_factory=dict)
    notes: str = ""


def _generation_case(row: dict, source: str, where: str, profiles: dict) -> GenerationCase:
    _require(row, ("id", "layer", "split", "expected_status"), where)
    _check_choice(row["layer"], LAYERS - {"retrieval"}, "layer", where)
    _check_choice(row["split"], SPLITS, "split", where)
    expected = row["expected_status"]
    expected = [expected] if isinstance(expected, str) else list(expected)
    for s in expected:
        _check_choice(s, STATUSES, "expected_status", where)
    profile_id = row.get("profile_id")
    if profile_id is not None and profile_id not in profiles:
        raise DatasetError(f"{where}: unknown profile_id {profile_id!r}")
    if row["layer"] == "generation" and not row.get("request"):
        raise DatasetError(f"{where}: generation case needs a 'request'")
    if row["layer"] == "adaptation" and not row.get("trajectory"):
        raise DatasetError(f"{where}: adaptation case needs a 'trajectory'")
    return GenerationCase(
        id=row["id"], source=source, layer=row["layer"], request=row.get("request", ""),
        split=row["split"], profile_id=profile_id, expected_status=expected,
        rt=row.get("rt"), forbidden_patterns=row.get("forbidden_patterns", []),
        required_patterns=row.get("required_patterns", []),
        profile_overrides=row.get("profile_overrides", {}),
        trajectory=row.get("trajectory", []), expected_state=row.get("expected_state", {}),
        notes=row.get("notes", ""))


def load_generation_cases(
    scenarios_path: Path = PROGRAM_SCENARIOS,
    safety_path: Path = SAFETY_CASES,
    profiles: dict | None = None,
) -> list[GenerationCase]:
    profiles = load_profiles() if profiles is None else profiles
    cases = []
    for path, source in ((scenarios_path, "scenario"), (safety_path, "red_team")):
        for lineno, row in read_jsonl(path):
            cases.append(_generation_case(row, source, f"{path.name}:{lineno}", profiles))
    _unique([c.id for c in cases], "generation case")
    return cases
