"""Dataset contract checks for the two benchmark case files."""

import json

import pytest

from src.evaluation import datasets


def test_retrieval_cases_link_to_relevance_judgments():
    cases = datasets.load_retrieval_cases()
    assert len(cases) >= 150
    assert all(case.spans for case in cases if case.answerable)


def test_retrieval_queries_are_unique():
    queries = [case.query.strip().lower() for case in datasets.load_retrieval_cases()]
    assert len(queries) == len(set(queries))


def test_unanswerable_queries_do_not_have_relevance_labels():
    cases = datasets.load_retrieval_cases()
    share = sum(not case.answerable for case in cases) / len(cases)
    assert 0.10 <= share <= 0.25
    assert all(not case.spans for case in cases if not case.answerable)


def test_unified_generation_cases_cover_program_and_safety_checks():
    cases = datasets.load_generation_cases()
    assert {case.id for case in cases} == {f"S{i:03}" for i in range(1, 8)} | {
        f"RT{i}" for i in range(1, 9)
    }
    assert all(case.request and case.expected_status for case in cases)
    assert next(case for case in cases if case.id == "RT6").profile == {}


def write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_loader_rejects_answerable_case_without_evidence(tmp_path):
    cases = write(
        tmp_path / "cases.jsonl", [{"id": "X", "query": "q", "answerable": True}]
    )
    qrels = write(tmp_path / "qrels.jsonl", [])
    with pytest.raises(datasets.DatasetError, match="no relevance judgment"):
        datasets.load_retrieval_cases(cases, qrels)


def test_loader_rejects_unknown_query_id(tmp_path):
    cases = write(
        tmp_path / "cases.jsonl", [{"id": "X", "query": "q", "answerable": False}]
    )
    qrels = write(
        tmp_path / "qrels.jsonl",
        [{"query_id": "Y", "video_id": "v", "start": 0, "end": 5}],
    )
    with pytest.raises(datasets.DatasetError, match="unknown query_id"):
        datasets.load_retrieval_cases(cases, qrels)


def test_loader_reports_file_and_line_on_bad_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id":"X"}\n{oops\n', encoding="utf-8")
    with pytest.raises(datasets.DatasetError, match="bad.jsonl:2"):
        datasets.read_jsonl(bad)


def test_generation_loader_rejects_unknown_status(tmp_path):
    path = write(
        tmp_path / "cases.jsonl",
        [{"id": "X", "request": "q", "expected_status": "maybe"}],
    )
    with pytest.raises(datasets.DatasetError, match="expected_status"):
        datasets.load_generation_cases(path)
