"""Guards on the frozen evaluation datasets themselves."""

import json

import pytest

from src.evaluation import datasets


def test_retrieval_cases_load_and_link_to_judgments():
    cases = datasets.load_retrieval_cases()
    assert len(cases) >= 150
    assert all(c.spans for c in cases if c.answerable)


def test_queries_are_unique():
    texts = [c.query.strip().lower() for c in datasets.load_retrieval_cases()]
    assert len(texts) == len(set(texts))


def test_unanswerable_share_follows_course_guideline():
    cases = datasets.load_retrieval_cases()
    share = sum(not c.answerable for c in cases) / len(cases)
    assert 0.10 <= share <= 0.25


def test_dev_split_is_about_one_third_and_has_unanswerable_cases():
    cases = datasets.load_retrieval_cases()
    dev = [c for c in cases if c.split == "dev"]
    assert 0.25 <= len(dev) / len(cases) <= 0.40
    assert any(not c.answerable for c in dev)


def test_generation_cases_cover_all_red_team_scenarios():
    cases = datasets.load_generation_cases()
    covered = {c.rt for c in cases if c.rt}
    assert covered == {f"RT{i}" for i in range(1, 16)}


def test_every_case_has_dev_or_test_split():
    for c in datasets.load_generation_cases():
        assert c.split in {"dev", "test"}


def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_loader_rejects_answerable_case_without_evidence(tmp_path):
    cases = write(tmp_path / "c.jsonl", [{"id": "X", "query": "q", "split": "dev", "answerable": True}])
    qrels = write(tmp_path / "q.jsonl", [])
    with pytest.raises(datasets.DatasetError, match="no relevance judgment"):
        datasets.load_retrieval_cases(cases, qrels)


def test_loader_rejects_unknown_query_id(tmp_path):
    cases = write(tmp_path / "c.jsonl", [{"id": "X", "query": "q", "split": "dev", "answerable": False}])
    qrels = write(tmp_path / "q.jsonl", [{"query_id": "Y", "video_id": "v", "start": 0, "end": 5}])
    with pytest.raises(datasets.DatasetError, match="unknown query_id"):
        datasets.load_retrieval_cases(cases, qrels)


def test_loader_reports_file_and_line_on_bad_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "X"}\n{oops\n', encoding="utf-8")
    with pytest.raises(datasets.DatasetError, match="bad.jsonl:2"):
        datasets.read_jsonl(bad)


def test_loader_rejects_unknown_status(tmp_path):
    scen = write(tmp_path / "s.jsonl", [{"id": "S", "layer": "generation", "split": "dev",
                                         "request": "r", "expected_status": "maybe"}])
    empty = write(tmp_path / "e.jsonl", [])
    with pytest.raises(datasets.DatasetError, match="expected_status"):
        datasets.load_generation_cases(scen, empty, profiles={})
