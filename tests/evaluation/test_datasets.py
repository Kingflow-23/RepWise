"""Dataset contract checks for benchmark cases and corpus evidence."""

import json

import pytest

from src.evaluation import adapters, datasets
from src.evaluation.metrics.grounding import normalise


def test_retrieval_cases_link_to_relevance_judgments():
    cases = datasets.load_retrieval_cases()
    assert len(cases) >= 150
    assert all(case.spans for case in cases if case.answerable)


def test_retrieval_queries_are_unique_and_cleanly_encoded():
    cases = datasets.load_retrieval_cases()
    queries = [normalise(case.query) for case in cases]
    assert len(queries) == len(set(queries))
    assert all(not datasets._has_mojibake(case.query) for case in cases)


def test_every_relevance_span_overlaps_a_real_corpus_chunk():
    cases = datasets.load_retrieval_cases()
    chunks = adapters.load_chunks()
    video_chunks = {}
    for chunk in chunks:
        video_chunks.setdefault(chunk["video_id"], []).append(chunk)

    for case in cases:
        for span in case.spans:
            assert span.video_id in video_chunks, (case.id, span.video_id)
            assert any(
                chunk["start"] < span.end and chunk["end"] > span.start
                for chunk in video_chunks[span.video_id]
            ), (case.id, span)


def test_every_relevance_span_overlaps_a_timestamped_source_passage():
    cases = datasets.load_retrieval_cases()
    passages_dir = datasets.REPO / "data" / "corpus" / "passages"
    passages_by_video = {}
    for path in passages_dir.glob("*.jsonl"):
        for _, passage in datasets.read_jsonl(path):
            passages_by_video.setdefault(passage["video_id"], []).append(passage)

    for case in cases:
        for span in case.spans:
            assert any(
                passage["start"] < span.end and passage["end"] > span.start
                for passage in passages_by_video.get(span.video_id, [])
            ), (case.id, span)


def test_relevance_judgments_show_verbatim_source_evidence():
    judgments = datasets.read_jsonl(datasets.RELEVANCE_JUDGMENTS)
    passages_dir = datasets.REPO / "data" / "corpus" / "passages"
    passages_by_video = {}
    for path in passages_dir.glob("*.jsonl"):
        for _, passage in datasets.read_jsonl(path):
            passages_by_video.setdefault(passage["video_id"], []).append(passage)

    for _, judgment in judgments:
        expected_text = " ".join(
            passage["text"].strip()
            for passage in passages_by_video[judgment["video_id"]]
            if passage["start"] < judgment["end"]
            and passage["end"] > judgment["start"]
        )
        assert judgment["evidence_text"].strip()
        assert judgment["evidence_text"] in expected_text
        assert judgment["source_url"] == (
            f"https://youtu.be/{judgment['video_id']}?t={judgment['start']}"
        )
        assert judgment["relevance_rationale"].strip()


def test_unanswerable_queries_have_no_labels_and_explain_why():
    cases = datasets.load_retrieval_cases()
    share = sum(not case.answerable for case in cases) / len(cases)
    assert 0.10 <= share <= 0.25
    assert all(not case.spans for case in cases if not case.answerable)
    assert all(case.answerability_note for case in cases if not case.answerable)


def test_generation_and_red_team_cases_are_separate_and_coherent():
    cases = datasets.load_generation_cases()
    red_team = datasets.load_red_team_cases()
    assert {case.id for case in cases} == {f"G{i:03}" for i in range(1, 31)}
    assert {case.id for case in red_team} == {f"RT{i:03}" for i in range(1, 9)}
    assert all(case.request and case.expected_status for case in cases)
    assert all(case.profile and case.weeks >= 1 for case in cases + red_team)

    answerable_ids = {
        case.id for case in datasets.load_retrieval_cases() if case.answerable
    }
    assert all(case.notes for case in cases)
    assert all(
        case.evidence_query_ids
        and set(case.evidence_query_ids) <= answerable_ids
        for case in cases
    )
    assert len({tuple(sorted(case.profile["equipment"])) for case in cases}) >= 5
    assert {case.profile["experience"] for case in cases} >= {
        "beginner",
        "intermediate",
        "advanced",
    }


def write(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    return path


def test_loader_rejects_answerable_case_without_evidence(tmp_path):
    cases = write(
        tmp_path / "cases.jsonl",
        [{"id": "X", "query": "q", "answerable": True, "topic": "test"}],
    )
    qrels = write(tmp_path / "qrels.jsonl", [])
    with pytest.raises(datasets.DatasetError, match="no relevance judgment"):
        datasets.load_retrieval_cases(cases, qrels)


def test_loader_rejects_mojibake_in_retrieval_queries(tmp_path):
    cases = write(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "X",
                "query": "s\u00c3\u00a9ries",
                "answerable": False,
                "topic": "test",
                "answerability_note": "Outside the corpus.",
            }
        ],
    )
    qrels = write(tmp_path / "qrels.jsonl", [])
    with pytest.raises(datasets.DatasetError, match="invalid text encoding"):
        datasets.load_retrieval_cases(cases, qrels)


def test_loader_rejects_unknown_query_id(tmp_path):
    cases = write(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "X",
                "query": "q",
                "answerable": False,
                "topic": "test",
                "answerability_note": "Outside the corpus.",
            }
        ],
    )
    qrels = write(
        tmp_path / "qrels.jsonl",
        [
            {
                "query_id": "Y",
                "video_id": "v",
                "source_url": "https://youtu.be/v?t=0",
                "start": 0,
                "end": 5,
                "evidence_text": "source excerpt",
                "relevance_rationale": "evidence rationale",
            }
        ],
    )
    with pytest.raises(datasets.DatasetError, match="unknown query_id"):
        datasets.load_retrieval_cases(cases, qrels)


def test_loader_rejects_unanswerable_case_without_rationale(tmp_path):
    cases = write(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "X",
                "query": "q",
                "answerable": False,
                "topic": "test",
            }
        ],
    )
    qrels = write(tmp_path / "qrels.jsonl", [])
    with pytest.raises(datasets.DatasetError, match="answerability_note"):
        datasets.load_retrieval_cases(cases, qrels)


def test_loader_rejects_relevance_judgment_without_rationale(tmp_path):
    cases = write(
        tmp_path / "cases.jsonl",
        [{"id": "X", "query": "q", "answerable": True, "topic": "test"}],
    )
    qrels = write(
        tmp_path / "qrels.jsonl",
        [
            {
                "query_id": "X",
                "video_id": "v",
                "source_url": "https://youtu.be/v?t=0",
                "start": 0,
                "end": 5,
                "evidence_text": "source excerpt",
            }
        ],
    )
    with pytest.raises(
        datasets.DatasetError, match="missing field.*relevance_rationale"
    ):
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


def test_generation_loader_rejects_link_to_unanswerable_retrieval_case(tmp_path):
    path = write(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "G001",
                "request": "Create a plan",
                "expected_status": "ok",
                "notes": "A test scenario.",
                "evidence_query_ids": ["R017"],
            }
        ],
    )
    with pytest.raises(datasets.DatasetError, match="answerable retrieval cases"):
        datasets.load_generation_cases(path)
