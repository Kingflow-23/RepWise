"""Validate benchmark case schemas, relevance links, and corpus evidence."""

import json

import pytest

from src.evaluation import adapters, datasets
from src.evaluation.metrics.grounding import normalise

# The built corpus is gitignored, so it is absent on a fresh clone (and in CI).
# Tests that compare the datasets with it are skipped there instead of failing.
CORPUS = datasets.REPO / "data" / "corpus"
needs_corpus = pytest.mark.skipif(
    not any((CORPUS / "chunks").glob("*.jsonl"))
    or not any((CORPUS / "passages").glob("*.jsonl")),
    reason="corpus not built (data/corpus/chunks and passages are gitignored)",
)


def test_retrieval_cases_link_to_relevance_judgments():
    """Require the retrieval set to contain only answerable, judged queries."""
    cases = datasets.load_retrieval_cases()
    assert len(cases) == 128
    assert all(case.answerable and case.spans for case in cases)


def test_retrieval_queries_are_unique_and_cleanly_encoded():
    """Keep benchmark queries distinct after normalization and free of mojibake."""
    cases = datasets.load_retrieval_cases()
    queries = [normalise(case.query) for case in cases]
    assert len(queries) == len(set(queries))
    assert all(not datasets._has_mojibake(case.query) for case in cases)


@needs_corpus
def test_every_relevance_span_overlaps_a_real_corpus_chunk():
    """Ensure every judged interval occurs in at least one indexed chunk."""
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


@needs_corpus
def test_every_relevance_span_overlaps_a_timestamped_source_passage():
    """Ensure every judged interval is grounded in timestamped source passages."""
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


@needs_corpus
def test_relevance_judgments_show_verbatim_source_evidence():
    """Match each judgment's excerpt and URL to its source passage interval."""
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
            if passage["start"] < judgment["end"] and passage["end"] > judgment["start"]
        )
        assert judgment["evidence_text"].strip()
        assert judgment["evidence_text"] in expected_text
        assert judgment["source_url"] == (
            f"https://youtu.be/{judgment['video_id']}?t={judgment['start']}"
        )
        assert judgment["relevance_rationale"].strip()


def test_retrieval_dataset_contains_no_unanswerable_queries():
    """Keep unsupported queries out of the retrieval-scoring dataset."""
    cases = datasets.load_retrieval_cases()
    assert all(case.answerable for case in cases)


def test_generation_and_red_team_cases_are_separate_and_coherent():
    """Check suite IDs, valid one-week profiles, and answerable evidence links."""
    cases = datasets.load_generation_cases()
    red_team = datasets.load_red_team_cases()
    assert {case.id for case in cases} == {
        "G001",
        "G002",
        "G003",
        "G004",
        "G005",
        "G006",
        "G008",
        "G010",
        "G013",
        "G016",
        "G018",
        "G020",
        "G022",
        "G025",
        "G027",
    }
    assert {case.id for case in red_team} == {f"RT{i:03}" for i in range(1, 16)}
    assert len(red_team) == 15
    assert {case.id for case in red_team if case.expected_scope == "out_of_scope"} == {
        "RT005",
        "RT006",
        "RT013",
        "RT014",
        "RT015",
    }
    assert all(case.expected_scope == "in_scope" for case in cases)
    assert all(case.request and case.expected_status for case in cases)
    assert all(
        case.profile
        and case.weeks == 1
        and case.workouts_per_week is not None
        and case.workouts_per_week <= len(case.profile["available_days"])
        for case in cases + red_team
    )

    answerable_ids = {
        case.id for case in datasets.load_retrieval_cases() if case.answerable
    }
    assert all(case.notes for case in cases)
    assert all(case.notes for case in red_team)
    assert all(
        case.evidence_query_ids and set(case.evidence_query_ids) <= answerable_ids
        for case in cases
    )
    assert len({tuple(sorted(case.profile["equipment"])) for case in cases}) >= 5
    assert {case.profile["experience"] for case in cases} >= {
        "beginner",
        "intermediate",
        "advanced",
    }


def test_smoke_manifest_selects_valid_representative_cases():
    """Keep the bounded evaluation subset linked to validated full datasets."""
    selected = datasets.load_smoke_case_ids()
    retrieval = datasets.select_cases(
        datasets.load_retrieval_cases(), selected["retrieval"], "retrieval"
    )
    generation = datasets.select_cases(
        datasets.load_generation_cases(), selected["generation"], "generation"
    )
    red_team = datasets.select_cases(
        datasets.load_red_team_cases(), selected["red_team"], "red-team"
    )

    assert len(retrieval) == 20
    assert {case.topic for case in retrieval} >= {
        "rest_intervals",
        "deload",
        "pain_safety",
    }
    assert all(case.answerable and case.spans for case in retrieval)
    assert [case.id for case in generation] == ["G001"]
    assert {case.id for case in red_team} == {"RT001", "RT004", "RT005", "RT013"}


def write(path, rows):
    """Write JSONL rows to a test fixture file and return its path.

    Args:
        path: Destination path for the generated fixture.
        rows: JSON-compatible objects to serialize one per line.
    """
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    return path


def test_loader_rejects_answerable_case_without_evidence(tmp_path):
    """Reject an answerable query that has no relevance judgments.

    Args:
        tmp_path: Hold the temporary query and judgment datasets.
    """
    cases = write(
        tmp_path / "cases.jsonl",
        [{"id": "X", "query": "q", "answerable": True, "topic": "test"}],
    )
    qrels = write(tmp_path / "qrels.jsonl", [])
    with pytest.raises(datasets.DatasetError, match="no relevance judgment"):
        datasets.load_retrieval_cases(cases, qrels)


def test_loader_rejects_mojibake_in_retrieval_queries(tmp_path):
    """Reject query text whose encoding has produced mojibake.

    Args:
        tmp_path: Hold the malformed query and its empty judgment file.
    """
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
    """Reject relevance judgments that refer to an absent query.

    Args:
        tmp_path: Hold the query and judgment files for the loader.
    """
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
    """Require an explanation for queries marked unanswerable.

    Args:
        tmp_path: Hold the invalid query and its empty judgment file.
    """
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
    """Require a rationale on each relevance judgment.

    Args:
        tmp_path: Hold the answerable query and incomplete judgment file.
    """
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
    """Include the source filename and line number in JSONL parse errors.

    Args:
        tmp_path: Provide a location for the intentionally malformed JSONL file.
    """
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id":"X"}\n{oops\n', encoding="utf-8")
    with pytest.raises(datasets.DatasetError, match="bad.jsonl:2"):
        datasets.read_jsonl(bad)


def test_generation_loader_rejects_unknown_status(tmp_path):
    """Reject generation cases with an unsupported expected status.

    Args:
        tmp_path: Hold the invalid generation-case file.
    """
    path = write(
        tmp_path / "cases.jsonl",
        [{"id": "X", "request": "q", "expected_status": "maybe"}],
    )
    with pytest.raises(datasets.DatasetError, match="expected_status"):
        datasets.load_generation_cases(path)


def test_generation_loader_rejects_link_to_missing_retrieval_case(tmp_path):
    """Reject generation evidence links not present in the retrieval dataset.

    Args:
        tmp_path: Hold the generation case with the invalid evidence link.
    """
    path = write(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "G001",
                "request": "Create a plan",
                "expected_status": "ok",
                "notes": "A test scenario.",
                "evidence_query_ids": ["R999"],
            }
        ],
    )
    with pytest.raises(datasets.DatasetError, match="answerable retrieval cases"):
        datasets.load_generation_cases(path)
