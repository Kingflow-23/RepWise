"""End-to-end smoke tests of the runner on a tiny synthetic corpus."""

import json

import pytest

from src.evaluation import adapters, runner
from config import evaluation_config as config
from src.evaluation.datasets import GenerationCase, RetrievalCase
from src.evaluation.metrics.retrieval import Span

CHUNKS = [
    {"video_id": "v1", "start": 0, "end": 60, "passage_ids": ["v1_p0001", "v1_p0004"], "flags": [],
     "text": "entre deux et trois minutes de récupération entre chaque série pour un débutant"},
    {"video_id": "v2", "start": 0, "end": 60, "passage_ids": ["v2_p0001", "v2_p0003"], "flags": [],
     "text": "la semaine de deload on réduit le volume de 50 à 70 pour cent en gardant les charges"},
    {"video_id": "v3", "start": 0, "end": 60, "passage_ids": ["v3_p0001", "v3_p0002"], "flags": ["sponsor"],
     "text": "code promo récupération deload volume"},
]


@pytest.fixture
def chunks_dir(tmp_path):
    d = tmp_path / "chunks"
    d.mkdir()
    for c in CHUNKS:
        (d / f"{c['video_id']}.jsonl").write_text(json.dumps(c, ensure_ascii=False) + "\n", encoding="utf-8")
    return d


def test_chunk_id_is_derived_from_passage_range(chunks_dir):
    ids = [c["chunk_id"] for c in adapters.load_chunks(chunks_dir)]
    assert ids[0] == "v1_p0001-p0004"


def test_bm25_finds_the_right_chunk_and_skips_sponsor(chunks_dir):
    bm25 = adapters.BM25Retriever(adapters.load_chunks(chunks_dir))
    top = bm25("combien de récupération entre les séries", 3)
    assert top[0]["video_id"] == "v1"
    assert all(c["video_id"] != "v3" for c in top)


def test_retrieval_suite_metrics_and_failures(chunks_dir):
    bm25 = adapters.BM25Retriever(adapters.load_chunks(chunks_dir))
    cases = [
        RetrievalCase("Q1", "temps de récupération entre séries", "test", True, spans=[Span("v1", 10, 20)]),
        RetrievalCase("Q2", "deload volume charges", "test", True, spans=[Span("v9", 0, 10)]),
        RetrievalCase("Q3", "prix abonnement", "test", False),
    ]
    res = runner.evaluate_retrieval(cases, bm25, ks=(1, 3), main_k=3)
    assert res.n_cases == 2 and res.n_passed == 1
    assert res.metrics["recall@1"].fmt() == "0.500 (1/2)"
    assert res.failures[0].case_id == "Q2" and res.failures[0].category == "not_retrieved"
    assert res.extra["n_unanswerable_skipped"] == 1


def gen_case(cid, expected, source="scenario", layer="generation", **kw):
    return GenerationCase(id=cid, source=source, layer=layer, request="r", split="test",
                          profile_id="P", expected_status=expected, **kw)


def test_generation_suite_with_naive_mock(chunks_dir):
    corpus = {c["chunk_id"]: c["text"] for c in adapters.load_chunks(chunks_dir)}
    planner = adapters.make_mock_naive(adapters.load_chunks(chunks_dir))
    profiles = {"P": {"days_per_week": 3, "max_session_min": 45, "equipment": ["dumbbells"]}}
    cases = [
        gen_case("S1", ["ok"]),
        gen_case("RT4", ["blocked", "clarify"], source="red_team", rt="RT4"),
        gen_case("RT9", ["hold"], source="red_team", layer="adaptation", rt="RT9", trajectory=[{"t": 1}]),
    ]
    res = runner.evaluate_generation(cases, planner, profiles, corpus)
    categories = {f.category for f in res.failures}
    assert {"fabricated_citation", "unsafe_release"} <= categories
    assert res.metrics["output_schema_validity"].value == 1
    assert res.metrics["red_team_pass_rate"].fmt() == "0.000 (0/1)"
    assert res.not_run and res.not_run[0]["case_id"] == "RT9"
    gates = {g["gate"]: g["status"] for g in runner.release_gates({"generation": res})}
    assert "fail" in gates.values() and "not evaluated" in gates.values()


def test_abstaining_mock_scores_abstention():
    cases = [gen_case("A", ["abstain"]), gen_case("B", ["ok"])]
    res = runner.evaluate_generation(cases, adapters.mock_abstain, {"P": {}}, corpus=None)
    assert res.metrics["abstention_accuracy"].value == 1
    assert res.metrics["false_abstention_rate"].value == 1
    assert res.metrics["citation_precision"].value is None


def test_crashing_planner_is_reported_not_raised():
    def broken(_case):
        raise RuntimeError("LLM timeout")
    res = runner.evaluate_generation([gen_case("X", ["ok"])], broken, {"P": {}}, corpus=None)
    assert res.failures[0].category == "pipeline_error"


def test_cli_writes_reports(chunks_dir, tmp_path, monkeypatch):
    """The command evaluates both suites and writes results.json and report.md."""
    out = tmp_path / "runs"
    monkeypatch.setattr(config, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(config, "OUT_DIR", out)
    code = runner.main(["--suite", "all", "--label", "smoke"])
    assert code == 0
    run = next(out.iterdir())
    payload = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert set(payload["suites"]) == {"retrieval", "generation"}
    assert "# RepWise evaluation report" in (run / "report.md").read_text(encoding="utf-8")


def test_cli_without_corpus_skips_retrieval_and_says_so(tmp_path, monkeypatch):
    """Without the chunk corpus the retrieval suite is skipped and the report warns about it."""
    out = tmp_path / "runs"
    monkeypatch.setattr(config, "CHUNKS_DIR", tmp_path / "missing")
    monkeypatch.setattr(config, "OUT_DIR", out)
    code = runner.main(["--suite", "all"])
    assert code == 0
    run = next(out.iterdir())
    payload = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert set(payload["suites"]) == {"generation"}
    assert "Retrieval suite skipped" in (run / "report.md").read_text(encoding="utf-8")


def test_fail_on_gates_returns_error_code(chunks_dir, tmp_path, monkeypatch):
    """With FAIL_ON_GATES, a failed release gate makes the command return 1."""
    monkeypatch.setattr(config, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(config, "OUT_DIR", tmp_path)
    monkeypatch.setattr(config, "FAIL_ON_GATES", True)
    code = runner.main(["--suite", "generation"])
    assert code == 1
