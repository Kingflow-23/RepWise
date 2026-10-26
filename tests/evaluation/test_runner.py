"""End-to-end smoke tests of the runner on a tiny synthetic corpus."""

import json

import pytest

from src.evaluation import adapters, runner
from config import evaluation_config as config
from src.evaluation.datasets import GenerationCase, RetrievalCase
from src.evaluation.metrics.retrieval import Span

CHUNKS = [
    {
        "video_id": "v1",
        "start": 0,
        "end": 60,
        "passage_ids": ["v1_p0001", "v1_p0004"],
        "flags": [],
        "text": "entre deux et trois minutes de rÃ©cupÃ©ration entre chaque sÃ©rie pour un dÃ©butant",
    },
    {
        "video_id": "v2",
        "start": 0,
        "end": 60,
        "passage_ids": ["v2_p0001", "v2_p0003"],
        "flags": [],
        "text": "la semaine de deload on rÃ©duit le volume de 50 Ã  70 pour cent en gardant les charges",
    },
    {
        "video_id": "v3",
        "start": 0,
        "end": 60,
        "passage_ids": ["v3_p0001", "v3_p0002"],
        "flags": ["sponsor"],
        "text": "code promo rÃ©cupÃ©ration deload volume",
    },
]


@pytest.fixture
def chunks_dir(tmp_path):
    d = tmp_path / "chunks"
    d.mkdir()
    for c in CHUNKS:
        (d / f"{c['video_id']}.jsonl").write_text(
            json.dumps(c, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return d


def test_chunk_id_is_derived_from_passage_range(chunks_dir):
    ids = [c["chunk_id"] for c in adapters.load_chunks(chunks_dir)]
    assert ids[0] == "v1_p0001-p0004"


def test_bm25_finds_the_right_chunk_and_skips_sponsor(chunks_dir):
    bm25 = adapters.BM25Retriever(adapters.load_chunks(chunks_dir))
    top = bm25("combien de rÃ©cupÃ©ration entre les sÃ©ries", 3)
    assert top[0]["video_id"] == "v1"
    assert all(c["video_id"] != "v3" for c in top)


def test_retrieval_suite_metrics_and_failures(chunks_dir):
    bm25 = adapters.BM25Retriever(adapters.load_chunks(chunks_dir))
    cases = [
        RetrievalCase(
            "Q1",
            "temps de rÃ©cupÃ©ration entre sÃ©ries",
            True,
            spans=[Span("v1", 10, 20)],
        ),
        RetrievalCase("Q2", "deload volume charges", True, spans=[Span("v9", 0, 10)]),
        RetrievalCase("Q3", "prix abonnement", False),
    ]
    res = runner.evaluate_retrieval(cases, bm25, ks=(1, 3), main_k=3)
    assert res.n_cases == 2 and res.n_passed == 1
    assert res.metrics["recall@1"].fmt() == "0.500 (1/2)"
    assert (
        res.failures[0].case_id == "Q2" and res.failures[0].category == "not_retrieved"
    )
    assert res.extra["n_unanswerable_skipped"] == 1


def gen_case(cid, expected, **kw):
    return GenerationCase(
        id=cid,
        request="r",
        expected_status=expected,
        profile={"days_per_week": 3, "max_session_min": 45, "equipment": ["dumbbells"]},
        **kw,
    )


def test_generation_suite_with_naive_mock(chunks_dir):
    corpus = {c["chunk_id"]: c["text"] for c in adapters.load_chunks(chunks_dir)}
    planner = adapters.make_mock_naive(adapters.load_chunks(chunks_dir))
    cases = [
        gen_case("S1", ["ok"]),
        gen_case("S006", ["blocked"]),
    ]
    res = runner.evaluate_generation(cases, planner, corpus)
    categories = {f.category for f in res.failures}
    assert {"fabricated_citation", "unsafe_release"} <= categories
    assert res.metrics["output_schema_validity"].value == 1
    assert res.n_cases == 2
    gates = {g["gate"]: g["status"] for g in runner.release_gates({"generation": res})}
    assert "fail" in gates.values() or "not measured" in gates.values()


def test_abstaining_mock_scores_abstention():
    cases = [gen_case("A", ["abstain"]), gen_case("B", ["ok"])]
    res = runner.evaluate_generation(cases, adapters.mock_abstain, corpus=None)
    assert res.metrics["abstention_accuracy"].value == 1
    assert res.metrics["false_abstention_rate"].value == 1
    assert res.metrics["citation_precision"].value is None


def test_crashing_planner_is_reported_not_raised():
    def broken(_case):
        raise RuntimeError("LLM timeout")

    res = runner.evaluate_generation([gen_case("X", ["ok"])], broken, corpus=None)
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
    report_text = (run / "report.md").read_text(encoding="utf-8")
    assert "# RepWise evaluation report" in report_text
    assert "split" not in report_text.lower()


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
    """A test planner that releases a plan when evidence is expected to be absent fails the gate."""
    monkeypatch.setattr(config, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(config, "OUT_DIR", tmp_path)
    monkeypatch.setattr(config, "FAIL_ON_GATES", True)
    monkeypatch.setattr(config, "PLANNER", "mock-naive")
    code = runner.main(["--mode", "generation", "--llm", "mock"])
    assert code == 1
