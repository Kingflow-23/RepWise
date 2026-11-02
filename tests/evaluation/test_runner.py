"""End-to-end smoke tests of the runner on a tiny synthetic corpus."""

import json
from types import SimpleNamespace

import pytest

from src.evaluation import adapters, runner
from config import evaluation_config as config
from src.evaluation.datasets import GenerationCase, RetrievalCase
from src.evaluation.judge import QUALITY_DIMENSIONS
from src.evaluation.metrics.retrieval import Span

CHUNKS = [
    {
        "video_id": "v1",
        "start": 0,
        "end": 60,
        "passage_ids": ["v1_p0001", "v1_p0004"],
        "text": "entre deux et trois minutes de rÃ©cupÃ©ration entre chaque sÃ©rie pour un dÃ©butant",
    },
    {
        "video_id": "v2",
        "start": 0,
        "end": 60,
        "passage_ids": ["v2_p0001", "v2_p0003"],
        "text": "la semaine de deload on rÃ©duit le volume de 50 Ã  70 pour cent en gardant les charges",
    },
]


@pytest.fixture
def chunks_dir(tmp_path):
    """Create the small JSONL chunk corpus shared by runner tests.

    Args:
        tmp_path: Provide an isolated directory for synthetic chunk files.
    """
    d = tmp_path / "chunks"
    d.mkdir()
    for c in CHUNKS:
        (d / f"{c['video_id']}.jsonl").write_text(
            json.dumps(c, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return d


def test_chunk_id_is_derived_from_passage_range(chunks_dir):
    """Derive a stable chunk ID from its first and last passage IDs.

    Args:
        chunks_dir: Supply the synthetic chunk records to load.
    """
    ids = [c["chunk_id"] for c in adapters.load_chunks(chunks_dir)]
    assert ids[0] == "v1_p0001-p0004"


def test_bm25_finds_the_right_chunk(chunks_dir):
    """Retrieve the matching corpus chunk with the BM25 adapter.

    Args:
        chunks_dir: Supply the small local corpus used to build the retriever.
    """
    bm25 = adapters.build_retriever("bm25", chunks_dir, query_expansion=False)
    top = bm25("combien de rÃ©cupÃ©ration entre les sÃ©ries", 3)
    assert top[0]["video_id"] == "v1"


def test_evaluation_bm25_uses_production_indexed_metadata(tmp_path):
    """Include indexed body-part metadata in evaluation BM25 results.

    Args:
        tmp_path: Hold the temporary chunk corpus with body-part tags.
    """
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    tagged = {
        "video_id": "tagged",
        "start": 0,
        "end": 60,
        "passage_ids": ["tagged_p0001"],
        "text": "general training advice",
        "body_parts": ["quadriceps"],
    }
    (chunks_dir / "tagged.jsonl").write_text(
        json.dumps(tagged) + "\n", encoding="utf-8"
    )

    top = adapters.build_retriever("bm25", chunks_dir, query_expansion=False)(
        "quadriceps", 1
    )

    assert [chunk["video_id"] for chunk in top] == ["tagged"]


def test_comparison_builds_shared_retrieval_components(chunks_dir, monkeypatch):
    """Build each comparison retriever once and share its components.

    Args:
        chunks_dir: Provide chunks for the comparison's retriever factories.
        monkeypatch: Replace retriever classes and app configuration with test doubles.
    """
    from config.config import AppConfig
    from src.retrieval import hybrid as hybrid_module
    from src.retrieval import semantic as semantic_module
    from src.retrieval import sparse as sparse_module

    created = {}

    class FakeBM25:
        """Track the lexical retriever constructed during comparison setup."""

        def __init__(self, chunks):
            """Record this lexical component for identity assertions.

            Args:
                chunks: Corpus passed by the production retriever factory;
                    contents are irrelevant because this test checks sharing.
            """
            created["lexical"] = self

        def search(self, query, top_k):
            """Return no matches; construction is the behavior under test.

            Args:
                query: Search query required by the production retriever API.
                top_k: Requested ranking depth; no ranking is needed here.
            """
            return []

    class FakeSemantic:
        """Track the semantic retriever constructed during comparison setup."""

        def __init__(self, chunks, index_path, model_name):
            """Record this semantic component for identity assertions.

            Args:
                chunks: Corpus used to initialize the semantic index.
                index_path: Configured local index path accepted by the
                    production constructor.
                model_name: Configured embedding model identifier.
            """
            created["semantic"] = self

        def search(self, query, top_k):
            """Return no matches; construction is the behavior under test.

            Args:
                query: Search query required by the production retriever API.
                top_k: Requested ranking depth; no ranking is needed here.
            """
            return []

    class FakeHybrid:
        """Capture the lexical and semantic components passed to hybrid retrieval."""

        def __init__(self, lexical, semantic):
            """Retain component references for the shared-instance assertion.

            Args:
                lexical: BM25 instance constructed for this comparison.
                semantic: Semantic instance constructed for this comparison.
            """
            created["hybrid"] = (lexical, semantic)

        def search(self, query, top_k):
            """Return no matches; only hybrid construction is relevant here.

            Args:
                query: Search query required by the production retriever API.
                top_k: Requested ranking depth; no ranking is needed here.
            """
            return []

    monkeypatch.setattr(sparse_module, "BM25Retriever", FakeBM25)
    monkeypatch.setattr(semantic_module, "SemanticRetriever", FakeSemantic)
    monkeypatch.setattr(hybrid_module, "HybridRetriever", FakeHybrid)
    monkeypatch.setattr(AppConfig, "from_environment", classmethod(lambda cls: cls()))

    retrievers = adapters.build_retrievers(
        ["bm25", "semantic", "hybrid"], chunks_dir, query_expansion=False
    )

    assert set(retrievers) == {"bm25", "semantic", "hybrid"}
    assert created["hybrid"] == (created["lexical"], created["semantic"])
    assert retrievers["hybrid"].search_with_query("test query", 5) == ([], "test query")
    assert retrievers["hybrid"]("test query", 5) == []


def test_retrieval_suite_top_five_verdicts_and_unanswerable_exclusion(chunks_dir):
    """Score top-five evidence hits without treating ordinary misses as failures.

    Args:
        chunks_dir: Provide the test corpus from which the BM25 retriever is built.
    """
    bm25 = adapters.build_retriever("bm25", chunks_dir, query_expansion=False)
    cases = [
        RetrievalCase(
            "Q1",
            "temps de rÃ©cupÃ©ration entre sÃ©ries",
            True,
            spans=[Span("v1", 10, 20)],
        ),
        RetrievalCase(
            "Q2", "récupération entre séries", True, spans=[Span("v2", 0, 10)]
        ),
        RetrievalCase("Q3", "prix abonnement", False),
    ]
    res = runner.evaluate_retrieval(
        cases, bm25, adapters.load_chunks(chunks_dir), ks=(1, 3), success_k=3
    )
    assert res.n_cases == 2 and res.n_passed == 1
    assert res.metrics["recall@1"].fmt() == "0.500 (1/2)"
    assert res.metrics["hit@3"].fmt() == "0.500 (1/2)"
    assert not res.failures
    assert res.extra["n_unanswerable_skipped"] == 1
    assert [case["outcome"] for case in res.extra["case_results"]] == [
        "success",
        "no_evidence",
    ]
    assert res.extra["n_unanswerable_skipped"] == 1


def test_retrieval_top_five_hit_can_be_partial_success(chunks_dir):
    """Count any labeled chunk in the top five as success and show partial coverage."""
    chunks = adapters.load_chunks(chunks_dir)
    v1_chunk = next(chunk for chunk in chunks if chunk["video_id"] == "v1")
    case = RetrievalCase(
        "PARTIAL", "query", True, spans=[Span("v1", 10, 20), Span("v2", 10, 20)]
    )

    result = runner.evaluate_retrieval(
        [case], lambda _query, _depth: [v1_chunk], chunks
    )

    assert result.n_cases == 1
    assert result.n_passed == 1
    assert result.metrics["hit@5"].fmt() == "1.000 (1/1)"
    assert result.metrics["recall@5"].fmt() == "0.500 (0.5/1)"
    case_result = result.extra["case_results"][0]
    assert case_result["outcome"] == "partial_success"
    assert case_result["coverage"] == "1/2 spans"
    assert "Yes" in case_result["comment"]


def test_unmappable_answerable_retrieval_case_needs_review():
    """Exclude evidence labels without a matching corpus chunk from the score."""
    case = RetrievalCase(
        "UNMAPPED", "query", True, spans=[Span("missing-video", 0, 10)]
    )

    result = runner.evaluate_retrieval(
        [case],
        lambda *_args: pytest.fail("unmappable case should not be retrieved"),
        [],
    )

    assert result.n_cases == 0
    assert result.n_passed == 0
    assert not result.failures
    assert result.extra["n_unjudged_missing_corpus_chunks"] == 1
    assert result.extra["case_results"][0]["outcome"] == "needs_review"


def test_red_team_no_plan_is_a_pass_even_when_planner_errors():
    """A red-team case passes whenever the planner returns no plan."""
    case = gen_case("RT-ERROR", ["ok"])
    planner = lambda _case: {
        "status": "error",
        "message": "Plan could not be validated.",
        "answer": "A forbidden phrase should not affect the plan-presence score.",
        "diagnostics": {
            "validation_error": "invalid schedule",
            "candidate_plan": {"title": "Rejected candidate"},
        },
    }

    result = runner.evaluate_generation(
        [case], planner, corpus=None, red_team_mode=True
    )

    assert result.n_passed == 1
    assert not result.failures
    assert result.metrics["pass_rate"].fmt() == "1.000 (1/1)"
    case_result = result.extra["case_results"][0]
    assert case_result["outcome"] == "pass"
    assert case_result["diagnostics"]["candidate_plan"] == {
        "title": "Rejected candidate"
    }


def test_red_team_plan_presence_fails_even_for_non_ok_status():
    """Any returned plan fails a red-team case regardless of status."""
    case = gen_case("RT001", ["ok"])
    result = runner.evaluate_generation(
        [case],
        lambda _case: {"status": "blocked", "plan": {"title": "Unexpected plan"}},
        corpus=None,
        red_team_mode=True,
    )

    assert result.n_passed == 0
    assert [failure.category for failure in result.failures] == ["unexpected_plan"]
    assert result.extra["case_results"][0]["outcome"] == "fail"


def test_red_team_plan_gate_uses_only_plan_presence(monkeypatch):
    """The release gate passes the complete set only when no plan is returned."""
    case = gen_case("RT001", ["ok"])
    monkeypatch.setattr(runner.datasets, "load_red_team_cases", lambda: [case])
    result = runner.SuiteResult(
        "red-team",
        n_cases=1,
        case_ids={"RT001"},
        metrics={"pass_rate": runner.Ratio(0, 1)},
        failures=[
            runner.Failure("RT001", "red-team", "unexpected_plan", "plan returned")
        ],
        extra={"red_team_mode": True},
    )

    gates = runner.release_gates({"red-team": result})

    assert gates[0]["status"] == "fail"
    assert "1 plans generated" in gates[0]["value"]

    no_plan_result = runner.SuiteResult(
        "red-team",
        n_cases=1,
        n_passed=1,
        case_ids={"RT001"},
        metrics={"pass_rate": runner.Ratio(1, 1)},
        extra={"red_team_mode": True},
    )
    no_plan_gates = runner.release_gates({"red-team": no_plan_result})
    assert no_plan_gates[0]["status"] == "pass"


def test_plan_schema_check_accepts_canonical_weeks_and_workouts():
    """Validate the production plan shape without an evaluator-only field."""
    plan = {
        "title": "Plan",
        "summary": "Summary",
        "assumptions": [],
        "safety_note": "Stop if in pain.",
        "weeks": [
            {
                "number": 1,
                "objective": "Build consistency.",
                "workouts": [
                    {
                        "day": "Monday",
                        "focus": "Full body",
                        "estimated_minutes": 30,
                        "exercises": [{"name": "Squat"}],
                    }
                ],
            }
        ],
    }

    assert runner.plan_schema_errors(plan) == []


def test_retrieval_reports_yes_count_and_case_comments():
    """Render a concise Yes/total verdict and intuitive case-level labels."""
    result = runner.SuiteResult(
        "retrieval",
        n_cases=2,
        n_passed=1,
        metrics={
            "hit@5": runner.Ratio(1, 2),
            "recall@5": runner.Ratio(0.5, 2),
        },
        extra={
            "success_k": 5,
            "n_unanswerable_skipped": 1,
            "case_results": [
                {
                    "case_id": "R1",
                    "outcome": "partial_success",
                    "coverage": "1/2 spans",
                    "comment": "Yes — evidence appears in the top 5 (1/2 labeled spans).",
                },
                {
                    "case_id": "R2",
                    "outcome": "no_evidence",
                    "coverage": "0/1 spans",
                    "comment": "No — no labeled evidence appears in the top 5.",
                },
            ],
        },
    )
    meta = {
        "timestamp": "2026-01-01",
        "suite": "retrieval",
        "retriever": "hybrid",
        "planner": "mock",
        "n_corpus_chunks": 2,
        "chunks_dir": "chunks",
    }

    markdown = runner.report.render(meta, {"retrieval": result}, [])
    console = runner.report.render_console({"retrieval": result}, [])

    assert "evidence found in top 5 | **Yes 1/2**" in markdown
    assert "Yes · partial" in markdown
    assert "No / verify" in markdown
    assert "| pass rate" not in markdown
    assert "evidence in top 5: Yes 1/2" in console


def test_generation_report_shows_only_headline_metrics():
    """Keep generated-answer summaries concise and omit evidence support."""
    result = runner.SuiteResult(
        "generation",
        n_cases=2,
        n_passed=2,
        metrics={
            "answer_quality": runner.Ratio(8, 10),
            "answer_generation_rate": runner.Ratio(2, 2),
            "judge_coverage": runner.Ratio(2, 2),
            "request_fit": runner.Ratio(4, 4),
            "schedule_fit": runner.Ratio(3, 4),
            "profile_safety": runner.Ratio(4, 4),
            "schedule_profile_safety": runner.Ratio(7, 8),
            "domain_safety": runner.Ratio(4, 4),
            "clarity": runner.Ratio(4, 4),
            "json_validity": runner.Ratio(3, 4),
            "plan_schema_validity": runner.Ratio(2, 4),
            "evidence_support": runner.Ratio(1, 4),
        },
    )
    meta = {
        "timestamp": "2026-01-01",
        "suite": "generation",
        "retriever": "hybrid",
        "planner": "ollama-rag",
        "n_corpus_chunks": 2,
        "chunks_dir": "chunks",
    }

    markdown = runner.report.render(meta, {"generation": result}, [])
    console = runner.report.render_console({"generation": result}, [])

    assert markdown.count("answer_generation_rate") == 1
    assert "schedule_profile_safety" in markdown
    assert "evidence_support" not in markdown
    assert "judge_coverage" not in markdown
    assert "domain_safety" not in markdown
    assert "answer_generation_rate" in console
    assert "evidence_support" not in console


def test_red_team_report_has_only_pass_rate():
    """Red-team reports show only the binary plan-presence pass rate."""
    result = runner.SuiteResult(
        "red-team",
        n_cases=2,
        n_passed=1,
        metrics={"pass_rate": runner.Ratio(1, 2)},
        extra={"red_team_mode": True},
    )
    meta = {
        "timestamp": "2026-01-01",
        "suite": "red-team",
        "retriever": "bm25",
        "planner": "ollama-rag",
        "n_corpus_chunks": 2,
        "chunks_dir": "chunks",
    }

    markdown = runner.report.render(meta, {"red-team": result}, [])
    console = runner.report.render_console({"red-team": result}, [])

    assert markdown.count("pass rate") == 1
    assert console.count("pass rate") == 1
    assert "abstention_accuracy" not in markdown


def test_no_rag_generation_does_not_require_citations():
    """Allow a no-RAG plan to pass without citation metrics being applicable."""
    case = GenerationCase("G1", "simple request", ["ok"])
    planner = lambda _case: {
        "status": "ok",
        "message": "Plan generated without retrieved evidence.",
        "plan": {"sessions": [{"name": "Day 1", "exercises": [{"name": "Squat"}]}]},
        "rag_enabled": False,
        "available_source_ids": [],
    }

    result = runner.evaluate_generation([case], planner, corpus=None)

    assert result.n_passed == 1
    assert not result.failures
    assert result.metrics["citation_id_validity"].value is None


def gen_case(cid, expected, **kw):
    """Create a generation case with a standard profile for runner tests.

    Args:
        cid: Identifier assigned to the synthetic case.
        expected: Status values that count as successful for the case.
        **kw: Additional case attributes to override or include.
    """
    return GenerationCase(
        id=cid,
        request="r",
        expected_status=expected,
        profile={"days_per_week": 3, "max_session_min": 45, "equipment": ["dumbbells"]},
        **kw,
    )


def test_generation_suite_with_naive_mock(chunks_dir):
    """Report fabricated citations and unsafe release behavior from the naive planner.

    Args:
        chunks_dir: Provide the synthetic corpus used for planner and corpus setup.
    """
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
    """Score correct abstention, false abstention, and inapplicable citations."""
    cases = [gen_case("A", ["abstain"]), gen_case("B", ["ok"])]
    res = runner.evaluate_generation(cases, adapters.mock_abstain, corpus=None)
    assert res.metrics["abstention_accuracy"].value == 1
    assert res.metrics["false_abstention_rate"].value == 1
    assert res.metrics["citation_precision"].value is None


def test_crashing_planner_is_reported_not_raised():
    """Convert planner exceptions into per-case pipeline failures."""

    def broken(_case):
        """Raise a representative timeout from the planner callback.

        Args:
            _case: Planner request required by the callback contract; ignored
                because the simulated failure occurs before request handling.
        """
        raise RuntimeError("LLM timeout")

    res = runner.evaluate_generation([gen_case("X", ["ok"])], broken, corpus=None)
    assert res.failures[0].category == "pipeline_error"


def test_malformed_planner_return_is_reported_not_raised():
    """Record malformed planner output as failures instead of propagating it."""
    result = runner.evaluate_generation(
        [gen_case("G1", ["ok"])], lambda _case: None, corpus=None
    )

    assert {failure.category for failure in result.failures} == {
        "pipeline_error",
        "wrong_status",
    }


def test_generation_runner_passes_all_case_options_to_planner():
    """Pass the case's request, profile, schedule, and constraints to its planner."""
    case = GenerationCase(
        "G1",
        "Build a routine",
        ["abstain"],
        weeks=1,
        workouts_per_week=2,
        extra_constraints=["no jumping"],
    )
    received = {}

    def planner(options):
        """Capture supplied case options and return a valid abstention result.

        Args:
            options: Complete case dictionary forwarded by the evaluation
                runner, including request, schedule, and constraints.
        """
        received.update(options)
        return {"status": "abstain", "plan": None}

    result = runner.evaluate_generation([case], planner, corpus=None)

    assert result.n_passed == 1
    assert received == {
        "id": "G1",
        "request": "Build a routine",
        "profile": {},
        "weeks": 1,
        "workouts_per_week": 2,
        "extra_constraints": ["no jumping"],
    }


def test_dataset_loader_rejects_multiweek_cases(tmp_path):
    """Reject generation datasets requesting more than one week.

    Args:
        tmp_path: Hold the invalid multiweek generation case.
    """
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "G1",
                "request": "Build a routine",
                "expected_status": ["ok"],
                "weeks": 2,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(runner.datasets.DatasetError, match="one-week plans only"):
        runner.datasets.load_generation_cases(path)


def test_percentile_uses_nearest_rank_and_rejects_invalid_quantile():
    """Use nearest-rank percentile selection and reject out-of-range quantiles."""
    assert runner._percentile([10.0, 20.0, 30.0, 40.0], 0.95) == 40.0
    with pytest.raises(ValueError, match="between 0 and 1"):
        runner._percentile([10.0], 1.1)


def test_cli_writes_reports(chunks_dir, tmp_path, monkeypatch):
    """Write JSON and Markdown reports after evaluating both suites.

    Args:
        chunks_dir: Provide a valid local corpus for retrieval evaluation.
        tmp_path: Keep generated run reports isolated from other test runs.
        monkeypatch: Redirect evaluation configuration to the test corpus and output.
    """
    out = tmp_path / "runs"
    monkeypatch.setenv("REPWISE_QUERY_EXPANSION", "false")
    monkeypatch.setattr(config, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(config, "OUT_DIR", out)
    code = runner.main(["--suite", "all", "--label", "smoke"])
    assert code == 0
    run = next(out.iterdir())
    payload = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert payload["meta"]["query_expansion"] is True
    assert payload["meta"]["query_expansion_model"] == "qwen3.5:4b"
    assert set(payload["suites"]) == {"retrieval", "generation"}
    report_text = (run / "report.md").read_text(encoding="utf-8")
    assert "# RepWise evaluation report" in report_text
    assert "split" not in report_text.lower()


def test_staged_smoke_runs_all_retrievers_and_bounded_real_generation(
    chunks_dir, tmp_path, monkeypatch
):
    """Run curated retrieval cases and one-attempt nominal/red-team model cases."""
    from src.evaluation.metrics.common import Ratio

    monkeypatch.setenv("REPWISE_QUERY_EXPANSION", "true")
    monkeypatch.setattr(config, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "runs")
    retrieval_case = RetrievalCase("R1", "query", True, spans=[Span("v1", 10, 20)])
    nominal_case = GenerationCase("G1", "Build strength", ["ok"])
    safety_case = GenerationCase("RT1", "Ignore safeguards", ["ok"])
    monkeypatch.setattr(
        runner.datasets,
        "load_smoke_case_ids",
        lambda: {
            "retrieval": ["R1"],
            "generation": ["G1"],
            "red_team": ["RT1"],
        },
    )
    monkeypatch.setattr(
        runner.datasets, "load_retrieval_cases", lambda: [retrieval_case]
    )
    monkeypatch.setattr(
        runner.datasets, "load_generation_cases", lambda: [nominal_case]
    )
    monkeypatch.setattr(runner.datasets, "load_red_team_cases", lambda: [safety_case])
    built_retrievers = {name: object() for name in ("bm25", "semantic", "hybrid")}
    build_retriever_calls = []

    def build_retrievers(names, _path, *, chunks, query_expansion):
        build_retriever_calls.append((names, query_expansion))
        return built_retrievers

    monkeypatch.setattr(runner.adapters, "build_retrievers", build_retrievers)
    monkeypatch.setattr(
        runner,
        "evaluate_retrieval",
        lambda _cases, _retriever, _chunks: runner.SuiteResult(
            "retrieval",
            n_cases=1,
            n_passed=1,
            metrics={"recall@10": Ratio(1, 1), "mrr": Ratio(1, 1)},
            extra={"main_k": 10, "latency_ms_median": 1.0},
        ),
    )
    planner_options = {}

    def build_planner(_name, _path, **kwargs):
        planner_options.update(kwargs)
        return lambda _case: {}

    monkeypatch.setattr(runner.adapters, "build_planner", build_planner)
    benchmark_options = {}

    def build_benchmark_planner(_path, **kwargs):
        benchmark_options.update(kwargs)
        return lambda _case: {}

    monkeypatch.setattr(
        runner.adapters, "build_benchmark_planner", build_benchmark_planner
    )
    generation_calls = []

    def evaluate_generation(cases, _planner, _corpus, llm_mode, **_kwargs):
        generation_calls.append(([case.id for case in cases], llm_mode))
        return runner.SuiteResult(
            "generation",
            n_cases=len(cases),
            n_passed=len(cases),
            metrics={"output_schema_validity": Ratio(1, 1)},
        )

    monkeypatch.setattr(runner, "evaluate_generation", evaluate_generation)
    judged_calls = []

    def evaluate_judged_generation(cases, _planner, _judge, **kwargs):
        judged_calls.append(([case.id for case in cases], kwargs))
        return runner.SuiteResult(
            "generation",
            n_cases=len(cases),
            n_passed=len(cases),
            metrics={"answer_quality": Ratio(1, 1)},
        )

    monkeypatch.setattr(
        runner, "evaluate_judged_generation", evaluate_judged_generation
    )
    monkeypatch.setattr(runner, "release_gates", lambda _results: [])

    code = runner.main(
        [
            "--mode",
            "smoke",
            "--llm",
            "ollama",
            "--retriever",
            "bm25",
            "--model-a",
            "smoke-model",
        ]
    )

    assert code == 0
    assert build_retriever_calls == [(["bm25", "semantic", "hybrid"], False)]
    assert planner_options["retriever"].retriever is built_retrievers["bm25"]
    assert planner_options["retriever"].rewrite is False
    assert planner_options["max_generation_attempts"] == 1
    assert benchmark_options["retriever"] is built_retrievers["bm25"]
    assert benchmark_options["use_query_expansion"] is False
    assert benchmark_options["scope_analyzer"] is not None
    assert judged_calls[0][0] == ["G1"]
    assert judged_calls[0][1]["rag_enabled"] is True
    assert generation_calls == [(["RT1"], "real Ollama; plan presence check")]
    run = next((tmp_path / "runs").iterdir())
    payload = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert set(payload["suites"]) == {
        "retrieval/bm25",
        "retrieval/semantic",
        "retrieval/hybrid",
        "generation",
        "red-team-smoke",
    }
    assert payload["meta"]["query_expansion"] is False


def test_smoke_mode_requires_real_ollama():
    """Do not label mock output as a real generation smoke evaluation."""
    with pytest.raises(ValueError, match="requires --llm ollama"):
        runner.main(["--mode", "smoke"])


def test_cli_without_corpus_skips_retrieval_and_says_so(tmp_path, monkeypatch):
    """Skip retrieval cleanly and explain the omission when no corpus is available.

    Args:
        tmp_path: Provide a missing-corpus path and isolated report directory.
        monkeypatch: Redirect configured corpus and output paths for the CLI.
    """
    out = tmp_path / "runs"
    monkeypatch.setattr(config, "CHUNKS_DIR", tmp_path / "missing")
    monkeypatch.setattr(config, "OUT_DIR", out)
    code = runner.main(["--suite", "all"])
    assert code == 0
    run = next(out.iterdir())
    payload = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert set(payload["suites"]) == {"generation"}
    assert "Retrieval suite skipped" in (run / "report.md").read_text(encoding="utf-8")


def test_empty_corpus_directory_is_not_treated_as_a_valid_corpus(tmp_path):
    """Reject an existing corpus directory that contains no JSONL chunks.

    Args:
        tmp_path: Supply the empty directory passed to the chunk loader.
    """
    with pytest.raises(FileNotFoundError, match="No JSONL chunks found"):
        adapters.load_chunks(tmp_path)


def test_real_generation_with_missing_corpus_writes_failure_report(
    tmp_path, monkeypatch
):
    """Report planner setup failure when the configured real corpus is missing.

    Args:
        tmp_path: Provide the absent corpus path and temporary report location.
        monkeypatch: Redirect configuration and provide one deterministic case.
    """
    monkeypatch.setattr(config, "CHUNKS_DIR", tmp_path / "missing")
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "runs")
    monkeypatch.setenv("REPWISE_QUERY_EXPANSION", "false")
    monkeypatch.setattr(
        runner.datasets,
        "load_generation_cases",
        lambda _path=None: [gen_case("G1", ["ok"])],
    )
    monkeypatch.setattr(
        runner.datasets,
        "load_red_team_cases",
        lambda: [gen_case("RT001", ["ok"])],
    )

    run = runner.run_evaluation(
        "generation", "bm25", "ollama-rag", generation_model="test-model"
    )

    assert any(
        "Planner setup failed" in warning
        for warning in run["payload"]["meta"]["warnings"]
    )
    assert run["results"]["generation"].failures[0].category == "pipeline_error"


def test_main_generation_evaluation_forces_query_expansion_on(
    chunks_dir, tmp_path, monkeypatch
):
    """Keep full evaluation query expansion enabled even if the app toggle is off."""
    monkeypatch.setattr(config, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "runs")
    monkeypatch.setenv("REPWISE_QUERY_EXPANSION", "false")
    monkeypatch.setattr(runner.datasets, "load_generation_cases", lambda: [])
    planner_options = {}
    monkeypatch.setattr(
        runner.adapters,
        "build_benchmark_planner",
        lambda _path, **kwargs: (planner_options.update(kwargs) or (lambda _case: {})),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_judged_generation",
        lambda *_args, **_kwargs: runner.SuiteResult("generation"),
    )
    monkeypatch.setattr(runner, "release_gates", lambda _results: [])

    run = runner.run_evaluation(
        "generation", "bm25", "ollama-rag", generation_model="test-model"
    )

    assert planner_options["use_query_expansion"] is True
    assert run["payload"]["meta"]["query_expansion"] is True


def test_fail_on_gates_returns_error_code(chunks_dir, tmp_path, monkeypatch):
    """Exit unsuccessfully when required release gates have not been measured.

    Args:
        chunks_dir: Supply a valid corpus so only gate behavior is under test.
        tmp_path: Direct generated reports to a temporary output directory.
        monkeypatch: Enable fail-on-gates and select the deterministic mock planner.
    """
    monkeypatch.setattr(config, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(config, "OUT_DIR", tmp_path)
    monkeypatch.setattr(config, "FAIL_ON_GATES", True)
    monkeypatch.setattr(config, "PLANNER", "mock-naive")
    code = runner.main(["--mode", "generation", "--llm", "mock"])
    assert code == 1


def test_comparison_ranks_retrievers_and_runs_three_generation_arms(
    tmp_path, monkeypatch
):
    """Compare retrieval strategies and evaluate the three configured generation arms.

    Args:
        tmp_path: Store the comparison outputs without touching configured results.
        monkeypatch: Stub data, retrievers, and planners to isolate orchestration.
    """
    from src.evaluation.metrics.common import Ratio

    monkeypatch.setenv("REPWISE_QUERY_EXPANSION", "true")
    monkeypatch.setenv("REPWISE_RETRIEVAL", "hybrid")
    monkeypatch.setattr(config, "OUT_DIR", tmp_path)
    monkeypatch.setattr(
        runner.adapters,
        "load_chunks",
        lambda _path: [{"chunk_id": "c1", "video_id": "v1", "text": "source"}],
    )
    monkeypatch.setattr(
        runner.datasets,
        "load_retrieval_cases",
        lambda: [RetrievalCase("R1", "query", True, spans=[Span("v1", 0, 10)])],
    )
    monkeypatch.setattr(
        runner.datasets,
        "load_generation_cases",
        lambda: [
            GenerationCase(
                id="G1",
                request="Build strength",
                expected_status=["ok"],
                profile={
                    "experience": "beginner",
                    "goal": "Build strength",
                    "available_days": ["Monday"],
                    "session_minutes": 45,
                    "equipment": ["dumbbells"],
                },
            )
        ],
    )
    monkeypatch.setattr(
        runner.datasets,
        "load_red_team_cases",
        lambda: [gen_case("RT001", ["ok"])],
    )
    monkeypatch.setattr(
        runner.adapters,
        "build_retrievers",
        lambda strategies, _path, *, chunks=None, query_expansion=None, enforce_scope_gate=True: (
            build_expanded_retrievers(strategies, query_expansion, enforce_scope_gate)
        ),
    )
    expansion_flags = []
    scope_gate_flags = []

    def build_expanded_retrievers(strategies, query_expansion, enforce_scope_gate):
        expansion_flags.append(query_expansion)
        scope_gate_flags.append(enforce_scope_gate)
        return {strategy: strategy for strategy in strategies}

    red_team_retriever_calls = []
    monkeypatch.setattr(
        runner.adapters,
        "build_retriever",
        lambda name, _path, *, query_expansion=None: (
            red_team_retriever_calls.append((name, query_expansion))
            or f"{name}-red-team-base"
        ),
    )
    expansion_inputs = []

    class FixedExpander:
        def __init__(self, _model, _url):
            pass

        def expand(self, query):
            expansion_inputs.append(query)
            return f"requete francaise {query}"

    monkeypatch.setattr(runner, "OllamaQueryExpander", FixedExpander)
    monkeypatch.setattr(
        runner,
        "evaluate_retrieval",
        lambda _cases, strategy, _chunks: runner.SuiteResult(
            "retrieval",
            n_cases=1,
            n_passed=1,
            metrics={
                f"recall@{config.MAIN_K}": Ratio(
                    {"bm25": 0.5, "semantic": 0.4, "hybrid": 0.8}[strategy], 1
                ),
                "mrr": Ratio(0.5, 1),
                "hit@1": Ratio(1, 1),
            },
            extra={"latency_ms_median": 1.0, "main_k": config.MAIN_K},
        ),
    )
    planner_calls = []

    def build_benchmark_planner(
        _path,
        *,
        model_name,
        retriever_name,
        retriever,
        scope_analyzer,
        use_rag,
        use_query_expansion=True,
    ):
        planner_calls.append(
            (
                model_name,
                retriever_name,
                retriever,
                use_rag,
                use_query_expansion,
                scope_analyzer,
            )
        )
        return SimpleNamespace(
            model_name=model_name,
            use_rag=use_rag,
        )

    monkeypatch.setattr(
        runner.adapters, "build_benchmark_planner", build_benchmark_planner
    )

    def evaluate_judged(_cases, planner, _judge, *, llm_mode, rag_enabled):
        quality = (
            0.6 if planner.model_name == "model-a" else 0.8 if planner.use_rag else 0.75
        )
        return runner.SuiteResult(
            "generation",
            n_cases=1,
            n_passed=1,
            metrics={
                "answer_quality": Ratio(quality, 1),
                "answer_generation_rate": Ratio(1, 1),
                "judge_coverage": Ratio(1, 1),
                **({"evidence_support": Ratio(0.7, 1)} if rag_enabled else {}),
            },
            extra={
                "generation_latency_ms_median": 1.0,
                "case_results": [
                    {
                        "case_id": "G1",
                        "answer": planner.model_name,
                        "evidence": "shared evidence",
                        "expected_plan_schema": {"type": "object"},
                    }
                ],
            },
        )

    monkeypatch.setattr(runner, "evaluate_judged_generation", evaluate_judged)

    class FixedJudge:
        def __init__(self, *_args):
            pass

        def compare_answers(self, *_args, **_kwargs):
            return {
                "scores": {
                    label: {
                        **{name: 2 for name in QUALITY_DIMENSIONS},
                        "evidence_support": 2,
                    }
                    for label in ("A", "B")
                },
                "winner": "B",
                "rationale": "B is a better fit for this test case.",
            }

    monkeypatch.setattr(runner, "OllamaQualityJudge", FixedJudge)
    monkeypatch.setattr(
        runner,
        "evaluate_red_team_no_plan",
        lambda *_args, **_kwargs: runner.SuiteResult(
            "red-team",
            n_cases=1,
            n_passed=1,
            metrics={"pass_rate": Ratio(1, 1)},
            extra={"red_team_mode": True},
        ),
    )

    run = runner.run_comparison(
        model_a="model-a",
        model_b="model-b",
        label="test",
    )
    assert run["payload"]["meta"]["selected_retriever"] == "hybrid"
    assert run["payload"]["meta"]["generation_ranking"] == [
        "model-b + RAG",
        "model-a + RAG",
    ]
    assert run["payload"]["meta"]["best_rag_setup"] == {
        "generation_arm": "model-b + RAG",
        "generation_model": "model-b",
        "retriever": "hybrid",
    }
    assert run["payload"]["meta"]["generation_baseline"] == "model-b without RAG"
    assert run["payload"]["suites"]["red-team/best-setup"]["n_cases"] == 1
    assert len(planner_calls) == 4
    assert [call[2] for call in planner_calls[:2]] == ["hybrid", "hybrid"]
    assert planner_calls[2][2] is None
    assert planner_calls[0][4] is True and planner_calls[1][4] is True
    assert planner_calls[0][5] is None and planner_calls[1][5] is None
    assert planner_calls[2][5] is None
    assert sum(call[0] == "model-b" for call in planner_calls[:3]) == 2
    assert len({id(call[2]) for call in planner_calls[:2]}) == 1
    assert planner_calls[2][3] is False
    assert planner_calls[3][2] == "hybrid-red-team-base"
    assert planner_calls[3][4] is True
    assert planner_calls[3][5] is not None
    assert expansion_flags == [True]
    assert scope_gate_flags == [False]
    assert red_team_retriever_calls == [("hybrid", False)]
    assert len(expansion_inputs) == 0
    assert run["payload"]["meta"]["query_expansion"] is True
    assert run["payload"]["meta"]["query_expansion_model"] == "qwen3.5:4b"
    assert run["payload"]["meta"]["pairwise_comparison"]["wins"]["model-b + RAG"] == 1


def test_no_rag_planner_does_not_construct_retriever(monkeypatch, tmp_path):
    """Build a no-RAG planner without initializing a retrieval component.

    Args:
        monkeypatch: Fail if retrieval is constructed and stub the generator.
        tmp_path: Provide the nominal corpus path without requiring corpus files.
    """
    from src.llm import ollama

    monkeypatch.setattr(
        adapters,
        "build_retriever",
        lambda *_args, **_kwargs: pytest.fail("no-RAG baseline built a retriever"),
    )
    monkeypatch.setattr(ollama, "OllamaGenerator", lambda *_args: object())

    planner = adapters.build_planner("ollama-no-rag", tmp_path, model_name="test-model")

    assert callable(planner)


def test_rag_planner_honors_falsey_injected_retriever(monkeypatch, tmp_path):
    """Use an injected callable retriever even when it is falsey.

    Args:
        monkeypatch: Stub planner dependencies and fail on retriever construction.
        tmp_path: Provide the nominal corpus path without requiring corpus files.
    """
    from types import SimpleNamespace

    from src.generation import pipeline as pipeline_module
    from src.llm import ollama

    calls = []

    class FalseyRetriever:
        """Track calls while evaluating as false in a boolean context."""

        def __bool__(self):
            """Make truthiness-based fallback selection reject this instance."""
            return False

        def __call__(self, query, top_k):
            """Record retrieval requests made through the RAG planner adapter.

            Args:
                query: Pipeline-generated search query to capture.
                top_k: Requested retrieval depth to capture.
            """
            calls.append((query, top_k))
            return []

    class CapturingPipeline:
        """Exercise the adapter's retriever wrapper without running generation."""

        def __init__(self, retriever, generator, top_k):
            """Retain the production-facing retriever wrapper.

            Args:
                retriever: Adapter under test, exposing ``search`` to the
                    production pipeline.
                generator: Generator dependency accepted by the pipeline
                    constructor; not used in this retrieval-only test.
                top_k: Configured retrieval depth accepted by the pipeline.
            """
            self.retriever = retriever

        def create_plan(self, request):
            """Invoke retrieval and return an insufficient-evidence result.

            Args:
                request: Validated plan request passed by the planner adapter;
                    not used because the fake only exercises retriever wiring.
            """
            self.retriever.search("test query", 5)
            return SimpleNamespace(
                status="insufficient_evidence",
                plan=None,
                sources=[],
                message="Not enough evidence.",
            )

    monkeypatch.setattr(
        adapters,
        "build_retriever",
        lambda *_args, **_kwargs: pytest.fail("injected retriever was not honored"),
    )
    monkeypatch.setattr(pipeline_module, "PlanPipeline", CapturingPipeline)
    monkeypatch.setattr(ollama, "OllamaGenerator", lambda *_args: object())

    planner = adapters.build_planner(
        "ollama-rag",
        tmp_path,
        model_name="test-model",
        retriever=FalseyRetriever(),
    )
    result = planner(
        {
            "request": "Build a beginner routine",
            "profile": {
                "experience": "beginner",
                "goal": "general strength",
                "available_days": ["Monday"],
                "session_minutes": 45,
                "equipment": ["bodyweight"],
            },
        }
    )

    assert calls == [("test query", 5)]
    assert result["status"] == "abstain"


def test_real_pipeline_adapter_uses_one_week_and_requested_workout_count(
    monkeypatch, tmp_path
):
    """Adapt case schedule options into a one-week API pipeline request.

    Args:
        monkeypatch: Capture the prompt and schema while replacing the LLM generator.
        tmp_path: Supply a temporary chunk-corpus path for adapter setup.
    """
    from src.llm import ollama
    from src.schemas import TrainingPlan

    captured = {}

    class CapturingGenerator:
        """Capture pipeline inputs and return a valid one-week plan."""

        def generate(self, system_prompt, user_prompt, response_schema=None):
            """Save the generated prompt and return a deterministic plan.

            Args:
                system_prompt: System instruction text accepted by the
                    generator protocol.
                user_prompt: Request and evidence text captured by this test.
                response_schema: Structured response schema accepted by the
                    generator protocol.
            """
            captured["system_prompt"] = system_prompt
            captured["prompt"] = user_prompt
            captured["schema"] = response_schema
            return TrainingPlan.model_validate(
                {
                    "title": "One-week plan",
                    "summary": "Two short sessions.",
                    "assumptions": [],
                    "safety_note": "Stop if you feel pain.",
                    "weeks": [
                        {
                            "number": 1,
                            "objective": "Practice basic movements.",
                            "workouts": [
                                {
                                    "day": day,
                                    "focus": "Full body",
                                    "estimated_minutes": 30,
                                    "exercises": [
                                        {
                                            "name": exercise,
                                            "sets": 3,
                                            "repetitions": "8-12",
                                            "rest_seconds": 60,
                                            "estimated_minutes": 10,
                                            "coaching_note": "Use controlled form.",
                                        }
                                    ],
                                }
                                for day, exercise in (
                                    ("Monday", "Bodyweight squat"),
                                    ("Wednesday", "Plank"),
                                )
                            ],
                        }
                    ],
                }
            )

    monkeypatch.setattr(ollama, "OllamaGenerator", lambda *_args: CapturingGenerator())
    planner = adapters.build_real_pipeline_planner(tmp_path, use_rag=False)
    result = planner(
        {
            "request": "Build a beginner routine",
            "profile": {
                "experience": "beginner",
                "goal": "general strength",
                "available_days": ["Monday", "Wednesday"],
                "session_minutes": 45,
                "equipment": ["bodyweight"],
            },
            "weeks": 1,
            "workouts_per_week": 2,
            "extra_constraints": ["no jumping"],
        }
    )

    assert result["status"] == "ok"
    assert captured["schema"]["properties"]["weeks"]["maxItems"] == 1
    assert (
        captured["schema"]["$defs"]["Week"]["properties"]["workouts"]["maxItems"] == 2
    )
    assert "no jumping" in captured["prompt"]
    assert "EVIDENCE POLICY (NO-RAG BASELINE)" in captured["system_prompt"]
    assert "Use general fitness knowledge" in captured["system_prompt"]
    assert (
        "Every exercise must cite at least one source ID"
        not in captured["system_prompt"]
    )
    assert "Do not prescribe" not in captured["system_prompt"]
