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
    bm25 = adapters.build_retriever("bm25", chunks_dir)
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

    top = adapters.build_retriever("bm25", chunks_dir)("quadriceps", 1)

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

    retrievers = adapters.build_retrievers(["bm25", "semantic", "hybrid"], chunks_dir)

    assert set(retrievers) == {"bm25", "semantic", "hybrid"}
    assert created["hybrid"] == (created["lexical"], created["semantic"])


def test_retrieval_suite_metrics_and_failures(chunks_dir):
    """Score answerable and unanswerable cases and retain retrieval failures.

    Args:
        chunks_dir: Provide the test corpus from which the BM25 retriever is built.
    """
    bm25 = adapters.build_retriever("bm25", chunks_dir)
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
    monkeypatch.setattr(
        runner.datasets,
        "load_generation_cases",
        lambda _path=None: [gen_case("G1", ["ok"])],
    )

    run = runner.run_evaluation(
        "generation", "bm25", "ollama-rag", generation_model="test-model"
    )

    assert any(
        "Planner setup failed" in warning
        for warning in run["payload"]["meta"]["warnings"]
    )
    assert run["results"]["generation"].failures[0].category == "pipeline_error"


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
        lambda: [gen_case("G1", ["ok"])],
    )
    monkeypatch.setattr(
        runner.datasets,
        "load_red_team_cases",
        lambda: [gen_case("RT001", ["ok"])],
    )
    monkeypatch.setattr(
        runner.adapters,
        "build_retrievers",
        lambda strategies, _path, *, chunks=None: {
            strategy: strategy for strategy in strategies
        },
    )
    monkeypatch.setattr(
        runner,
        "evaluate_retrieval",
        lambda _cases, strategy: runner.SuiteResult(
            "retrieval",
            n_cases=1,
            n_passed=1,
            metrics={
                "recall@5": Ratio(
                    {"bm25": 0.5, "semantic": 0.4, "hybrid": 0.8}[strategy], 1
                ),
                "mrr": Ratio(0.5, 1),
                "hit@1": Ratio(1, 1),
            },
            extra={"latency_ms_median": 1.0},
        ),
    )
    planner_calls = []
    monkeypatch.setattr(
        runner.adapters,
        "build_planner",
        lambda planner_name, _path, *, model_name, retriever_name, retriever: planner_calls.append(
            (planner_name, model_name, retriever_name, retriever)
        )
        or (lambda _case: {}),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_generation",
        lambda _cases, _planner, _corpus, llm_mode: runner.SuiteResult(
            "generation",
            n_cases=1,
            n_passed=1,
            metrics={
                "request_constraint_compliance": Ratio(1, 1),
                "output_schema_validity": Ratio(1, 1),
                "citation_id_validity": Ratio(1, 1),
            },
            extra={"latency_ms_median": 1.0},
        ),
    )

    run = runner.run_comparison(
        model_a="model-a",
        model_b="model-b",
        label="test",
    )
    assert run["payload"]["meta"]["selected_retriever"] == "hybrid"
    assert run["payload"]["meta"]["generation_ranking"] == [
        "model-a + RAG",
        "model-b + RAG",
    ]
    assert run["payload"]["meta"]["best_rag_setup"] == {
        "generation_arm": "model-a + RAG",
        "generation_model": "model-a",
        "retriever": "hybrid",
    }
    assert run["payload"]["meta"]["generation_baseline"] == "model-a without RAG"
    assert run["payload"]["suites"]["red-team/best-setup"]["n_cases"] == 1
    assert len(planner_calls) == 3
    assert {call[2] for call in planner_calls} == {"hybrid"}
    assert sum(call[1] == "model-a" for call in planner_calls) == 2
    assert len({id(call[3]) for call in planner_calls}) == 1


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
