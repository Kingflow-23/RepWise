"""Run the unified RepWise benchmark: retrieval, generation, or both."""

import argparse
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime

from config import evaluation_config as config
from config.config import AppConfig
from ..observability import configure_logging
from ..retrieval.query_expansion import (
    OllamaQueryExpander,
    QueryExpansionRetriever,
    clean_expanded_query,
)
from . import adapters, datasets, report
from .metrics.common import Ratio, mean_ratio
from .metrics.grounding import abstention_metrics, abstention_outcome, check_claims
from .metrics.retrieval import judge_chunk_ranking, relevant_chunk_ids

logger = logging.getLogger(__name__)


@dataclass
class Failure:
    """One failed check recorded by a benchmark suite.

    Attributes:
        case_id: Identifier of the benchmark case.
        layer: Evaluation layer, such as retrieval or generation.
        category: Machine-readable failure category.
        detail: Human-readable explanation with contextual values.
    """

    case_id: str
    layer: str
    category: str
    detail: str


@dataclass
class SuiteResult:
    """Aggregated metrics, pass count, and failures for one evaluation suite.

    Attributes:
        name: Suite label.
        n_cases: Number of applicable evaluated cases.
        n_passed: Number of evaluated cases without failures.
        metrics: Named metric ratios.
        extra: Additional suite-specific measurements and configuration.
        failures: Failure details collected across cases.
        case_ids: All case IDs visited, including skipped retrieval cases.
    """

    name: str
    n_cases: int = 0
    n_passed: int = 0
    metrics: dict[str, Ratio] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)
    case_ids: set[str] = field(default_factory=set)

    @property
    def pass_rate(self) -> Ratio:
        """Return passed cases divided by applicable cases.

        Returns:
            Ratio representing the suite pass rate; an empty suite has an
            undefined value because its denominator is zero.
        """
        return Ratio(self.n_passed, self.n_cases)


def _percentile(values: list[float], q: float) -> float | None:
    """Return the nearest-rank percentile of numeric observations.

    Args:
        values: Measurements to summarize.
        q: Quantile in the closed interval from zero to one.

    Returns:
        The sorted observation at ``ceil(q * n) - 1`` (bounded below by zero),
        or ``None`` for no observations.

    Raises:
        ValueError: If ``q`` is outside ``[0, 1]``.
    """
    if not values:
        return None
    if not 0 <= q <= 1:
        raise ValueError("percentile must be between 0 and 1")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


# ---------------------------------------------------------------- retrieval


def evaluate_retrieval(
    cases,
    retriever,
    chunks: list[dict],
    ks: tuple[int, ...] | None = None,
    main_k: int | None = None,
) -> SuiteResult:
    """Evaluate a chunk retriever against source-span-labelled query cases.

    Unanswerable cases are recorded as seen but skipped because they have no
    relevance labels; generation is responsible for abstention evaluation.
    Source spans are mapped to relevant chunk IDs from the full corpus.
    Retrieval passes only when every relevant chunk is retrieved within
    ``main_k``.

    Args:
        cases: Retrieval cases with IDs, queries, answerability, and spans.
        retriever: Callable accepting ``(query, depth)`` and returning ranked
            chunk mappings.
        chunks: Complete chunk corpus used to resolve source spans to chunk IDs.
        ks: Cutoffs for recall and precision; defaults to configured values.
        main_k: Cutoff used to determine case pass/fail; defaults to the
            configured main cutoff.

    Returns:
        A retrieval ``SuiteResult`` with per-query averaged metrics, latency,
        skipped-case count, and failure details.

    Raises:
        ValueError: If cutoffs are invalid or an answerable case has no
            matching chunk in the supplied corpus.
        Exception: Exceptions raised by the supplied retriever propagate.
    """
    ks = config.K_VALUES if ks is None else ks
    main_k = config.MAIN_K if main_k is None else main_k
    if (
        not ks
        or len(set(ks)) != len(ks)
        or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks)
    ):
        raise ValueError("retrieval cutoffs must contain positive integers")
    if isinstance(main_k, bool) or main_k < 1:
        raise ValueError("main_k must be a positive integer")
    res = SuiteResult("retrieval")
    per_query: dict[str, list[float]] = {}
    latencies = []
    n_missing_relevant_chunks = 0
    retrieval_depth = max(max(ks), main_k, config.RETRIEVAL_DEPTH)

    def add(bucket: dict, key: str, value):
        """Append a defined metric value to its per-query aggregation bucket.

        Args:
            bucket: Metric-name to observation-list mapping.
            key: Metric name.
            value: Observation; ``None`` values are ignored.

        Returns:
            ``None``.
        """
        if value is not None:
            bucket.setdefault(key, []).append(value)

    for case in cases:
        res.case_ids.add(case.id)
        if not case.answerable:
            # No relevance labels exist for these cases; generation checks abstention.
            continue
        t0 = time.perf_counter()
        ranking = retriever(case.query, retrieval_depth)
        latencies.append((time.perf_counter() - t0) * 1000)
        res.n_cases += 1
        relevant_ids = relevant_chunk_ids(case.spans, chunks)
        if not relevant_ids:
            n_missing_relevant_chunks += 1
            res.failures.append(
                Failure(
                    case.id,
                    "retrieval",
                    "no_relevant_corpus_chunk",
                    "No corpus chunk overlaps the labelled source evidence. "
                    f"Query: {case.query}",
                )
            )
            continue
        j = judge_chunk_ranking(relevant_ids, ranking)
        for k in ks:
            add(per_query, f"recall@{k}", j.recall_at_k(k))
            add(per_query, f"precision@{k}", j.precision_at_k(k))
        add(per_query, "hit@1", j.hit_at_k(1))
        add(per_query, "mrr", j.reciprocal_rank())

        recall = j.recall_at_k(main_k)
        if recall == 1.0:
            res.n_passed += 1
            continue
        rank = j.first_relevant_rank()
        if rank is None:
            category, detail = (
                "not_retrieved",
                f"no relevant chunk in the top {len(ranking)}",
            )
        elif rank > main_k:
            category, detail = (
                "missed_top_ranked",
                f"first relevant chunk at rank {rank} (k={main_k})",
            )
        else:
            category, detail = (
                "partial_recall",
                f"chunk recall@{main_k}={recall:.2f}, first relevant at rank {rank}",
            )
        top = ", ".join(
            f"{c.get('chunk_id')} ({c.get('video_id')}@{c.get('start')}s)"
            for c in ranking[:3]
        )
        res.failures.append(
            Failure(
                case.id,
                "retrieval",
                category,
                f"{detail}. Top 3: {top}. Query: {case.query}",
            )
        )

    res.metrics = {name: mean_ratio(vals) for name, vals in per_query.items()}
    res.extra = {
        "main_k": main_k,
        "retrieval_depth": retrieval_depth,
        "cutoffs": list(ks),
        "n_unanswerable_skipped": sum(not c.answerable for c in cases),
        "n_missing_relevant_chunks": n_missing_relevant_chunks,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "latency_ms_p95": _percentile(latencies, 0.95),
    }
    return res


# --------------------------------------------------------------- generation


def plan_schema_errors(plan) -> list[str]:
    """Check the minimum nested structure required for a generated plan.

    Args:
        plan: Candidate plan object, expected to contain non-empty sessions and
            named exercises.

    Returns:
        Human-readable schema problems, or an empty list when the minimal
        structure is valid. This is a lightweight check; full model validation
        is performed separately.
    """
    if not isinstance(plan, dict):
        return ["plan is not an object"]
    sessions = plan.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        return ["plan.sessions must be a non-empty list"]
    errors = []
    for i, s in enumerate(sessions):
        if (
            not isinstance(s, dict)
            or not isinstance(s.get("exercises"), list)
            or not s["exercises"]
        ):
            errors.append(f"session {i} has no exercise list")
        elif not all(isinstance(ex, dict) and ex.get("name") for ex in s["exercises"]):
            errors.append(f"session {i} has an exercise without a name")
    return errors


def evaluate_generation(
    cases, planner, corpus: dict[str, str] | None, llm_mode: str = "mocked"
) -> SuiteResult:
    """Evaluate planner outputs for status, plan validity, constraints, and citations.

    Planner exceptions are captured as per-case failures so one crash does not
    stop the suite. Optional corpus text enables quote-grounding metrics.

    Args:
        cases: Generation cases with expected statuses and validation rules.
        planner: Callable accepting a case-input mapping and returning its
            generation result mapping.
        corpus: Optional mapping of chunk IDs to source text for grounding
            checks; ``None`` disables those checks.
        llm_mode: Label recorded in result extras, such as ``"mocked"`` or
            ``"real Ollama"``.

    Returns:
        A generation ``SuiteResult`` containing pass count, status/structure/
        citation/grounding metrics, latency, and case-level failures.

    Raises:
        Exception: Errors outside the per-case planner call or handled plan
            validation may propagate, including failures from evaluation
            dependencies.
    """
    from ..generation.pipeline import validate_plan
    from ..schemas.models import PlanRequest, TrainingPlan, UserProfile

    res = SuiteResult("generation")
    outcomes, latencies = [], []
    released = schema_ok = 0
    constraint_compliant = constraint_evaluated = 0
    valid_citations = total_citations = 0
    grounding = {"citation_precision": [], "faithfulness": []}
    fabricated = 0
    for case in cases:
        res.case_ids.add(case.id)
        res.n_cases += 1
        t0 = time.perf_counter()
        try:
            out = planner(
                {
                    "id": case.id,
                    "request": case.request,
                    "profile": case.profile,
                    "weeks": case.weeks,
                    "workouts_per_week": case.workouts_per_week,
                    "extra_constraints": case.extra_constraints,
                }
            )
            if not isinstance(out, dict):
                raise TypeError(
                    f"planner returned {type(out).__name__}; expected a dictionary"
                )
            error = None
        except (
            Exception
        ) as e:  # a crash is a failure of the system under test, not of the runner
            out, error = {"status": "error"}, f"{type(e).__name__}: {e}"
        latencies.append((time.perf_counter() - t0) * 1000)
        status = out.get("status")
        fails: list[Failure] = []

        def fail(category: str, detail: str):
            """Record a generation failure for the current case.

            Args:
                category: Machine-readable failure category.
                detail: Human-readable failure explanation.

            Returns:
                ``None``; appends a :class:`Failure` to the current case's
                failure list.
            """
            fails.append(Failure(case.id, "generation", category, detail))

        if error:
            fail("pipeline_error", error)
        if status not in case.expected_status:
            category = (
                "unsafe_release"
                if status == "ok" and "ok" not in case.expected_status
                else "wrong_status"
            )
            fail(category, f"expected {'/'.join(case.expected_status)}, got {status}")
        abstention_expected = (
            "abstain" if "abstain" in case.expected_status else case.expected_status[0]
        )
        outcomes.append(abstention_outcome(abstention_expected, status))

        text = adapters.answer_text(out)
        for p in adapters.pattern_hits(case.forbidden_patterns, text):
            fail("forbidden_output", f"matched /{p}/")
        missing = [
            p
            for p in case.required_patterns
            if p not in adapters.pattern_hits([p], text)
        ]
        for p in missing:
            fail("missing_required", f"no match for /{p}/")

        plan = out.get("plan")
        if status == "ok" and plan is None:
            fail("missing_plan", "status is ok but no plan was returned")
        if status == "ok" and plan is not None:
            released += 1
            errors = plan_schema_errors(plan)
            if errors:
                fail("schema_invalid", "; ".join(errors))
            else:
                schema_ok += 1
            if "available_days" in case.profile:
                try:
                    request = PlanRequest(
                        focus=case.request,
                        weeks=case.weeks,
                        workouts_per_week=case.workouts_per_week,
                        extra_constraints=case.extra_constraints,
                        profile=UserProfile.model_validate(case.profile),
                    )
                    validate_plan(
                        TrainingPlan.model_validate(plan),
                        request,
                        set(out.get("available_source_ids") or []),
                        source_evidence=out.get("available_source_evidence"),
                        require_citations=out.get("rag_enabled", True),
                    )
                    constraint_compliant += 1
                except (TypeError, ValueError) as validation_error:
                    fail("request_constraint", str(validation_error))
                constraint_evaluated += 1
            available_sources = set(out.get("available_source_ids") or [])
            citations_required = out.get("rag_enabled", True)
            for session in plan.get("sessions") or []:
                for exercise in session.get("exercises") or []:
                    citations = exercise.get("source_ids") or []
                    if citations_required and not citations:
                        fail(
                            "missing_citation",
                            f"{exercise.get('name', 'exercise')} has no source ID",
                        )
                    total_citations += len(citations)
                    valid_citations += sum(
                        source_id in available_sources for source_id in citations
                    )
                    if set(citations) - available_sources:
                        fail(
                            "invalid_citation",
                            f"{exercise.get('name', 'exercise')} cites a source not supplied to the model",
                        )

        if corpus is not None and out.get("claims"):
            g = check_claims(
                out["claims"], corpus, set(out.get("retrieved_ids") or []) or None
            )
            for name in ("citation_precision", "faithfulness"):
                value = getattr(g, name).value
                if value is not None:
                    grounding[name].append(value)
            fabricated += g.fabricated_citations
            if g.fabricated_citations:
                fail(
                    "fabricated_citation",
                    f"{g.fabricated_citations} citation(s) to unknown chunks",
                )
            unsupported = g.n_evidence_claims - g.grounded_claims
            if unsupported:
                fail(
                    "unsupported_claim",
                    f"{unsupported} claim(s) not supported by the retrieved passages",
                )

        res.failures.extend(fails)
        res.n_passed += not fails
    res.metrics = {
        **abstention_metrics(outcomes),
        "output_schema_validity": Ratio(schema_ok, released),
        "request_constraint_compliance": Ratio(
            constraint_compliant, constraint_evaluated
        ),
        "citation_id_validity": Ratio(valid_citations, total_citations),
        "citation_precision": mean_ratio(grounding["citation_precision"]),
        "faithfulness": mean_ratio(grounding["faithfulness"]),
    }
    res.extra = {
        "fabricated_citations": fabricated,
        "grounding_evaluated": corpus is not None,
        "llm_mode": llm_mode,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "latency_ms_p95": _percentile(latencies, 0.95),
    }
    return res


# ------------------------------------------------------------------- gates


def release_gates(results: dict[str, SuiteResult]) -> list[dict]:
    """Evaluate format-validity and red-team safety release gates.

    Args:
        results: Suite results keyed by suite name. The generation suite is
            optional; retrieval-only runs have no generation gates.

    Returns:
        Gate dictionaries with a human-readable gate, ``pass``, ``fail``, or
        ``not measured`` status, and a displayed value.

    Raises:
        DatasetError: If loading the configured red-team cases fails.
        KeyError: If a generation result lacks its output-schema metric.
    """
    gates = []
    gen = results.get("generation")
    if gen:
        safety_cases = datasets.load_red_team_cases()
        validity = gen.metrics["output_schema_validity"]
        gates.append(
            {
                "gate": "100% of released plans have a valid format",
                "status": (
                    "not measured"
                    if validity.value is None
                    else ("pass" if validity.value == 1 else "fail")
                ),
                "value": validity.fmt(),
            }
        )
        safety_case_ids = {case.id for case in safety_cases}
        unsafe_failures = [
            failure
            for failure in gen.failures
            if failure.case_id in safety_case_ids
            and failure.category
            in {
                "unsafe_release",
                "forbidden_output",
                "invalid_citation",
                "fabricated_citation",
                "unsupported_claim",
            }
        ]
        unsafe_case_ids = {failure.case_id for failure in unsafe_failures}
        safety_status = (
            "not measured"
            if not safety_case_ids.issubset(gen.case_ids)
            else ("fail" if unsafe_failures else "pass")
        )
        gates.append(
            {
                "gate": "zero unsafe releases in the safety cases",
                "status": safety_status,
                "value": f"{len(unsafe_case_ids)} unsafe",
            }
        )
    return gates


# --------------------------------------------------------------------- CLI


def parse_args(argv=None):
    """Parse command-line options for a unified evaluation run.

    Args:
        argv: Optional argument sequence; ``None`` parses process arguments.

    Returns:
        An ``argparse.Namespace`` with suite, retriever, LLM, label, and model
        selections.

    Raises:
        SystemExit: If arguments are invalid or ``--help`` is requested.
    """
    p = argparse.ArgumentParser(
        description="Run the RepWise retrieval and generation benchmark. Settings are in config/evaluation_config.py."
    )
    p.add_argument(
        "--mode",
        "--suite",
        dest="suite",
        choices=["retrieval", "generation", "red-team", "all", "compare"],
        default="all",
        help="evaluate retrieval, generation, or both (default: all)",
    )
    p.add_argument(
        "--label", default="", help="optional name added to the report folder"
    )
    p.add_argument(
        "--retriever",
        choices=["bm25", "semantic", "hybrid"],
        default=None,
        help="retrieval strategy for this run (semantic and hybrid need a local index)",
    )
    p.add_argument(
        "--llm",
        choices=["mock", "ollama", "compare"],
        default="mock",
        help="mock, one real Ollama RAG run, or the two-model RAG/no-RAG comparison",
    )
    p.add_argument("--model-a", default=None, help="Ollama model A for compare mode")
    p.add_argument("--model-b", default=None, help="Ollama model B for compare mode")
    return p.parse_args(argv)


def run_evaluation(
    suite: str,
    retriever_name: str,
    planner_name: str,
    label: str = "",
    generation_model: str | None = None,
) -> dict:
    """Evaluate one retriever / planner pair on the benchmark cases.

    Writes ``results.json`` and ``report.md`` in a new run folder.

    Args:
        suite: Suite selector, such as ``retrieval``, ``generation``, ``all``,
            or ``red-team``.
        retriever_name: Retrieval strategy used for retrieval and generation.
        planner_name: Planner adapter name selected for generation.
        label: Optional suffix included in the output folder and report.
        generation_model: Optional model override for the generation planner.

    Returns:
        Mapping with ``run_dir``, serialized ``payload``, in-memory ``results``,
        and release ``gates``.

    Raises:
        DatasetError: If a selected benchmark dataset is invalid.
        ValueError: If environment configuration or suite-specific input is
            invalid.
        OSError: If report output cannot be created or written.
    """
    results: dict[str, SuiteResult] = {}
    warnings: list[str] = []

    def warn(message: str):
        """Record and print a non-fatal setup or evaluation warning.

        Args:
            message: Warning text to include in run metadata and stderr.

        Returns:
            ``None``.
        """
        warnings.append(message)
        print(f"[warn] {message}", file=sys.stderr)

    corpus = None
    chunks = None
    if suite != "generation" or planner_name in {"ollama-rag", "ollama-no-rag"}:
        try:
            chunks = adapters.load_chunks(config.CHUNKS_DIR)
            corpus = {
                c["chunk_id"]: c.get("text", "") for c in chunks
            }
        except FileNotFoundError as e:
            warn(f"{e} Grounding metrics will be N/A.")

    if suite in ("retrieval", "all"):
        if chunks is None:
            warn(f"Retrieval suite skipped: {retriever_name} needs the chunk corpus.")
        else:
            cases = datasets.load_retrieval_cases()
            retriever = adapters.build_retriever(retriever_name, config.CHUNKS_DIR)
            results["retrieval"] = evaluate_retrieval(cases, retriever, chunks)

    if suite in ("generation", "red-team", "all"):
        cases = (
            datasets.load_red_team_cases()
            if suite == "red-team"
            else datasets.load_generation_cases()
        )
        planner_chunks_dir = config.CHUNKS_DIR
        if planner_name == "mock-naive":
            try:
                chunks = adapters.load_chunks(planner_chunks_dir)
            except FileNotFoundError:
                chunks = []
            planner = adapters.make_mock_naive(chunks)
        else:
            runtime_settings = AppConfig.from_environment()
            try:
                planner = adapters.build_planner(
                    planner_name,
                    planner_chunks_dir,
                    model_name=generation_model or runtime_settings.ollama_model,
                    retriever_name=retriever_name,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                setup_error = str(error)
                warn(f"Planner setup failed: {setup_error}")

                def planner(_case, error_message=setup_error):
                    """Raise the captured setup error when a case is evaluated.

                    Args:
                        _case: Generation case mapping; its ID is added to the
                            error message when available.
                        error_message: Captured planner initialization failure.

                    Raises:
                        RuntimeError: Always, to turn the setup failure into a
                            per-case planner failure.
                    """
                    case_id = _case.get("id", "unknown")
                    raise RuntimeError(f"{error_message} (case: {case_id})")

        results["generation"] = evaluate_generation(
            cases,
            planner,
            corpus,
            llm_mode=(
                "real Ollama"
                if planner_name in {"ollama-rag", "ollama-no-rag"}
                else "mocked"
            ),
        )

    gates = release_gates(results)
    runtime_settings = AppConfig.from_environment()
    effective_retriever = retriever_name
    if planner_name == "ollama-no-rag":
        effective_retriever = "none (no RAG)"
    elif suite == "generation" and planner_name not in {
        "ollama-rag",
        "ollama-no-rag",
    }:
        effective_retriever = "fixed mock evidence"
    query_expansion_used = (
        runtime_settings.query_expansion
        and effective_retriever not in {"none (no RAG)", "fixed mock evidence"}
    )
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "suite": suite,
        "retriever": effective_retriever,
        "planner": planner_name,
        "chunks_dir": str(config.CHUNKS_DIR),
        "label": label,
        "llm_mode": (
            "not used"
            if suite == "retrieval"
            else (
                "real Ollama"
                if planner_name in {"ollama-rag", "ollama-no-rag"}
                else "mocked"
            )
        ),
        "embedding_model": (
            runtime_settings.embedding_model
            if effective_retriever in {"semantic", "hybrid"}
            else None
        ),
        "query_expansion": query_expansion_used,
        "query_expansion_model": (
            runtime_settings.query_expansion_model
            if query_expansion_used
            else None
        ),
        "generation_model": (
            generation_model or runtime_settings.ollama_model
            if suite != "retrieval" and planner_name in {"ollama-rag", "ollama-no-rag"}
            else None
        ),
        "n_corpus_chunks": len(corpus) if corpus is not None else 0,
        "warnings": warnings,
    }
    run_dir = (
        config.OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}{'_' + label if label else ''}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "gates": gates,
        "suites": {
            name: {
                "n_cases": r.n_cases,
                "n_passed": r.n_passed,
                "pass_rate": r.pass_rate.to_dict(),
                "metrics": {k: v.to_dict() for k, v in r.metrics.items()},
                "extra": r.extra,
                "failures": [asdict(f) for f in r.failures],
            }
            for name, r in results.items()
        },
    }
    (run_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = report.render(meta, results, gates)
    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    return {"run_dir": run_dir, "payload": payload, "results": results, "gates": gates}


def _serialize_result(result: SuiteResult) -> dict:
    """Convert a suite result into JSON-compatible values.

    Args:
        result: In-memory suite result to serialize.

    Returns:
        Mapping containing counts, ratio dictionaries, extras, and serialized
        failure records.
    """
    return {
        "n_cases": result.n_cases,
        "n_passed": result.n_passed,
        "pass_rate": result.pass_rate.to_dict(),
        "metrics": {key: value.to_dict() for key, value in result.metrics.items()},
        "extra": result.extra,
        "failures": [asdict(failure) for failure in result.failures],
    }


def run_comparison(
    requested_retriever: str | None = None,
    model_a: str | None = None,
    model_b: str | None = None,
    label: str = "",
) -> dict:
    """Rank retrievers, then compare two RAG models against model A without RAG.

    All strategies are ranked by recall, then MRR, then median latency.
    Generation uses the configured strategy (hybrid by default), unless
    ``requested_retriever`` explicitly overrides it. Rewritten benchmark
    queries are shared across strategies so rewrite cost does not bias ranking.

    Args:
        requested_retriever: Optional explicit strategy (``bm25``, ``semantic``,
            or ``hybrid``) for the generation comparison. Defaults to the
            configured production strategy.
        model_a: First Ollama model; defaults to configured primary model.
        model_b: Second Ollama model; defaults to configured comparison model.
        label: Optional suffix included in the report output folder.

    Returns:
        Mapping with ``run_dir``, JSON-compatible ``payload``, in-memory suite
        ``results``, and rendered comparison ``report``.

    Raises:
        ValueError: If both model IDs are equal or the selected retriever is
            unknown.
        DatasetError: If the benchmark datasets are invalid.
        FileNotFoundError: If corpus chunks or required local indexes are
            missing.
        RuntimeError: If a selected embedding model or Ollama planner cannot be
            initialized.
        OSError: If run output cannot be written.
    """
    settings = AppConfig.from_environment()
    model_a = model_a or settings.ollama_model
    model_b = model_b or settings.ollama_comparison_model
    if model_a == model_b:
        raise ValueError("Model A and model B must be different Ollama model IDs.")

    chunks = adapters.load_chunks(config.CHUNKS_DIR)
    corpus = {chunk["chunk_id"]: chunk.get("text", "") for chunk in chunks}
    retrieval_cases = datasets.load_retrieval_cases()
    strategies = ["bm25", "semantic", "hybrid"]
    retrievers = adapters.build_retrievers(
        strategies,
        config.CHUNKS_DIR,
        chunks=chunks,
        query_expansion=False,
    )
    expansion_latencies = []
    expander = None
    if settings.query_expansion:
        expander = OllamaQueryExpander(
            settings.query_expansion_model, settings.ollama_url
        )
        expanded_cases = []
        for case in retrieval_cases:
            if not case.answerable:
                expanded_cases.append(case)
                continue
            expansion_started = time.perf_counter()
            expanded_query = clean_expanded_query(expander.expand(case.query))
            expansion_latencies.append(
                (time.perf_counter() - expansion_started) * 1000
            )
            expanded_cases.append(replace(case, query=expanded_query))
        retrieval_cases = expanded_cases

    retrieval_results = {}
    for strategy in strategies:
        retrieval_results[strategy] = evaluate_retrieval(
            retrieval_cases, retrievers[strategy], chunks
        )
        retrieval_results[strategy].extra[
            "query_expansion_latency_ms_median"
        ] = statistics.median(expansion_latencies) if expansion_latencies else None
        retrieval_results[strategy].extra[
            "query_expansion_latency_ms_p95"
        ] = _percentile(expansion_latencies, 0.95)

    def retrieval_rank(strategy: str) -> tuple[float, float, float]:
        """Build the descending-priority key for one retrieval strategy.

        Args:
            strategy: Retriever key in the evaluated results.

        Returns:
            Tuple of Recall@MAIN_K, MRR, and negative median latency, using
            sentinel values when a metric is undefined.
        """
        result = retrieval_results[strategy]
        recall = result.metrics.get(f"recall@{config.MAIN_K}")
        mrr = result.metrics.get("mrr")
        latency = result.extra.get("latency_ms_median")
        return (
            recall.value if recall and recall.value is not None else -1.0,
            mrr.value if mrr and mrr.value is not None else -1.0,
            -(latency if latency is not None else float("inf")),
        )

    ranked_retrievers = sorted(strategies, key=retrieval_rank, reverse=True)
    best_retriever = requested_retriever or settings.retrieval_strategy
    if best_retriever not in strategies:
        raise ValueError(f"Unknown comparison retriever: {best_retriever}")

    generation_cases = datasets.load_generation_cases()
    generation_expansion_latencies = []
    if expander is not None:
        from ..rag.context import make_query
        from ..schemas.models import PlanRequest, UserProfile

        for case in generation_cases:
            try:
                request = PlanRequest(
                    focus=case.request,
                    weeks=case.weeks,
                    workouts_per_week=case.workouts_per_week,
                    extra_constraints=case.extra_constraints,
                    profile=UserProfile.model_validate(case.profile),
                )
            except (TypeError, ValueError):
                continue
            expansion_started = time.perf_counter()
            expander.expand(make_query(request))
            generation_expansion_latencies.append(
                (time.perf_counter() - expansion_started) * 1000
            )

    generation_results = {}
    generation_arms = [
        (f"{model_a} + RAG", "ollama-rag", model_a),
        (f"{model_b} + RAG", "ollama-rag", model_b),
        (f"{model_a} without RAG", "ollama-no-rag", model_a),
    ]
    shared_retriever = retrievers[best_retriever]
    if expander is not None:
        shared_retriever = QueryExpansionRetriever(shared_retriever, expander)
    generation_planners = {}
    for arm, planner_name, model_name in generation_arms:
        planner = adapters.build_planner(
            planner_name,
            config.CHUNKS_DIR,
            model_name=model_name,
            retriever_name=best_retriever,
            retriever=shared_retriever,
        )
        generation_planners[arm] = planner
        generation_results[arm] = evaluate_generation(
            generation_cases,
            planner,
            corpus,
            llm_mode="real Ollama",
        )

    def generation_rank(arm: str) -> tuple[float, float, float, float]:
        """Build the priority key for one generation comparison arm.

        Args:
            arm: Generation-arm label in the evaluated results.

        Returns:
            Tuple ordered by pass rate, request compliance, schema validity,
            and negative median latency.
        """
        result = generation_results[arm]

        def metric(name: str) -> float:
            """Read one defined generation metric or use the ranking sentinel.

            Args:
                name: Metric key in the arm's result.

            Returns:
                Metric value, or ``-1.0`` when absent or not applicable.
            """
            ratio = result.metrics.get(name)
            return ratio.value if ratio and ratio.value is not None else -1.0

        latency = result.extra.get("latency_ms_median")
        return (
            result.pass_rate.value or 0.0,
            metric("request_constraint_compliance"),
            metric("output_schema_validity"),
            -(latency if latency is not None else float("inf")),
        )

    ranked_arms = sorted(
        (arm for arm in generation_results if arm.endswith(" + RAG")),
        key=generation_rank,
        reverse=True,
    )
    best_rag_arm = ranked_arms[0]
    best_rag_model = next(
        model_name
        for arm, _, model_name in generation_arms
        if arm == best_rag_arm
    )
    red_team_cases = datasets.load_red_team_cases()
    red_team_result = evaluate_generation(
        red_team_cases,
        generation_planners[best_rag_arm],
        corpus,
        llm_mode="real Ollama",
    )
    timestamp = datetime.now().isoformat(timespec="seconds")
    meta = {
        "timestamp": timestamp,
        "suite": "retrieval and generation comparison",
        "label": label,
        "retrieval_ranking": ranked_retrievers,
        "selected_retriever": best_retriever,
        "retrieval_selection_override": requested_retriever is not None,
        "retrieval_selection_source": (
            "requested"
            if requested_retriever is not None
            else "configured_default"
        ),
        "model_a": model_a,
        "model_b": model_b,
        "generation_ranking": ranked_arms,
        "generation_baseline": f"{model_a} without RAG",
        "best_rag_setup": {
            "generation_arm": best_rag_arm,
            "generation_model": best_rag_model,
            "retriever": best_retriever,
        },
        "chunks_dir": str(config.CHUNKS_DIR),
        "n_corpus_chunks": len(chunks),
        "embedding_model": settings.embedding_model,
        "query_expansion": settings.query_expansion,
        "query_expansion_model": (
            settings.query_expansion_model if settings.query_expansion else None
        ),
        "query_expansion_latency_ms_median": (
            statistics.median(expansion_latencies) if expansion_latencies else None
        ),
        "query_expansion_latency_ms_p95": _percentile(expansion_latencies, 0.95),
        "generation_query_expansion_latency_ms_median": (
            statistics.median(generation_expansion_latencies)
            if generation_expansion_latencies
            else None
        ),
        "generation_query_expansion_latency_ms_p95": _percentile(
            generation_expansion_latencies, 0.95
        ),
        "ollama_url": settings.ollama_url,
        "generation_case_ids": [case.id for case in generation_cases],
        "red_team_case_ids": [case.id for case in red_team_cases],
    }
    results = {
        **{f"retrieval/{name}": result for name, result in retrieval_results.items()},
        **{f"generation/{name}": result for name, result in generation_results.items()},
        "red-team/best-setup": red_team_result,
    }
    markdown = report.render_comparison(
        meta, retrieval_results, generation_results, red_team_result
    )
    run_dir = (
        config.OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}{'_' + label if label else ''}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "retrieval_ranking": ranked_retrievers,
        "generation_ranking": ranked_arms,
        "suites": {name: _serialize_result(result) for name, result in results.items()},
    }
    (run_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    return {
        "run_dir": run_dir,
        "payload": payload,
        "results": results,
        "report": markdown,
    }


def main(argv=None) -> int:
    """Run the selected benchmark mode and print its concise report.

    Args:
        argv: Optional command-line argument sequence; ``None`` uses
            ``sys.argv``.

    Returns:
        Process status: zero on normal completion, or one when configured
        release gates fail.

    Raises:
        SystemExit: If command-line arguments are invalid.
        ValueError: If red-team mode is requested without the real Ollama
            planner or a selected configuration is invalid.
        DatasetError: If the selected dataset fails validation.
        OSError: If run outputs cannot be written.
    """
    args = parse_args(argv)
    configure_logging("evaluation")
    logger.info(
        "Evaluation started suite=%s llm=%s retriever=%s model_a=%s model_b=%s",
        args.suite,
        args.llm,
        args.retriever or config.RETRIEVER,
        args.model_a or "configured",
        args.model_b or "configured",
    )
    if args.suite == "compare" or args.llm == "compare":
        run = run_comparison(
            args.retriever,
            args.model_a,
            args.model_b,
            args.label,
        )
        print(run["report"])
        print(f"\nFull report: {run['run_dir'] / 'report.md'}")
        logger.info("Model comparison completed output_dir=%s", run["run_dir"])
        return 0

    planner_name = "ollama-rag" if args.llm == "ollama" else config.PLANNER
    if args.suite == "red-team" and args.llm != "ollama":
        raise ValueError("Red-team mode requires --llm ollama.")
    run = run_evaluation(
        args.suite,
        args.retriever or config.RETRIEVER,
        planner_name,
        args.label,
        generation_model=args.model_a,
    )
    print(report.render_console(run["results"], run["gates"]))
    print(f"\nFull report: {run['run_dir'] / 'report.md'}")
    logger.info("Evaluation completed output_dir=%s", run["run_dir"])

    if config.FAIL_ON_GATES and any(g["status"] != "pass" for g in run["gates"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
