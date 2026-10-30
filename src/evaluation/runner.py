"""Run the unified RepWise benchmark: retrieval, generation, or both."""

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from config import evaluation_config as config
from config.config import AppConfig
from . import adapters, datasets, report
from .metrics.common import Ratio, mean_ratio
from .metrics.grounding import abstention_metrics, abstention_outcome, check_claims
from .metrics.retrieval import judge_ranking


@dataclass
class Failure:
    case_id: str
    layer: str
    category: str
    detail: str


@dataclass
class SuiteResult:
    name: str
    n_cases: int = 0
    n_passed: int = 0
    metrics: dict[str, Ratio] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)
    case_ids: set[str] = field(default_factory=set)

    @property
    def pass_rate(self) -> Ratio:
        return Ratio(self.n_passed, self.n_cases)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


# ---------------------------------------------------------------- retrieval


def evaluate_retrieval(cases, retriever, ks=None, main_k=None) -> SuiteResult:
    ks = ks or config.K_VALUES
    main_k = main_k or config.MAIN_K
    res = SuiteResult("retrieval")
    per_query: dict[str, list[float]] = {}
    latencies = []

    def add(bucket: dict, key: str, value):

        if value is not None:
            bucket.setdefault(key, []).append(value)

    for case in cases:
        res.case_ids.add(case.id)
        if not case.answerable:
            # No relevance labels exist for these cases; generation checks abstention.
            continue
        t0 = time.perf_counter()
        ranking = retriever(case.query, max(max(ks), config.RETRIEVAL_DEPTH))
        latencies.append((time.perf_counter() - t0) * 1000)
        res.n_cases += 1
        j = judge_ranking(case.spans, ranking)
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
                f"no labelled span in the top {len(ranking)}",
            )
        elif rank > main_k:
            category, detail = (
                "missed_top_ranked",
                f"first relevant chunk at rank {rank} (k={main_k})",
            )
        else:
            category, detail = (
                "partial_recall",
                f"recall@{main_k}={recall:.2f}, first relevant at rank {rank}",
            )
        top = ", ".join(f"{c.get('video_id')}@{c.get('start')}s" for c in ranking[:3])
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
        "n_unanswerable_skipped": sum(not c.answerable for c in cases),
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "latency_ms_p95": _percentile(latencies, 0.95),
    }
    return res


# --------------------------------------------------------------- generation


def plan_schema_errors(plan) -> list[str]:
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
            out = planner({"request": case.request, "profile": case.profile})
            error = None
        except (
            Exception
        ) as e:  # a crash is a failure of the system under test, not of the runner
            out, error = {"status": "error"}, f"{type(e).__name__}: {e}"
        latencies.append((time.perf_counter() - t0) * 1000)
        status = out.get("status")
        fails: list[Failure] = []

        def fail(category: str, detail: str):
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
                        extra_constraints=case.extra_constraints,
                        profile=UserProfile.model_validate(case.profile),
                    )
                    validate_plan(
                        TrainingPlan.model_validate(plan),
                        request,
                        set(out.get("available_source_ids") or []),
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
    """MVP engineering gates from the project note (section 6)."""
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
        unsafe = [f for f in gen.failures if f.category == "unsafe_release"]
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

    Writes results.json and report.md in a new run folder and returns
    {"run_dir", "payload", "results", "gates"}.
    """
    results: dict[str, SuiteResult] = {}
    warnings: list[str] = []

    def warn(message: str):
        warnings.append(message)
        print(f"[warn] {message}", file=sys.stderr)

    corpus = None
    if suite != "generation" or planner_name in {"ollama-rag", "ollama-no-rag"}:
        try:
            corpus = {
                c["chunk_id"]: c.get("text", "")
                for c in adapters.load_chunks(config.CHUNKS_DIR)
            }
        except FileNotFoundError as e:
            warn(f"{e} Grounding metrics will be N/A.")

    if suite in ("retrieval", "all"):
        if corpus is None:
            warn(f"Retrieval suite skipped: {retriever_name} needs the chunk corpus.")
        else:
            cases = datasets.load_retrieval_cases()
            retriever = adapters.build_retriever(retriever_name, config.CHUNKS_DIR)
            results["retrieval"] = evaluate_retrieval(cases, retriever)

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
            planner = adapters.build_planner(
                planner_name,
                planner_chunks_dir,
                model_name=generation_model or runtime_settings.ollama_model,
                retriever_name=retriever_name,
            )
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
    effective_retriever = (
        config.RETRIEVER
        if suite != "retrieval" and planner_name == "ollama-rag"
        else retriever_name
    )
    if suite == "generation" and planner_name not in {
        "ollama-rag",
        "ollama-no-rag",
    }:
        effective_retriever = "fixed mock evidence"
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
            else "real Ollama"
            if planner_name in {"ollama-rag", "ollama-no-rag"}
            else "mocked"
        ),
        "embedding_model": (
            runtime_settings.embedding_model
            if effective_retriever in {"semantic", "hybrid"}
            else None
        ),
        "generation_model": (
            generation_model or runtime_settings.ollama_model
            if suite != "retrieval"
            and planner_name in {"ollama-rag", "ollama-no-rag"}
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
    """Rank retrievers, then compare two RAG models against model A without RAG."""
    settings = AppConfig.from_environment()
    model_a = model_a or settings.ollama_model
    model_b = model_b or settings.ollama_comparison_model
    if model_a == model_b:
        raise ValueError("Model A and model B must be different Ollama model IDs.")

    chunks = adapters.load_chunks(config.CHUNKS_DIR)
    corpus = {chunk["chunk_id"]: chunk.get("text", "") for chunk in chunks}
    retrieval_cases = datasets.load_retrieval_cases()
    retrieval_results = {}
    strategies = ["bm25", "semantic", "hybrid"]
    for strategy in strategies:
        retriever = adapters.build_retriever(strategy, config.CHUNKS_DIR)
        retrieval_results[strategy] = evaluate_retrieval(
            retrieval_cases, retriever
        )

    def retrieval_rank(strategy: str) -> tuple[float, float, float]:
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
    best_retriever = requested_retriever or ranked_retrievers[0]
    if best_retriever not in strategies:
        raise ValueError(f"Unknown comparison retriever: {best_retriever}")

    generation_cases = datasets.load_generation_cases()
    generation_results = {}
    generation_arms = [
        (f"{model_a} + RAG", "ollama-rag", model_a),
        (f"{model_b} + RAG", "ollama-rag", model_b),
        (f"{model_a} without RAG", "ollama-no-rag", model_a),
    ]
    shared_retriever = adapters.build_retriever(
        best_retriever, config.CHUNKS_DIR
    )
    for arm, planner_name, model_name in generation_arms:
        planner = adapters.build_planner(
            planner_name,
            config.CHUNKS_DIR,
            model_name=model_name,
            retriever_name=best_retriever,
            retriever=shared_retriever,
        )
        generation_results[arm] = evaluate_generation(
            generation_cases,
            planner,
            corpus,
            llm_mode="real Ollama",
        )

    def generation_rank(arm: str) -> tuple[float, float, float, float]:
        result = generation_results[arm]

        def metric(name: str) -> float:
            ratio = result.metrics.get(name)
            return ratio.value if ratio and ratio.value is not None else -1.0

        latency = result.extra.get("latency_ms_median")
        return (
            result.pass_rate.value or 0.0,
            metric("request_constraint_compliance"),
            metric("output_schema_validity"),
            -(latency if latency is not None else float("inf")),
        )

    ranked_arms = sorted(generation_results, key=generation_rank, reverse=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    meta = {
        "timestamp": timestamp,
        "suite": "retrieval and generation comparison",
        "label": label,
        "retrieval_ranking": ranked_retrievers,
        "selected_retriever": best_retriever,
        "retrieval_selection_override": requested_retriever is not None,
        "model_a": model_a,
        "model_b": model_b,
        "generation_ranking": ranked_arms,
        "chunks_dir": str(config.CHUNKS_DIR),
        "n_corpus_chunks": len(chunks),
        "embedding_model": settings.embedding_model,
        "ollama_url": settings.ollama_url,
        "generation_case_ids": [case.id for case in generation_cases],
    }
    results = {
        **{f"retrieval/{name}": result for name, result in retrieval_results.items()},
        **{f"generation/{name}": result for name, result in generation_results.items()},
    }
    markdown = report.render_comparison(meta, retrieval_results, generation_results)
    run_dir = (
        config.OUT_DIR
        / f"{datetime.now():%Y%m%d-%H%M%S}{'_' + label if label else ''}"
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
    return {"run_dir": run_dir, "payload": payload, "results": results, "report": markdown}


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.suite == "compare" or args.llm == "compare":
        run = run_comparison(
            args.retriever,
            args.model_a,
            args.model_b,
            args.label,
        )
        print(run["report"])
        print(f"\nFull report: {run['run_dir'] / 'report.md'}")
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

    if config.FAIL_ON_GATES and any(g["status"] != "pass" for g in run["gates"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
