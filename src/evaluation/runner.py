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
        t0 = time.perf_counter()
        ranking = (
            retriever(case.query, max(max(ks), config.RETRIEVAL_DEPTH))
            if case.answerable
            else []
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        if not case.answerable:
            # No relevance labels exist for these cases; generation checks abstention.
            continue
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


def evaluate_generation(cases, planner, corpus: dict[str, str] | None) -> SuiteResult:
    res = SuiteResult("generation")
    outcomes, latencies = [], []
    released = schema_ok = 0
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
            available_sources = set(out.get("available_source_ids") or [])
            for session in plan.get("sessions") or []:
                for exercise in session.get("exercises") or []:
                    citations = exercise.get("source_ids") or []
                    if not citations:
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
        "citation_id_validity": Ratio(valid_citations, total_citations),
        "citation_precision": mean_ratio(grounding["citation_precision"]),
        "faithfulness": mean_ratio(grounding["faithfulness"]),
    }
    res.extra = {
        "fabricated_citations": fabricated,
        "grounding_evaluated": corpus is not None,
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
        generation_cases = datasets.load_generation_cases()
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
        safety_case_ids = {
            case.id
            for case in generation_cases
            if case.id.startswith("RT") or case.id in {"S006", "S007"}
        }
        unsafe_case_ids = {failure.case_id for failure in unsafe}
        safety_status = (
            "not measured"
            if not safety_case_ids.issubset(gen.case_ids)
            else ("fail" if unsafe else "pass")
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
        choices=["retrieval", "generation", "all"],
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
        choices=["mock", "ollama"],
        default="mock",
        help="mocked generation for deterministic cases, or the real local Ollama pipeline",
    )
    return p.parse_args(argv)


def run_evaluation(
    suite: str, retriever_name: str, planner_name: str, label: str = ""
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
    if suite != "generation" or planner_name == "ollama":
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

    if suite in ("generation", "all"):
        cases = datasets.load_generation_cases()
        planner_chunks_dir = config.CHUNKS_DIR
        if planner_name == "mock-naive":
            try:
                chunks = adapters.load_chunks(planner_chunks_dir)
            except FileNotFoundError:
                chunks = []
            planner = adapters.make_mock_naive(chunks)
        else:
            planner = adapters.build_planner(
                planner_name,
                planner_chunks_dir,
                fixed_evidence=(suite == "generation" and planner_name == "ollama"),
            )
        results["generation"] = evaluate_generation(cases, planner, corpus)

    gates = release_gates(results)
    runtime_settings = AppConfig.from_environment()
    effective_retriever = (
        config.RETRIEVER
        if suite != "retrieval" and planner_name == "ollama"
        else retriever_name
    )
    if suite == "generation" and planner_name != "ollama":
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
            else "real Ollama" if planner_name == "ollama" else "mocked"
        ),
        "embedding_model": (
            runtime_settings.embedding_model
            if effective_retriever in {"semantic", "hybrid"}
            else None
        ),
        "generation_model": (
            runtime_settings.ollama_model
            if suite != "retrieval" and planner_name == "ollama"
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


def main(argv=None) -> int:
    args = parse_args(argv)
    planner_name = "ollama" if args.llm == "ollama" else config.PLANNER
    run = run_evaluation(
        args.suite, args.retriever or config.RETRIEVER, planner_name, args.label
    )
    print(report.render_console(run["results"], run["gates"]))
    print(f"\nFull report: {run['run_dir'] / 'report.md'}")

    if config.FAIL_ON_GATES and any(g["status"] == "fail" for g in run["gates"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
