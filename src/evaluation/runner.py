"""One command to run the evaluation suites and report pass rates and failures.

    python -m src.evaluation.runner --suite all --retriever bm25 --planner mock-naive

Suites:
    retrieval   labelled queries -> Recall@k, Precision@k, Hit@1, MRR, nDCG@k, latency
    generation  program scenarios + red-team cases -> status, abstention, grounding,
                hard constraints, completeness, duration, program validity
    all         both

Adaptation (multi-session) red-team cases are listed as "not run" until an
adaptation engine adapter exists; they are never counted as passed.
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from . import adapters, datasets, report
from .metrics.common import Ratio, mean_ratio, pooled
from .metrics.grounding import abstention_metrics, abstention_outcome, check_claims
from .metrics.program import check_program
from .metrics.retrieval import judge_ranking

DEFAULT_KS = (1, 3, 5, 10)
RETRIEVAL_DEPTH = 50
DEFAULT_OUT = datasets.REPO / "outputs" / "eval_runs"


@dataclass
class Failure:
    case_id: str
    layer: str
    category: str
    detail: str
    rt: str | None = None


@dataclass
class SuiteResult:
    name: str
    n_cases: int = 0
    n_passed: int = 0
    metrics: dict[str, Ratio] = field(default_factory=dict)
    breakdown: dict[str, dict[str, Ratio]] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)
    not_run: list[dict] = field(default_factory=list)

    @property
    def pass_rate(self) -> Ratio:
        return Ratio(self.n_passed, self.n_cases)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


# ---------------------------------------------------------------- retrieval

def evaluate_retrieval(cases, retriever, ks=DEFAULT_KS, main_k: int = 5) -> SuiteResult:
    res = SuiteResult("retrieval")
    per_query: dict[str, list[float]] = {}
    by_lang: dict[str, dict[str, list[float]]] = {}
    latencies = []

    def add(bucket: dict, key: str, value):
        if value is not None:
            bucket.setdefault(key, []).append(value)

    for case in cases:
        t0 = time.perf_counter()
        ranking = retriever(case.query, max(max(ks), RETRIEVAL_DEPTH))
        latencies.append((time.perf_counter() - t0) * 1000)
        if not case.answerable:
            # Retrieval scores need labelled evidence; abstention is judged in the generation suite.
            continue
        res.n_cases += 1
        j = judge_ranking(case.spans, ranking)
        for k in ks:
            add(per_query, f"recall@{k}", j.recall_at_k(k))
            add(per_query, f"precision@{k}", j.precision_at_k(k))
            add(per_query, f"ndcg@{k}", j.ndcg_at_k(k))
        add(per_query, "hit@1", j.hit_at_k(1))
        add(per_query, "mrr", j.reciprocal_rank())
        lang = by_lang.setdefault(case.lang, {})
        add(lang, f"recall@{main_k}", j.recall_at_k(main_k))
        add(lang, "mrr", j.reciprocal_rank())

        recall = j.recall_at_k(main_k)
        if recall == 1.0:
            res.n_passed += 1
            continue
        rank = j.first_relevant_rank()
        if rank is None:
            category, detail = "not_retrieved", f"no labelled span in the top {len(ranking)}"
        elif rank > main_k:
            category, detail = "missed_top_ranked", f"first relevant chunk at rank {rank} (k={main_k})"
        else:
            category, detail = "partial_recall", f"recall@{main_k}={recall:.2f}, first relevant at rank {rank}"
        top = ", ".join(f"{c.get('video_id')}@{c.get('start')}s" for c in ranking[:3])
        res.failures.append(Failure(case.id, "retrieval", category, f"{detail}. Top 3: {top}. Query: {case.query}"))

    res.metrics = {name: mean_ratio(vals) for name, vals in per_query.items()}
    res.breakdown = {lang: {m: mean_ratio(v) for m, v in vals.items()} for lang, vals in by_lang.items()}
    res.extra = {
        "main_k": main_k,
        "n_unanswerable_skipped": sum(not c.answerable for c in cases),
        "n_draft_labels": sum(c.review_status != "reviewed" for c in cases),
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
        if not isinstance(s, dict) or not isinstance(s.get("exercises"), list) or not s["exercises"]:
            errors.append(f"session {i} has no exercise list")
        elif not all(isinstance(ex, dict) and ex.get("name") for ex in s["exercises"]):
            errors.append(f"session {i} has an exercise without a name")
    return errors


def evaluate_generation(cases, planner, profiles: dict, corpus: dict[str, str] | None) -> SuiteResult:
    res = SuiteResult("generation")
    outcomes, latencies = [], []
    released = valid_released = schema_ok = 0
    constraint_ratios, completeness, prefs, overruns = [], [], [], []
    zero_violation_plans = 0
    grounding = {"citation_precision": [], "unsupported_claim_rate": [], "grounded_claim_rate": []}
    fabricated = 0
    redteam_total = redteam_passed = 0

    for case in cases:
        if case.layer == "adaptation":
            res.not_run.append({"case_id": case.id, "rt": case.rt, "reason": "no adaptation engine adapter yet"})
            continue
        res.n_cases += 1
        profile = {**profiles.get(case.profile_id, {}), **case.profile_overrides}
        t0 = time.perf_counter()
        try:
            out = planner({"request": case.request, "lang": case.lang, "profile": profile})
            error = None
        except Exception as e:  # a crash is a failure of the system under test, not of the runner
            out, error = {"status": "error"}, f"{type(e).__name__}: {e}"
        latencies.append((time.perf_counter() - t0) * 1000)
        status = out.get("status")
        fails: list[Failure] = []

        def fail(category: str, detail: str):
            fails.append(Failure(case.id, "generation", category, detail, case.rt))

        if error:
            fail("pipeline_error", error)
        if status not in case.expected_status:
            category = "unsafe_release" if status == "ok" and "ok" not in case.expected_status else "wrong_status"
            fail(category, f"expected {'/'.join(case.expected_status)}, got {status}")
        outcomes.append(abstention_outcome(case.expected_status[0], status))

        text = adapters.answer_text(out)
        for p in adapters.pattern_hits(case.forbidden_patterns, text):
            fail("forbidden_output", f"matched /{p}/")
        missing = [p for p in case.required_patterns if p not in adapters.pattern_hits([p], text)]
        for p in missing:
            fail("missing_required", f"no match for /{p}/")

        plan = out.get("plan")
        if status == "ok" and plan is not None:
            released += 1
            errors = plan_schema_errors(plan)
            if errors:
                fail("schema_invalid", "; ".join(errors))
            else:
                schema_ok += 1
                pr = check_program(plan, profile, case.forbidden_exercises)
                constraint_ratios.append(pr.hard_constraint_adherence)
                completeness.append(pr.parameter_completeness)
                prefs.append(pr.preference_satisfaction)
                overruns.extend(pr.overruns)
                zero_violation_plans += pr.zero_violations
                for v in pr.violations:
                    fail("constraint_violation", f"{v.name}: {v.detail}".rstrip(": "))
                if pr.parameter_completeness.value not in (None, 1.0):
                    fail("incomplete_parameters", f"completeness {pr.parameter_completeness.fmt()}")
                if pr.zero_violations and pr.parameter_completeness.value in (None, 1.0):
                    valid_released += 1

        if corpus is not None and out.get("claims"):
            g = check_claims(out["claims"], corpus, set(out.get("retrieved_ids") or []) or None)
            grounding["citation_precision"].append(g.citation_precision)
            grounding["unsupported_claim_rate"].append(g.unsupported_claim_rate)
            grounding["grounded_claim_rate"].append(g.grounded_claim_rate)
            fabricated += g.fabricated_citations
            if g.fabricated_citations:
                fail("fabricated_citation", f"{g.fabricated_citations} citation(s) to unknown chunks")
            if g.unsupported_claim_rate.num:
                fail("unsupported_claim", f"{int(g.unsupported_claim_rate.num)} evidence claim(s) without a matching quote")

        res.failures.extend(fails)
        res.n_passed += not fails
        if case.source == "red_team":
            redteam_total += 1
            redteam_passed += not fails

    res.metrics = {
        **abstention_metrics(outcomes),
        "output_schema_validity": Ratio(schema_ok, released),
        "hard_constraint_adherence": pooled(constraint_ratios),
        "plans_with_zero_violations": Ratio(zero_violation_plans, schema_ok),
        "preference_satisfaction": pooled(prefs),
        "parameter_completeness": pooled(completeness),
        "duration_overrun_mean": mean_ratio(overruns),
        "program_validity_rate": Ratio(valid_released, released),
        "citation_precision": pooled(grounding["citation_precision"]),
        "unsupported_claim_rate": pooled(grounding["unsupported_claim_rate"]),
        "grounded_claim_rate": pooled(grounding["grounded_claim_rate"]),
        "red_team_pass_rate": Ratio(redteam_passed, redteam_total),
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
        validity = gen.metrics["program_validity_rate"]
        gates.append({"gate": "100% released-plan validator compliance",
                      "status": "n/a" if validity.value is None else ("pass" if validity.value == 1 else "fail"),
                      "value": validity.fmt()})
        unsafe = [f for f in gen.failures if f.category == "unsafe_release"]
        gates.append({"gate": "zero unsafe releases in the red-team suite",
                      "status": "pass" if not unsafe else "fail", "value": f"{len(unsafe)} unsafe"})
        gates.append({"gate": "adaptation traces and duplicate/correction tests",
                      "status": "not evaluated" if gen.not_run else "pass",
                      "value": f"{len(gen.not_run)} adaptation case(s) not run"})
    return gates


# --------------------------------------------------------------------- CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run RepWise evaluation suites.")
    p.add_argument("--suite", choices=["retrieval", "generation", "all"], default="all")
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--retriever", default="bm25", help="'bm25' or 'module.path:callable'")
    p.add_argument("--planner", default="mock-naive",
                   help="'mock-naive', 'mock-abstain' or 'module.path:callable'")
    p.add_argument("--chunks-dir", type=Path, default=adapters.DEFAULT_CHUNKS_DIR)
    p.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--main-k", type=int, default=5, help="k used for pass/fail of a retrieval case")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--label", default="", help="free text name for this run")
    p.add_argument("--gates", action="store_true", help="exit with code 1 if a release gate fails")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    keep = (lambda c: True) if args.split == "all" else (lambda c: c.split == args.split)
    results: dict[str, SuiteResult] = {}

    corpus = None
    try:
        corpus = {c["chunk_id"]: c.get("text", "") for c in adapters.load_chunks(args.chunks_dir)}
    except FileNotFoundError as e:
        print(f"[warn] {e} Grounding metrics will be N/A.", file=sys.stderr)

    if args.suite in ("retrieval", "all"):
        if corpus is None and args.retriever == "bm25":
            print("[warn] retrieval suite skipped: the bm25 baseline needs the chunk corpus.", file=sys.stderr)
        else:
            cases = [c for c in datasets.load_retrieval_cases() if keep(c)]
            retriever = adapters.build_retriever(args.retriever, args.chunks_dir)
            results["retrieval"] = evaluate_retrieval(cases, retriever, args.k, args.main_k)

    if args.suite in ("generation", "all"):
        profiles = datasets.load_profiles()
        cases = [c for c in datasets.load_generation_cases(profiles=profiles) if keep(c)]
        planner = adapters.build_planner(args.planner, args.chunks_dir)
        results["generation"] = evaluate_generation(cases, planner, profiles, corpus)

    gates = release_gates(results)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "suite": args.suite, "split": args.split, "retriever": args.retriever,
        "planner": args.planner, "chunks_dir": str(args.chunks_dir), "label": args.label,
        "n_corpus_chunks": len(corpus) if corpus is not None else 0,
    }
    run_dir = args.out / f"{datetime.now():%Y%m%d-%H%M%S}{'_' + args.label if args.label else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta, "gates": gates,
        "suites": {name: {
            "n_cases": r.n_cases, "n_passed": r.n_passed, "pass_rate": r.pass_rate.to_dict(),
            "metrics": {k: v.to_dict() for k, v in r.metrics.items()},
            "breakdown": {g: {k: v.to_dict() for k, v in m.items()} for g, m in r.breakdown.items()},
            "extra": r.extra, "failures": [asdict(f) for f in r.failures], "not_run": r.not_run,
        } for name, r in results.items()},
    }
    (run_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = report.render(meta, results, gates)
    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(report.render_console(results, gates))
    print(f"\nFull report: {run_dir / 'report.md'}")

    if args.gates and any(g["status"] == "fail" for g in gates):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
