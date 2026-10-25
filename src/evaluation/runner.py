"""One command to run the evaluation suites and report pass rates and failures.

    python -m src.evaluation.runner --suite all

Settings (systems under test, split, corpus folder, k values, output folder)
are in config/evaluation_config.py.

Suites:
    retrieval   labelled queries -> Recall@k, Precision@k, Hit@1, MRR, latency
    generation  program scenarios + red-team cases -> status, abstention,
                citation precision, faithfulness, output format
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

from config import evaluation_config as config
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
    rt: str | None = None


@dataclass
class SuiteResult:
    name: str
    n_cases: int = 0
    n_passed: int = 0
    metrics: dict[str, Ratio] = field(default_factory=dict)
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
        t0 = time.perf_counter()
        ranking = retriever(case.query, max(max(ks), config.RETRIEVAL_DEPTH))
        latencies.append((time.perf_counter() - t0) * 1000)
        if not case.answerable:
            # Retrieval scores need labelled evidence; abstention is judged in the generation suite.
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
            category, detail = "not_retrieved", f"no labelled span in the top {len(ranking)}"
        elif rank > main_k:
            category, detail = "missed_top_ranked", f"first relevant chunk at rank {rank} (k={main_k})"
        else:
            category, detail = "partial_recall", f"recall@{main_k}={recall:.2f}, first relevant at rank {rank}"
        top = ", ".join(f"{c.get('video_id')}@{c.get('start')}s" for c in ranking[:3])
        res.failures.append(Failure(case.id, "retrieval", category, f"{detail}. Top 3: {top}. Query: {case.query}"))

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
        if not isinstance(s, dict) or not isinstance(s.get("exercises"), list) or not s["exercises"]:
            errors.append(f"session {i} has no exercise list")
        elif not all(isinstance(ex, dict) and ex.get("name") for ex in s["exercises"]):
            errors.append(f"session {i} has an exercise without a name")
    return errors


def evaluate_generation(cases, planner, profiles: dict, corpus: dict[str, str] | None) -> SuiteResult:
    res = SuiteResult("generation")
    outcomes, latencies = [], []
    released = schema_ok = 0
    grounding = {"citation_precision": [], "faithfulness": []}
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
            out = planner({"request": case.request, "profile": profile})
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

        if corpus is not None and out.get("claims"):
            g = check_claims(out["claims"], corpus, set(out.get("retrieved_ids") or []) or None)
            for name in ("citation_precision", "faithfulness"):
                value = getattr(g, name).value
                if value is not None:
                    grounding[name].append(value)
            fabricated += g.fabricated_citations
            if g.fabricated_citations:
                fail("fabricated_citation", f"{g.fabricated_citations} citation(s) to unknown chunks")
            unsupported = g.n_evidence_claims - g.grounded_claims
            if unsupported:
                fail("unsupported_claim", f"{unsupported} claim(s) not supported by the retrieved passages")


        res.failures.extend(fails)
        res.n_passed += not fails
        if case.source == "red_team":
            redteam_total += 1
            redteam_passed += not fails

    res.metrics = {
        **abstention_metrics(outcomes),
        "output_schema_validity": Ratio(schema_ok, released),
        "citation_precision": mean_ratio(grounding["citation_precision"]),
        "faithfulness": mean_ratio(grounding["faithfulness"]),
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
        validity = gen.metrics["output_schema_validity"]
        gates.append({"gate": "100% of released plans have a valid format",
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
    p = argparse.ArgumentParser(
        description="Run the RepWise test cases and write a report. Other settings are in config/evaluation_config.py.")
    p.add_argument("--suite", choices=["retrieval", "generation", "all"], default="all",
                   help="what to test: retrieval of passages, generated plans, or both (default: all)")
    p.add_argument("--label", default="",
                   help="name added to the run folder, e.g. the benchmark entry being tested")
    return p.parse_args(argv)


def run_evaluation(suite: str, retriever_name: str, planner_name: str, label: str = "") -> dict:
    """Evaluate one retriever / planner pair on the test cases.

    Writes results.json and report.md in a new run folder and returns
    {"run_dir", "payload", "results", "gates"}. Used by main() and by the benchmark.
    """
    keep = (lambda c: True) if config.SPLIT == "all" else (lambda c: c.split == config.SPLIT)
    results: dict[str, SuiteResult] = {}
    warnings: list[str] = []

    def warn(message: str):
        warnings.append(message)
        print(f"[warn] {message}", file=sys.stderr)

    corpus = None
    try:
        corpus = {c["chunk_id"]: c.get("text", "") for c in adapters.load_chunks(config.CHUNKS_DIR)}
    except FileNotFoundError as e:
        warn(f"{e} Grounding metrics will be N/A.")

    if suite in ("retrieval", "all"):
        if corpus is None and retriever_name == "bm25":
            warn("Retrieval suite skipped: the bm25 baseline needs the chunk corpus.")
        else:
            cases = [c for c in datasets.load_retrieval_cases() if keep(c)]
            retriever = adapters.build_retriever(retriever_name, config.CHUNKS_DIR)
            results["retrieval"] = evaluate_retrieval(cases, retriever)

    if suite in ("generation", "all"):
        profiles = datasets.load_profiles()
        cases = [c for c in datasets.load_generation_cases(profiles=profiles) if keep(c)]
        planner = adapters.build_planner(planner_name, config.CHUNKS_DIR)
        results["generation"] = evaluate_generation(cases, planner, profiles, corpus)

    gates = release_gates(results)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "suite": suite, "split": config.SPLIT, "retriever": retriever_name,
        "planner": planner_name, "chunks_dir": str(config.CHUNKS_DIR), "label": label,
        "n_corpus_chunks": len(corpus) if corpus is not None else 0,
        "warnings": warnings,
    }
    run_dir = config.OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}{'_' + label if label else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta, "gates": gates,
        "suites": {name: {
            "n_cases": r.n_cases, "n_passed": r.n_passed, "pass_rate": r.pass_rate.to_dict(),
            "metrics": {k: v.to_dict() for k, v in r.metrics.items()},
            "extra": r.extra, "failures": [asdict(f) for f in r.failures], "not_run": r.not_run,
        } for name, r in results.items()},
    }
    (run_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = report.render(meta, results, gates)
    (run_dir / "report.md").write_text(markdown, encoding="utf-8")
    return {"run_dir": run_dir, "payload": payload, "results": results, "gates": gates}


def main(argv=None) -> int:
    args = parse_args(argv)
    run = run_evaluation(args.suite, config.RETRIEVER, config.PLANNER, args.label)
    print(report.render_console(run["results"], run["gates"]))
    print(f"\nFull report: {run['run_dir'] / 'report.md'}")

    if config.FAIL_ON_GATES and any(g["status"] == "fail" for g in run["gates"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
