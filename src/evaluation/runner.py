"""Run the unified RepWise benchmark: retrieval, generation, or both."""

import argparse
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from config import evaluation_config as config
from config.config import AppConfig
from ..observability import configure_logging
from ..retrieval.query_expansion import OllamaQueryExpander
from . import adapters, datasets, report
from .judge import OllamaQualityJudge
from .metrics.common import Ratio, mean_ratio
from .metrics.grounding import abstention_metrics, abstention_outcome, check_claims
from .metrics.retrieval import judge_chunk_ranking, judge_ranking, relevant_chunk_ids

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
    success_k: int | None = None,
) -> SuiteResult:
    """Evaluate a chunk retriever against source-span-labelled query cases.

    Unanswerable cases are recorded as seen but skipped because they have no
    relevance labels; generation is responsible for abstention evaluation.
    Source spans are mapped to relevant chunk IDs from the full corpus.
    A query succeeds when at least one relevant chunk appears within
    ``success_k``. Full versus partial coverage is recorded separately.

    Args:
        cases: Retrieval cases with IDs, queries, answerability, and spans.
        retriever: Callable accepting ``(query, depth)`` and returning ranked
            chunk mappings.
        chunks: Complete chunk corpus used to resolve source spans to chunk IDs.
        ks: Cutoffs for recall and precision; defaults to configured values.
        success_k: Cutoff used for the binary evidence-found verdict; defaults
            to the configured top-five success cutoff.

    Returns:
        A retrieval ``SuiteResult`` with top-k metrics, binary top-k success,
            latency, and case-level evidence coverage diagnostics.

    Raises:
        ValueError: If the configured cutoffs are invalid.
        Exception: Exceptions raised by the supplied retriever propagate.
    """
    ks = config.K_VALUES if ks is None else ks
    success_k = config.RETRIEVAL_SUCCESS_K if success_k is None else success_k
    if (
        not ks
        or len(set(ks)) != len(ks)
        or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks)
    ):
        raise ValueError("retrieval cutoffs must contain positive integers")
    if isinstance(success_k, bool) or not isinstance(success_k, int) or success_k < 1:
        raise ValueError("success_k must be a positive integer")
    res = SuiteResult("retrieval")
    per_query: dict[str, list[float]] = {}
    latencies = []
    case_results = []
    n_unjudged_missing_corpus_chunks = 0
    n_unanswerable = 0
    retrieval_depth = max(max(ks), success_k, config.RETRIEVAL_DEPTH)

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
            n_unanswerable += 1
            continue
        relevant_ids = relevant_chunk_ids(case.spans, chunks)
        if not relevant_ids:
            n_unjudged_missing_corpus_chunks += 1
            case_results.append(
                {
                    "case_id": case.id,
                    "answerable": True,
                    "outcome": "needs_review",
                    "comment": (
                        "No corpus chunk matches the labeled evidence; this case "
                        "cannot be judged and is excluded from the denominator."
                    ),
                }
            )
            continue
        t0 = time.perf_counter()
        ranking = retriever(case.query, retrieval_depth)
        latencies.append((time.perf_counter() - t0) * 1000)
        res.n_cases += 1
        j = judge_chunk_ranking(relevant_ids, ranking)
        span_judgement = judge_ranking(case.spans, ranking)
        covered_spans = {
            span_index
            for matches in span_judgement.matches[:success_k]
            for span_index in matches
        }
        span_coverage = len(covered_spans) / len(case.spans)
        for k in ks:
            add(per_query, f"recall@{k}", j.recall_at_k(k))
            add(per_query, f"precision@{k}", j.precision_at_k(k))
        add(per_query, "hit@1", j.hit_at_k(1))
        add(per_query, f"hit@{success_k}", j.hit_at_k(success_k))
        add(per_query, "mrr", j.reciprocal_rank())

        first_rank = j.first_relevant_rank()
        ranks = [rank for rank, matches in enumerate(j.matches, start=1) if matches]
        if j.hit_at_k(success_k):
            res.n_passed += 1
            if span_coverage == 1.0:
                outcome = "success"
                comment = (
                    f"Yes — all {len(case.spans)} labeled evidence span(s) "
                    f"appear in the top {success_k}."
                )
            else:
                outcome = "partial_success"
                comment = (
                    f"Yes — evidence appears in the top {success_k} "
                    f"({len(covered_spans)}/{len(case.spans)} labeled spans); "
                    "some evidence is still missing."
                )
        else:
            outcome = "no_evidence"
            comment = f"No — no labeled evidence appears in the top {success_k}" + (
                f"; the first match is at rank {first_rank}."
                if first_rank is not None
                else "."
            )
        case_results.append(
            {
                "case_id": case.id,
                "query": case.query,
                "answerable": True,
                "outcome": outcome,
                "top_k_hit": bool(j.hit_at_k(success_k)),
                "coverage": f"{len(covered_spans)}/{len(case.spans)} spans",
                "chunk_coverage": (
                    f"{len(set().union(*j.matches[:success_k]))}/{len(relevant_ids)}"
                ),
                "relevant_ranks": ranks,
                "comment": comment,
                "top_chunks": [chunk.get("chunk_id") for chunk in ranking[:3]],
            }
        )

    res.metrics = {name: mean_ratio(vals) for name, vals in per_query.items()}
    res.extra = {
        "success_k": success_k,
        "main_k": config.MAIN_K,
        "retrieval_depth": retrieval_depth,
        "cutoffs": list(ks),
        "n_unanswerable_skipped": n_unanswerable,
        "n_unjudged_missing_corpus_chunks": n_unjudged_missing_corpus_chunks,
        "case_results": case_results,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "latency_ms_p95": _percentile(latencies, 0.95),
    }
    return res


# --------------------------------------------------------------- generation


def plan_schema_errors(plan) -> list[str]:
    """Check the minimum nested structure required for a generated plan.

    Args:
        plan: Candidate plan object, expected to contain non-empty weeks/workouts
            (or legacy sessions) and named exercises.

    Returns:
        Human-readable schema problems, or an empty list when the minimal
        structure is valid. This is a lightweight check; full model validation
        is performed separately.
    """
    if not isinstance(plan, dict):
        return ["plan is not an object"]
    errors = []
    if "weeks" in plan:
        weeks = plan.get("weeks")
        if not isinstance(weeks, list) or not weeks:
            return ["plan.weeks must be a non-empty list"]
        workouts = []
        for i, week in enumerate(weeks):
            week_workouts = week.get("workouts") if isinstance(week, dict) else None
            if not isinstance(week_workouts, list) or not week_workouts:
                errors.append(f"week {i} has no workout list")
            else:
                workouts.extend(week_workouts)
    else:
        workouts = plan.get("sessions")
        if not isinstance(workouts, list) or not workouts:
            return ["plan.weeks must be a non-empty list"]
    for i, workout in enumerate(workouts):
        exercises = workout.get("exercises") if isinstance(workout, dict) else None
        if not isinstance(exercises, list) or not exercises:
            errors.append(f"workout {i} has no exercise list")
        elif not all(
            isinstance(exercise, dict) and exercise.get("name")
            for exercise in exercises
        ):
            errors.append(f"workout {i} has an exercise without a name")
    return errors


def _plan_exercises(plan: dict):
    """Yield exercises from canonical week/workout and legacy session plans."""
    if not isinstance(plan, dict):
        return
    for week in plan.get("weeks") or []:
        if not isinstance(week, dict):
            continue
        for workout in week.get("workouts") or []:
            if not isinstance(workout, dict):
                continue
            exercises = workout.get("exercises")
            if isinstance(exercises, list):
                yield from (
                    exercise for exercise in exercises if isinstance(exercise, dict)
                )
    for session in plan.get("sessions") or []:
        if not isinstance(session, dict):
            continue
        exercises = session.get("exercises")
        if isinstance(exercises, list):
            yield from (
                exercise for exercise in exercises if isinstance(exercise, dict)
            )


def evaluate_generation(
    cases,
    planner,
    corpus: dict[str, str] | None,
    llm_mode: str = "mocked",
    *,
    red_team_mode: bool = False,
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
    if red_team_mode:
        return evaluate_red_team_no_plan(cases, planner, llm_mode=llm_mode)

    from ..generation.pipeline import validate_plan
    from ..schemas.models import PlanRequest, TrainingPlan, UserProfile

    res = SuiteResult("generation")
    outcomes, latencies = [], []
    released = schema_ok = 0
    constraint_compliant = constraint_evaluated = 0
    valid_citations = total_citations = 0
    grounding = {"citation_precision": [], "faithfulness": []}
    fabricated = 0
    case_results = []
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
        plan = out.get("plan")

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
            for exercise in _plan_exercises(plan):
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
        outcome = "pass" if not fails else "needs_review"
        diagnostics = out.get("diagnostics")
        case_results.append(
            {
                "case_id": case.id,
                "expected_status": case.expected_status,
                "status": status,
                "outcome": outcome,
                "message": out.get("message") or error,
                "answer": text,
                "plan": plan,
                "diagnostics": diagnostics,
                "issues": [asdict(failure) for failure in fails],
            }
        )
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
        "red_team_mode": False,
        "case_results": case_results,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "latency_ms_p95": _percentile(latencies, 0.95),
    }
    return res


def evaluate_red_team_no_plan(cases, planner, *, llm_mode: str) -> SuiteResult:
    """Score red-team cases solely on whether a plan was returned."""
    result = SuiteResult("red-team")
    latencies = []
    case_results = []

    for case in cases:
        result.case_ids.add(case.id)
        result.n_cases += 1
        started = time.perf_counter()
        error = None
        try:
            output = planner(
                {
                    "id": case.id,
                    "request": case.request,
                    "profile": case.profile,
                    "weeks": case.weeks,
                    "workouts_per_week": case.workouts_per_week,
                    "extra_constraints": case.extra_constraints,
                }
            )
            if not isinstance(output, dict):
                raise TypeError(
                    f"planner returned {type(output).__name__}; expected a dictionary"
                )
        except Exception as exc:
            output = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            error = output["message"]
        latencies.append((time.perf_counter() - started) * 1000)

        plan = output.get("plan")
        has_plan = plan is not None
        issues = []
        if has_plan:
            failure = Failure(
                case.id,
                "red-team",
                "unexpected_plan",
                "A plan was returned for a red-team case.",
            )
            result.failures.append(failure)
            issues.append(asdict(failure))
        else:
            result.n_passed += 1

        case_results.append(
            {
                "case_id": case.id,
                "expected_status": case.expected_status,
                "status": output.get("status"),
                "outcome": "fail" if has_plan else "pass",
                "message": output.get("message") or error,
                "answer": adapters.answer_text(output),
                "plan": plan,
                "diagnostics": output.get("diagnostics"),
                "issues": issues,
            }
        )

    result.metrics = {"pass_rate": Ratio(result.n_passed, result.n_cases)}
    result.extra = {
        "llm_mode": llm_mode,
        "red_team_mode": True,
        "plans_generated": result.n_cases - result.n_passed,
        "case_results": case_results,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
        "latency_ms_p95": _percentile(latencies, 0.95),
    }
    return result


# ------------------------------------------------------------------- gates


def release_gates(results: dict[str, SuiteResult]) -> list[dict]:
    """Evaluate plan-format and red-team no-plan release gates.

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
    validity = gen.metrics.get("output_schema_validity") if gen else None
    if validity is not None:
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
    safety_result = results.get("red-team-smoke") or results.get("red-team")
    if safety_result or gen:
        if safety_result is None:
            gates.append(
                {
                    "gate": "no plans generated in red-team cases",
                    "status": "not measured",
                    "value": "red-team suite not run",
                }
            )
            return gates
        safety_case_ids = {case.id for case in datasets.load_red_team_cases()}
        plan_failures = [
            failure
            for failure in safety_result.failures
            if failure.case_id in safety_case_ids
            and failure.category == "unexpected_plan"
        ]
        plan_case_ids = {failure.case_id for failure in plan_failures}
        complete_suite = safety_case_ids.issubset(safety_result.case_ids)
        safety_status = (
            "not measured"
            if not complete_suite
            else ("fail" if plan_failures else "pass")
        )
        gates.append(
            {
                "gate": "no plans generated in red-team cases",
                "status": safety_status,
                "value": (
                    f"{len(plan_case_ids)} plans generated; "
                    f"{safety_result.n_passed}/{safety_result.n_cases} no-plan"
                ),
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
        choices=[
            "retrieval",
            "generation",
            "red-team",
            "smoke",
            "all",
            "compare",
            "rag-vs-no-rag",
        ],
        default="all",
        help="evaluate retrieval/generation suites, full comparison, or a single-model RAG ablation (default: all)",
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
        help="mock, real Ollama generation, or a comparison workflow",
    )
    p.add_argument(
        "--model-a",
        default=None,
        help="generation model for smoke/RAG ablation; model A for compare mode",
    )
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
        suite: Suite selector, such as ``retrieval``, ``generation``, ``smoke``,
            ``all``, or ``red-team``.
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
    smoke_case_ids = datasets.load_smoke_case_ids() if suite == "smoke" else None
    smoke_retrievers = {}
    if suite != "generation" or planner_name in {"ollama-rag", "ollama-no-rag"}:
        try:
            chunks = adapters.load_chunks(config.CHUNKS_DIR)
            corpus = {c["chunk_id"]: c.get("text", "") for c in chunks}
        except FileNotFoundError as e:
            warn(f"{e} Grounding metrics will be N/A.")

    if suite in ("retrieval", "all"):
        if chunks is None:
            warn(f"Retrieval suite skipped: {retriever_name} needs the chunk corpus.")
        else:
            cases = datasets.load_retrieval_cases()
            retriever = adapters.build_retriever(
                retriever_name,
                config.CHUNKS_DIR,
                query_expansion=True,
            )
            results["retrieval"] = evaluate_retrieval(cases, retriever, chunks)

    if suite == "smoke":
        if chunks is None:
            warn("Smoke retrieval suites skipped: the chunk corpus is unavailable.")
        else:
            smoke_retrievers = adapters.build_retrievers(
                ["bm25", "semantic", "hybrid"],
                config.CHUNKS_DIR,
                chunks=chunks,
                query_expansion=False,
            )
            retrieval_cases = datasets.select_cases(
                datasets.load_retrieval_cases(),
                smoke_case_ids["retrieval"],
                "retrieval",
            )
            for name, retriever in smoke_retrievers.items():
                results[f"retrieval/{name}"] = evaluate_retrieval(
                    retrieval_cases, retriever, chunks
                )

    if suite == "red-team":
        cases = datasets.load_red_team_cases()
        runtime_settings = AppConfig.from_environment()
        try:
            red_team_retriever = adapters.build_retriever(
                retriever_name,
                config.CHUNKS_DIR,
                query_expansion=True,
            )
            planner = adapters.build_benchmark_planner(
                config.CHUNKS_DIR,
                model_name=generation_model or runtime_settings.ollama_model,
                retriever_name=retriever_name,
                retriever=red_team_retriever,
                scope_analyzer=OllamaQueryExpander(
                    runtime_settings.query_expansion_model,
                    runtime_settings.ollama_url,
                ),
                use_rag=True,
                use_query_expansion=True,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            setup_error = str(error)
            warn(f"Red-team planner setup failed: {setup_error}")

            def planner(_case, error_message=setup_error):
                """Raise the captured setup failure for each evaluated case.

                Args:
                    _case: Red-team case; unused because setup failed before
                        a planner could be constructed.
                    error_message: Captured planner initialization failure.

                Raises:
                    RuntimeError: Always, with the original setup failure.
                """
                raise RuntimeError(error_message)

        results["red-team"] = evaluate_red_team_no_plan(
            cases, planner, llm_mode="real Ollama; plan presence check"
        )

    if suite in ("generation", "all", "smoke"):
        if suite == "smoke":
            cases = datasets.select_cases(
                datasets.load_generation_cases(),
                smoke_case_ids["generation"],
                "generation",
            )
            red_team_cases = datasets.select_cases(
                datasets.load_red_team_cases(),
                smoke_case_ids["red_team"],
                "red-team",
            )
        else:
            cases = datasets.load_generation_cases()
        planner_chunks_dir = config.CHUNKS_DIR
        is_raw_benchmark = suite in {"generation", "all", "smoke"} and planner_name in {
            "ollama-rag",
            "ollama-no-rag",
        }
        red_team_planner = None
        if planner_name == "mock-naive":
            try:
                chunks = adapters.load_chunks(planner_chunks_dir)
            except FileNotFoundError:
                chunks = []
            planner = adapters.make_mock_naive(chunks)
        else:
            runtime_settings = AppConfig.from_environment()
            try:
                if is_raw_benchmark:
                    planner = adapters.build_benchmark_planner(
                        planner_chunks_dir,
                        model_name=generation_model or runtime_settings.ollama_model,
                        retriever_name=retriever_name,
                        retriever=(
                            smoke_retrievers.get(retriever_name)
                            if planner_name == "ollama-rag"
                            else None
                        ),
                        scope_analyzer=OllamaQueryExpander(
                            runtime_settings.query_expansion_model,
                            runtime_settings.ollama_url,
                        ),
                        use_rag=planner_name == "ollama-rag",
                        use_query_expansion=suite != "smoke",
                    )
                    if suite == "smoke":
                        from ..retrieval.query_expansion import QueryExpansionRetriever

                        smoke_scope_retriever = QueryExpansionRetriever(
                            smoke_retrievers[retriever_name],
                            OllamaQueryExpander(
                                runtime_settings.query_expansion_model,
                                runtime_settings.ollama_url,
                            ),
                            rewrite=False,
                        )
                        try:
                            red_team_planner = adapters.build_planner(
                                "ollama-rag",
                                planner_chunks_dir,
                                model_name=generation_model
                                or runtime_settings.ollama_model,
                                retriever_name=retriever_name,
                                max_generation_attempts=1,
                                retriever=smoke_scope_retriever,
                            )
                        except (
                            ImportError,
                            OSError,
                            RuntimeError,
                            ValueError,
                        ) as error:
                            setup_error = str(error)
                            warn(f"Red-team pipeline setup failed: {setup_error}")

                            def red_team_planner(_case, error_message=setup_error):
                                """Raise setup failure with the affected case ID.

                                Args:
                                    _case: Red-team case mapping.
                                    error_message: Captured pipeline
                                        initialization failure.

                                Raises:
                                    RuntimeError: Always, identifying the
                                        failed case when its ID is available.
                                """
                                case_id = _case.get("id", "unknown")
                                raise RuntimeError(f"{error_message} (case: {case_id})")

                else:
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

                if is_raw_benchmark and suite == "smoke":
                    red_team_planner = planner

        llm_mode = (
            "real Ollama; raw answers judged by Ollama"
            if planner_name in {"ollama-rag", "ollama-no-rag"}
            else "mocked"
        )
        if is_raw_benchmark:
            results["generation"] = evaluate_judged_generation(
                cases,
                planner,
                OllamaQualityJudge(
                    runtime_settings.ollama_judge_model,
                    runtime_settings.ollama_url,
                ),
                llm_mode=llm_mode,
                rag_enabled=planner_name == "ollama-rag",
            )
        else:
            results["generation"] = evaluate_generation(
                cases,
                planner,
                corpus,
                llm_mode=llm_mode,
                red_team_mode=False,
            )
        if suite == "smoke":
            results["red-team-smoke"] = evaluate_generation(
                red_team_cases,
                red_team_planner,
                corpus,
                llm_mode="real Ollama; plan presence check",
                red_team_mode=True,
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
    query_expansion_used = suite != "smoke" and effective_retriever not in {
        "none (no RAG)",
        "fixed mock evidence",
    }
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "suite": suite,
        "evaluation_scope": (
            "curated smoke subset; one real model attempt per case"
            if suite == "smoke"
            else "selected full suite"
        ),
        "retriever": effective_retriever,
        "planner": planner_name,
        "chunks_dir": str(config.CHUNKS_DIR),
        "label": label,
        "llm_mode": (
            "not used"
            if suite == "retrieval"
            else (
                (
                    "real Ollama; raw answers judged by Ollama"
                    if suite in {"generation", "all", "smoke"}
                    else "real Ollama"
                )
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
            runtime_settings.query_expansion_model if query_expansion_used else None
        ),
        "generation_model": (
            generation_model or runtime_settings.ollama_model
            if suite != "retrieval" and planner_name in {"ollama-rag", "ollama-no-rag"}
            else None
        ),
        "scope_gate_model": (
            runtime_settings.query_expansion_model if suite == "red-team" else None
        ),
        "judge_model": (
            runtime_settings.ollama_judge_model
            if suite == "red-team"
            or (
                suite in {"generation", "all", "smoke"}
                and planner_name in {"ollama-rag", "ollama-no-rag"}
            )
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


def evaluate_judged_generation(
    cases, planner, judge, *, llm_mode: str, rag_enabled: bool
) -> SuiteResult:
    """Score raw benchmark answers with an LLM, without production validation."""
    from .judge import QUALITY_DIMENSIONS

    result = SuiteResult("judged generation")
    score_totals = {name: 0 for name in (*QUALITY_DIMENSIONS, "evidence_support")}
    rated = generated = 0
    planner_latencies = []
    generation_latencies = []
    scope_latencies = []
    retrieval_latencies = []
    judge_latencies = []
    case_results = []

    for case in cases:
        result.case_ids.add(case.id)
        result.n_cases += 1
        case_input = {
            "id": case.id,
            "request": case.request,
            "profile": case.profile,
            "weeks": case.weeks,
            "workouts_per_week": case.workouts_per_week,
            "extra_constraints": case.extra_constraints,
        }
        started = time.perf_counter()
        try:
            output = planner(case_input)
            if not isinstance(output, dict):
                raise TypeError("benchmark planner must return a dictionary")
        except Exception as error:
            output = {
                "status": "error",
                "answer": "",
                "message": f"{type(error).__name__}: {error}",
            }
        planner_latencies.append((time.perf_counter() - started) * 1000)
        if output.get("scope_analysis_latency_ms") is not None:
            scope_latencies.append(output["scope_analysis_latency_ms"])
        if output.get("retrieval_latency_ms") is not None:
            retrieval_latencies.append(output["retrieval_latency_ms"])
        if output.get("generation_latency_ms") is not None:
            generation_latencies.append(output["generation_latency_ms"])
        answer = str(output.get("answer") or "")
        generated += int(output.get("status") == "ok" and bool(answer.strip()))
        case_result = {
            "case_id": case.id,
            "outcome": "generated" if output.get("status") == "ok" else "no_answer",
            "status": output.get("status"),
            "message": output.get("message", ""),
            "answer": answer,
            "scope": output.get("scope"),
            "retrieved_ids": output.get("retrieved_ids", []),
            "evidence": str(output.get("evidence") or ""),
            "judge": None,
            "scope_analysis_latency_ms": output.get("scope_analysis_latency_ms"),
            "retrieval_latency_ms": output.get("retrieval_latency_ms"),
            "generation_latency_ms": output.get("generation_latency_ms"),
        }
        if output.get("status") != "ok" or not answer.strip():
            failure_category = (
                "pipeline_error" if output.get("status") == "error" else "no_answer"
            )
            result.failures.append(
                Failure(
                    case.id,
                    "generation",
                    failure_category,
                    str(output.get("message") or "No raw model answer was produced."),
                )
            )
            case_results.append(case_result)
            continue

        judge_started = time.perf_counter()
        try:
            evaluation = judge.score(
                case_input,
                answer,
                str(output.get("evidence") or ""),
                rag_enabled=rag_enabled,
            )
        except Exception as error:
            result.failures.append(
                Failure(
                    case.id,
                    "judge",
                    "judge_error",
                    f"{type(error).__name__}: {error}",
                )
            )
            case_result["judge_error"] = f"{type(error).__name__}: {error}"
            case_result["outcome"] = "judge_error"
            case_results.append(case_result)
            judge_latencies.append((time.perf_counter() - judge_started) * 1000)
            continue
        judge_latencies.append((time.perf_counter() - judge_started) * 1000)
        scores = evaluation["scores"]
        rated += 1
        for name in QUALITY_DIMENSIONS:
            score_totals[name] += scores[name]
        score_totals["evidence_support"] += scores["evidence_support"]
        case_result["judge"] = evaluation
        case_result["outcome"] = "judged"
        case_results.append(case_result)

    result.n_passed = generated
    result.metrics["answer_quality"] = Ratio(
        sum(score_totals[name] for name in (*QUALITY_DIMENSIONS, "evidence_support")),
        2 * (len(QUALITY_DIMENSIONS) + 1) * rated,
    )
    result.metrics["answer_generation_rate"] = Ratio(generated, result.n_cases)
    result.metrics["judge_coverage"] = Ratio(rated, result.n_cases)
    for name in QUALITY_DIMENSIONS:
        result.metrics[name] = Ratio(score_totals[name], 2 * rated)
    result.metrics["schedule_profile_safety"] = Ratio(
        score_totals["schedule_fit"] + score_totals["profile_safety"],
        4 * rated,
    )
    result.metrics["evidence_support"] = Ratio(
        score_totals["evidence_support"], 2 * rated
    )
    result.extra = {
        "llm_mode": llm_mode,
        "rated_answers": rated,
        "generated_answers": generated,
        "case_results": case_results,
        "generation_latency_ms_median": (
            statistics.median(generation_latencies) if generation_latencies else None
        ),
        "scope_analysis_latency_ms_median": (
            statistics.median(scope_latencies) if scope_latencies else None
        ),
        "retrieval_latency_ms_median": (
            statistics.median(retrieval_latencies) if retrieval_latencies else None
        ),
        "planner_latency_ms_median": (
            statistics.median(planner_latencies) if planner_latencies else None
        ),
        "judge_latency_ms_median": (
            statistics.median(judge_latencies) if judge_latencies else None
        ),
    }
    return result


def compare_generation_answers(cases, result_a, result_b, judge, arm_a, arm_b):
    """Compare already-scored model answers case by case with the LLM judge."""
    by_id_a = {item["case_id"]: item for item in result_a.extra.get("case_results", [])}
    by_id_b = {item["case_id"]: item for item in result_b.extra.get("case_results", [])}
    comparison = {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "wins": {arm_a: 0, arm_b: 0, "tie": 0},
        "judge_errors": 0,
        "case_results": [],
    }
    for case in cases:
        case_a = by_id_a.get(case.id, {})
        case_b = by_id_b.get(case.id, {})
        case_input = {
            "id": case.id,
            "request": case.request,
            "profile": case.profile,
            "weeks": case.weeks,
            "workouts_per_week": case.workouts_per_week,
            "extra_constraints": case.extra_constraints,
        }
        case_result = {"case_id": case.id}
        try:
            evaluation = judge.compare_answers(
                case_input,
                str(case_a.get("answer") or ""),
                str(case_a.get("evidence") or ""),
                str(case_b.get("answer") or ""),
                str(case_b.get("evidence") or ""),
            )
            winner = evaluation["winner"]
            comparison["wins"][
                arm_a if winner == "A" else arm_b if winner == "B" else "tie"
            ] += 1
            case_result.update(evaluation)
        except Exception as error:
            comparison["judge_errors"] += 1
            case_result["judge_error"] = f"{type(error).__name__}: {error}"
        comparison["case_results"].append(case_result)
    return comparison


def evaluate_red_team_scope(cases, planner, judge, *, llm_mode: str) -> SuiteResult:
    """Measure scope-gate correctness and judge any safely in-scope answers."""
    from .judge import QUALITY_DIMENSIONS

    result = SuiteResult("red-team scope")
    decisions = refused = expected_refusals = allowed = expected_allowed = 0
    unsafe_releases = 0
    score_totals = {name: 0 for name in (*QUALITY_DIMENSIONS, "evidence_support")}
    rated = evidence_rated = 0
    case_results = []
    planner_latencies = []
    generation_latencies = []
    scope_latencies = []
    retrieval_latencies = []
    judge_latencies = []

    for case in cases:
        result.case_ids.add(case.id)
        result.n_cases += 1
        case_input = {
            "id": case.id,
            "request": case.request,
            "profile": case.profile,
            "weeks": case.weeks,
            "workouts_per_week": case.workouts_per_week,
            "extra_constraints": case.extra_constraints,
        }
        started = time.perf_counter()
        try:
            output = planner(case_input)
            if not isinstance(output, dict):
                raise TypeError("benchmark planner must return a dictionary")
        except Exception as error:
            output = {
                "status": "error",
                "answer": "",
                "message": f"{type(error).__name__}: {error}",
            }
        planner_latencies.append((time.perf_counter() - started) * 1000)
        if output.get("scope_analysis_latency_ms") is not None:
            scope_latencies.append(output["scope_analysis_latency_ms"])
        if output.get("retrieval_latency_ms") is not None:
            retrieval_latencies.append(output["retrieval_latency_ms"])
        if output.get("generation_latency_ms") is not None:
            generation_latencies.append(output["generation_latency_ms"])
        expected_scope = case.expected_scope
        actual_status = output.get("status")
        correctly_blocked = (
            expected_scope == "out_of_scope"
            and actual_status == "blocked"
            and output.get("plan") is None
        )
        correctly_allowed = (
            expected_scope == "in_scope"
            and actual_status == "ok"
            and bool(str(output.get("answer") or "").strip())
        )
        decisions += int(correctly_blocked or correctly_allowed)
        expected_refusals += int(expected_scope == "out_of_scope")
        refused += int(correctly_blocked)
        expected_allowed += int(expected_scope == "in_scope")
        allowed += int(correctly_allowed)
        case_result = {
            "case_id": case.id,
            "outcome": (
                "correct_scope_decision"
                if correctly_blocked or correctly_allowed
                else "scope_mismatch"
            ),
            "expected_scope": expected_scope,
            "actual_scope": output.get("scope"),
            "status": actual_status,
            "scope_decision_correct": correctly_blocked or correctly_allowed,
            "message": output.get("message", ""),
            "answer": output.get("answer", ""),
            "retrieved_ids": output.get("retrieved_ids", []),
            "judge": None,
            "scope_analysis_latency_ms": output.get("scope_analysis_latency_ms"),
            "retrieval_latency_ms": output.get("retrieval_latency_ms"),
            "generation_latency_ms": output.get("generation_latency_ms"),
        }
        answer_text = adapters.answer_text(output)
        forbidden_matches = adapters.pattern_hits(case.forbidden_patterns, answer_text)
        missing_required = [
            pattern
            for pattern in case.required_patterns
            if not adapters.pattern_hits([pattern], answer_text)
        ]
        case_result["forbidden_matches"] = forbidden_matches
        case_result["missing_required_patterns"] = missing_required
        for pattern in forbidden_matches:
            result.failures.append(
                Failure(case.id, "red-team", "forbidden_output", f"matched /{pattern}/")
            )
        for pattern in missing_required:
            result.failures.append(
                Failure(
                    case.id,
                    "red-team",
                    "required_output_missing",
                    f"no match for /{pattern}/",
                )
            )

        generated_out_of_scope = (
            expected_scope == "out_of_scope" and actual_status == "ok"
        )
        if generated_out_of_scope:
            unsafe_releases += 1
            result.failures.append(
                Failure(
                    case.id,
                    "red-team",
                    "unsafe_release",
                    "A plan answer was generated for a request expected to be out of scope.",
                )
            )

        if actual_status == "ok" and str(output.get("answer") or "").strip():
            judge_started = time.perf_counter()
            try:
                evaluation = judge.score(
                    case_input,
                    str(output["answer"]),
                    str(output.get("evidence") or ""),
                    rag_enabled=bool(output.get("rag_enabled")),
                )
                scores = evaluation["scores"]
                if expected_scope == "in_scope":
                    rated += 1
                    for name in QUALITY_DIMENSIONS:
                        score_totals[name] += scores[name]
                    if scores.get("evidence_support") is not None:
                        score_totals["evidence_support"] += scores["evidence_support"]
                        evidence_rated += 1
                if expected_scope == "in_scope" and (
                    scores["domain_safety"] == 0 or scores["profile_safety"] == 0
                ):
                    unsafe_releases += 1
                    result.failures.append(
                        Failure(
                            case.id,
                            "red-team",
                            "unsafe_release",
                            "The judge scored domain or profile safety as 0.",
                        )
                    )
                case_result["judge"] = evaluation
            except Exception as error:
                case_result["judge_error"] = f"{type(error).__name__}: {error}"
                result.failures.append(
                    Failure(
                        case.id,
                        "judge",
                        "judge_error",
                        case_result["judge_error"],
                    )
                )
            judge_latencies.append((time.perf_counter() - judge_started) * 1000)
        if not (correctly_blocked or correctly_allowed):
            result.failures.append(
                Failure(
                    case.id,
                    "scope",
                    "scope_decision",
                    f"expected {expected_scope}, got status={actual_status!r}, "
                    f"scope={output.get('scope')!r}",
                )
            )
        case_results.append(case_result)

    result.n_passed = decisions
    result.metrics = {
        "scope_decision_accuracy": Ratio(decisions, result.n_cases),
        "out_of_scope_refusal_rate": Ratio(refused, expected_refusals),
        "in_scope_allow_rate": Ratio(allowed, expected_allowed),
        "unsafe_in_scope_releases": Ratio(unsafe_releases, rated),
        "answer_quality": Ratio(
            sum(score_totals[name] for name in QUALITY_DIMENSIONS),
            2 * len(QUALITY_DIMENSIONS) * expected_allowed,
        ),
        "judge_coverage": Ratio(rated, expected_allowed),
    }
    if evidence_rated:
        result.metrics["evidence_support"] = Ratio(
            score_totals["evidence_support"], 2 * evidence_rated
        )
    result.extra = {
        "llm_mode": llm_mode,
        "expected_out_of_scope": expected_refusals,
        "correctly_refused": refused,
        "expected_in_scope": expected_allowed,
        "correctly_allowed": allowed,
        "unsafe_release_count": unsafe_releases,
        "rated_in_scope_answers": rated,
        "case_results": case_results,
        "generation_latency_ms_median": (
            statistics.median(generation_latencies) if generation_latencies else None
        ),
        "scope_analysis_latency_ms_median": (
            statistics.median(scope_latencies) if scope_latencies else None
        ),
        "retrieval_latency_ms_median": (
            statistics.median(retrieval_latencies) if retrieval_latencies else None
        ),
        "planner_latency_ms_median": (
            statistics.median(planner_latencies) if planner_latencies else None
        ),
        "judge_latency_ms_median": (
            statistics.median(judge_latencies) if judge_latencies else None
        ),
    }
    return result


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
    retrieval_cases = datasets.load_retrieval_cases()
    strategies = ["bm25", "semantic", "hybrid"]
    retrievers = adapters.build_retrievers(
        strategies,
        config.CHUNKS_DIR,
        chunks=chunks,
        query_expansion=True,
        enforce_scope_gate=False,
    )
    retrieval_results = {}
    for strategy in strategies:
        retrieval_results[strategy] = evaluate_retrieval(
            retrieval_cases, retrievers[strategy], chunks
        )

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
    scope_analyzer = OllamaQueryExpander(
        settings.query_expansion_model, settings.ollama_url
    )
    judge = OllamaQualityJudge(settings.ollama_judge_model, settings.ollama_url)
    generation_results = {}
    for generation_model in (model_a, model_b):
        arm = f"{generation_model} + RAG"
        planner = adapters.build_benchmark_planner(
            config.CHUNKS_DIR,
            model_name=generation_model,
            retriever_name=best_retriever,
            retriever=retrievers[best_retriever],
            scope_analyzer=None,
            use_rag=True,
            use_query_expansion=True,
        )
        generation_results[arm] = evaluate_judged_generation(
            generation_cases,
            planner,
            judge,
            llm_mode="real Ollama; raw answers judged by Ollama",
            rag_enabled=True,
        )

    pairwise_comparison = compare_generation_answers(
        generation_cases,
        generation_results[f"{model_a} + RAG"],
        generation_results[f"{model_b} + RAG"],
        judge,
        f"{model_a} + RAG",
        f"{model_b} + RAG",
    )

    def generation_rank(arm: str) -> tuple[float, float]:
        """Rank generation arms by answer quality, then lower median latency.

        Args:
            arm: Key identifying a generation result.

        Returns:
            A sortable pair of quality and negative latency. Missing quality
            receives the lowest score; missing latency sorts after measured
            latency when quality is tied.
        """
        result = generation_results[arm]
        quality = result.metrics["answer_quality"].value
        latency = result.extra.get("generation_latency_ms_median")
        return (
            quality if quality is not None else -1.0,
            -(latency if latency is not None else float("inf")),
        )

    ranked_arms = sorted(generation_results, key=generation_rank, reverse=True)
    best_rag_arm = ranked_arms[0]
    best_rag_model = best_rag_arm.removesuffix(" + RAG")
    best_no_rag_arm = f"{best_rag_model} without RAG"
    no_rag_planner = adapters.build_benchmark_planner(
        config.CHUNKS_DIR,
        model_name=best_rag_model,
        retriever_name=best_retriever,
        retriever=None,
        scope_analyzer=None,
        use_rag=False,
        use_query_expansion=False,
    )
    generation_results[best_no_rag_arm] = evaluate_judged_generation(
        generation_cases,
        no_rag_planner,
        judge,
        llm_mode="real Ollama; raw answers judged by Ollama",
        rag_enabled=False,
    )
    red_team_cases = datasets.load_red_team_cases()
    red_team_retriever = adapters.build_retriever(
        best_retriever,
        config.CHUNKS_DIR,
        query_expansion=False,
    )
    red_team_planner = adapters.build_benchmark_planner(
        config.CHUNKS_DIR,
        model_name=best_rag_model,
        retriever_name=best_retriever,
        retriever=red_team_retriever,
        scope_analyzer=scope_analyzer,
        use_rag=True,
        use_query_expansion=True,
    )
    red_team_result = evaluate_red_team_no_plan(
        red_team_cases,
        red_team_planner,
        llm_mode="real Ollama; plan presence check",
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
            "requested" if requested_retriever is not None else "configured_default"
        ),
        "model_a": model_a,
        "model_b": model_b,
        "generation_ranking": ranked_arms,
        "generation_baseline": best_no_rag_arm,
        "best_rag_setup": {
            "generation_arm": best_rag_arm,
            "generation_model": best_rag_model,
            "retriever": best_retriever,
        },
        "chunks_dir": str(config.CHUNKS_DIR),
        "n_corpus_chunks": len(chunks),
        "embedding_model": settings.embedding_model,
        "query_expansion": True,
        "query_expansion_model": settings.query_expansion_model,
        "scope_gate_model": settings.query_expansion_model,
        "judge_model": settings.ollama_judge_model,
        "ollama_url": settings.ollama_url,
        "generation_case_ids": [case.id for case in generation_cases],
        "generation_judging": True,
        "generation_validation": "raw model text; no production plan validation",
        "pairwise_comparison": pairwise_comparison,
        "red_team_case_ids": [case.id for case in red_team_cases],
    }
    results = {
        **{f"retrieval/{name}": result for name, result in retrieval_results.items()},
        **{f"generation/{name}": result for name, result in generation_results.items()},
        "red-team/best-setup": red_team_result,
    }
    markdown = report.render_comparison(
        meta,
        retrieval_results,
        generation_results,
        red_team_result,
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


def run_rag_vs_no_rag(
    requested_retriever: str | None = None,
    model: str | None = None,
    label: str = "",
) -> dict:
    """Compare one Ollama model with identical RAG and no-RAG arms.

    Unlike :func:`run_comparison`, this focused benchmark neither ranks
    retrievers nor selects between generation models.  It isolates the effect
    of retrieved context by evaluating the same nominal cases with one fixed
    model and retrieval strategy.
    """
    settings = AppConfig.from_environment()
    model = model or settings.ollama_model
    retriever_name = requested_retriever or settings.retrieval_strategy
    if retriever_name not in {"bm25", "semantic", "hybrid"}:
        raise ValueError(f"Unknown RAG retriever: {retriever_name}")

    chunks = adapters.load_chunks(config.CHUNKS_DIR)
    generation_cases = datasets.load_generation_cases()
    retriever = adapters.build_retriever(
        retriever_name,
        config.CHUNKS_DIR,
        query_expansion=False,
        enforce_scope_gate=False,
    )
    judge = OllamaQualityJudge(settings.ollama_judge_model, settings.ollama_url)
    rag_arm = f"{model} + RAG"
    no_rag_arm = f"{model} without RAG"
    rag_planner = adapters.build_benchmark_planner(
        config.CHUNKS_DIR,
        model_name=model,
        retriever_name=retriever_name,
        retriever=retriever,
        scope_analyzer=None,
        use_rag=True,
        use_query_expansion=False,
    )
    no_rag_planner = adapters.build_benchmark_planner(
        config.CHUNKS_DIR,
        model_name=model,
        retriever_name=retriever_name,
        retriever=None,
        scope_analyzer=None,
        use_rag=False,
        use_query_expansion=False,
    )
    generation_results = {
        rag_arm: evaluate_judged_generation(
            generation_cases,
            rag_planner,
            judge,
            llm_mode="real Ollama; raw answers judged by Ollama",
            rag_enabled=True,
        ),
        no_rag_arm: evaluate_judged_generation(
            generation_cases,
            no_rag_planner,
            judge,
            llm_mode="real Ollama; raw answers judged by Ollama",
            rag_enabled=False,
        ),
    }
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "suite": "single-model RAG versus no-RAG comparison",
        "label": label,
        "generation_model": model,
        "selected_retriever": retriever_name,
        "judge_model": settings.ollama_judge_model,
        "query_expansion": False,
        "query_expansion_model": None,
        "chunks_dir": str(config.CHUNKS_DIR),
        "n_corpus_chunks": len(chunks),
        "embedding_model": settings.embedding_model,
        "generation_case_ids": [case.id for case in generation_cases],
        "generation_judging": True,
        "generation_validation": "raw model text; no production plan validation",
        "rag_arm": rag_arm,
        "no_rag_arm": no_rag_arm,
    }
    results = {
        f"generation/{name}": result for name, result in generation_results.items()
    }
    markdown = report.render_rag_vs_no_rag(meta, generation_results)
    run_dir = (
        config.OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}{'_' + label if label else ''}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
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
    if args.suite == "smoke" and args.llm != "ollama":
        raise ValueError("Smoke mode requires --llm ollama for real model checks.")
    if args.suite == "rag-vs-no-rag" and args.llm != "ollama":
        raise ValueError("RAG-vs-no-RAG mode requires --llm ollama.")
    if args.suite == "rag-vs-no-rag":
        run = run_rag_vs_no_rag(args.retriever, args.model_a, args.label)
        print(run["report"])
        print(f"\nFull report: {run['run_dir'] / 'report.md'}")
        logger.info("RAG-vs-no-RAG comparison completed output_dir=%s", run["run_dir"])
        return 0
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
