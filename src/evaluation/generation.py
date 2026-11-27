"""Generation evaluation: are the raw answers good, and does RAG help?

Each *arm* (a model with RAG, or a model without RAG) answers the same
questions for the fixed ``EVAL_PROFILE``. Raw answers are scored by the local
LLM judge on the rubric in ``judge.py``; no production plan validation runs.

Steps:
1. Every model answers with RAG; arms are ranked by answer quality, then
   median generation latency.
2. With two models, the judge also compares their answers head to head.
3. The best RAG model answers again without retrieved evidence (baseline).

Answer quality sums the six rubric dimensions (0-2 each, evidence support
included) over judged answers. A no-RAG answer cannot earn evidence support.

Run:  python -m src.evaluation.generation
"""

import logging
import time
from typing import Callable

from config import evaluation_config as config
from config.config import AppConfig

from .datasets import GenerationCase, load_generation_cases
from .judge import QUALITY_DIMENSIONS, OllamaQualityJudge
from .metrics.common import Ratio
from .report import median, ms, write_run

logger = logging.getLogger(__name__)

RUBRIC = (*QUALITY_DIMENSIONS, "evidence_support")
LATENCIES = ("total_ms", "scope_analysis_ms", "retrieval_ms", "generation_ms", "judge_ms")


def make_request(query: str):
    """Wrap a dataset question in the production request schema."""
    from ..schemas.models import PlanRequest, UserProfile

    return PlanRequest(focus=query, profile=UserProfile(**config.EVAL_PROFILE))


def judge_input(case) -> dict:
    """What the judge sees as the request: the question and the profile."""
    return {"id": case.id, "request": case.query, "profile": config.EVAL_PROFILE}


def generate_answer(
    query: str, generator, *, retriever=None, scope_analyzer=None, top_k: int = 5
) -> dict:
    """Produce one raw answer: optional scope gate, optional retrieval, generation.

    Args:
        query: User question.
        generator: ``OllamaGenerator`` (uses ``generate_text``).
        retriever: Production retriever; ``None`` means no RAG.
        scope_analyzer: ``OllamaQueryExpander``; when set, out-of-scope or
            unclear requests are blocked and in-scope ones use its French
            search query.
        top_k: Number of evidence chunks given to the model.

    Returns:
        ``status`` (``ok`` or ``blocked``), ``answer``, ``evidence``,
        ``retrieved_ids``, ``scope`` and per-stage latencies in ms.
    """
    from ..rag.context import make_highlight_query, make_query, select_evidence
    from ..rag.prompt import make_benchmark_prompts

    request = make_request(query)
    search_query = make_query(request)
    out = {"status": "ok", "scope": "unchecked", "evidence": "", "retrieved_ids": []}

    if scope_analyzer is not None:
        started = time.perf_counter()
        analysis = scope_analyzer.analyze(search_query)
        out["scope_analysis_ms"] = (time.perf_counter() - started) * 1000
        out["scope"] = analysis.scope
        if analysis.scope != "in_scope":
            return {**out, "status": "blocked", "answer": analysis.user_message}
        search_query = analysis.search_query or search_query

    if retriever is not None:
        started = time.perf_counter()
        chunks = retriever.search(search_query, top_k)
        out["evidence"], _ = select_evidence(
            chunks, query=make_highlight_query(request, retrieval_query=search_query)
        )
        out["retrieved_ids"] = [chunk.chunk_id for chunk in chunks]
        out["retrieval_ms"] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    out["answer"] = generator.generate_text(
        *make_benchmark_prompts(request, out["evidence"])
    )
    out["generation_ms"] = (time.perf_counter() - started) * 1000
    return out


def answer_case(case, answer_fn: Callable[[str], dict]) -> dict:
    """Run ``answer_fn`` on one case; a crash becomes an ``error`` result."""
    started = time.perf_counter()
    try:
        output = answer_fn(case.query)
    except Exception as error:  # a crash is a result of the system under test
        output = {"status": "error", "answer": "", "error": f"{type(error).__name__}: {error}"}
    output["total_ms"] = (time.perf_counter() - started) * 1000
    return {"case_id": case.id, "query": case.query, "judge": None, **output}


def judge_case(result: dict, case, judge) -> dict | None:
    """Score a released answer in place; return the scores, or ``None`` on error."""
    started = time.perf_counter()
    try:
        result["judge"] = judge.score(
            judge_input(case), result["answer"], result.get("evidence", "")
        )
    except Exception as error:
        result["judge_error"] = f"{type(error).__name__}: {error}"
    result["judge_ms"] = (time.perf_counter() - started) * 1000
    return result["judge"]["scores"] if result["judge"] else None


def evaluate_arm(
    cases: list[GenerationCase], answer_fn: Callable[[str], dict], judge
) -> dict:
    """Answer and judge every case for one arm.

    Returns:
        ``metrics`` (Ratios), median ``latency_ms`` per stage and
        ``case_results`` (answer, evidence, judge scores and rationale).
    """
    totals = dict.fromkeys(RUBRIC, 0)
    generated = rated = 0
    case_results = []
    for case in cases:
        result = answer_case(case, answer_fn)
        case_results.append(result)
        if result["status"] != "ok" or not result["answer"].strip():
            result["outcome"] = "pipeline_error" if result["status"] == "error" else "no_answer"
            continue
        generated += 1
        scores = judge_case(result, case, judge)
        if scores is None:
            result["outcome"] = "judge_error"
            continue
        result["outcome"] = "judged"
        rated += 1
        for name in RUBRIC:
            totals[name] += scores[name]

    metrics = {
        "answer_quality": Ratio(sum(totals.values()), 2 * len(RUBRIC) * rated),
        "answer_generation_rate": Ratio(generated, len(cases)),
        "judge_coverage": Ratio(rated, len(cases)),
        **{name: Ratio(totals[name], 2 * rated) for name in RUBRIC},
        "schedule_profile_safety": Ratio(
            totals["schedule_fit"] + totals["profile_safety"], 4 * rated
        ),
    }
    return {
        "metrics": metrics,
        "latency_ms": {
            key: median([r.get(key) for r in case_results]) for key in LATENCIES
        },
        "case_results": case_results,
    }


def compare_arms(cases, arm_a: str, result_a: dict, arm_b: str, result_b: dict, judge) -> dict:
    """Ask the judge which of two arms answered each case better."""
    answers_a = {r["case_id"]: r for r in result_a["case_results"]}
    answers_b = {r["case_id"]: r for r in result_b["case_results"]}
    comparison = {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "wins": {arm_a: 0, arm_b: 0, "tie": 0},
        "judge_errors": 0,
        "case_results": [],
    }
    for case in cases:
        a, b = answers_a[case.id], answers_b[case.id]
        row = {"case_id": case.id}
        try:
            verdict = judge.compare_answers(
                judge_input(case),
                a.get("answer", ""),
                a.get("evidence", ""),
                b.get("answer", ""),
                b.get("evidence", ""),
            )
            winner = {"A": arm_a, "B": arm_b}.get(verdict["winner"], "tie")
            comparison["wins"][winner] += 1
            row.update(verdict)
        except Exception as error:
            comparison["judge_errors"] += 1
            row["judge_error"] = f"{type(error).__name__}: {error}"
        comparison["case_results"].append(row)
    return comparison


def run_generation(
    models: list[str] | None = None,
    *,
    retriever_name: str | None = None,
    no_rag_baseline: bool = True,
    label: str = "",
):
    """Run the generation arms and write ``report.md`` + ``results.json``.

    Args:
        models: Ollama models to compare with RAG; defaults to the configured
            main and comparison models.
        retriever_name: ``bm25``, ``semantic`` or ``hybrid``; defaults to the
            application's strategy.
        no_rag_baseline: Also run the best RAG model without retrieval.
        label: Optional suffix for the run folder.

    Returns:
        The run folder.
    """
    from ..llm.ollama import OllamaGenerator
    from ..retrieval.semantic import load_chunks
    from .retrieval import build_retrievers

    settings = AppConfig.from_environment()
    models = models or [settings.ollama_model, settings.ollama_comparison_model]
    retriever_name = retriever_name or settings.retrieval_strategy
    cases = load_generation_cases()
    retriever = build_retrievers([retriever_name], load_chunks(config.CHUNKS_DIR))[
        retriever_name
    ]
    judge = OllamaQualityJudge(settings.ollama_judge_model, settings.ollama_url)

    def arm(model: str, rag: bool) -> dict:
        logger.info("Generation arm started model=%s rag=%s", model, rag)
        generator = OllamaGenerator(model, settings.ollama_url)
        return evaluate_arm(
            cases,
            lambda query: generate_answer(
                query,
                generator,
                retriever=retriever if rag else None,
                top_k=settings.top_k,
            ),
            judge,
        )

    arms = {f"{model} + RAG": arm(model, rag=True) for model in models}

    def rank_key(name: str) -> tuple[float, float]:
        latency = arms[name]["latency_ms"]["generation_ms"]
        return (
            arms[name]["metrics"]["answer_quality"].value or 0.0,
            -(latency if latency is not None else float("inf")),
        )

    ranking = sorted(arms, key=rank_key, reverse=True)
    pairwise = None
    if len(models) == 2:
        a, b = (f"{model} + RAG" for model in models)
        pairwise = compare_arms(cases, a, arms[a], b, arms[b], judge)
    baseline = None
    if no_rag_baseline:
        best_model = ranking[0].removesuffix(" + RAG")
        baseline = f"{best_model} without RAG"
        arms[baseline] = arm(best_model, rag=False)

    payload = {
        "meta": {
            "suite": "generation",
            "label": label,
            "models": models,
            "retriever": retriever_name,
            "top_k": settings.top_k,
            "query_expansion": False,
            "judge_model": settings.ollama_judge_model,
            "profile": config.EVAL_PROFILE,
            "case_ids": [case.id for case in cases],
            "validation": "raw model text; no production plan validation",
        },
        "ranking": ranking,
        "baseline": baseline,
        "arms": arms,
        "pairwise": pairwise,
    }
    return write_run("generation", payload, render_markdown(payload), label)


def render_markdown(payload: dict) -> str:
    """Readable summary: arm table, head-to-head choices, RAG vs no RAG."""
    meta, arms = payload["meta"], payload["arms"]
    n_cases = len(meta["case_ids"])
    lines = [
        "# Generation evaluation",
        "",
        f"- **Cases:** {n_cases} identical questions per arm",
        f"- **Models:** {', '.join(f'`{m}`' for m in meta['models'])}",
        f"- **Answer judge:** `{meta['judge_model']}`",
        f"- **Retriever:** `{meta['retriever']}` (top {meta['top_k']}), "
        "query expansion off",
        "",
        "Raw answers are judged without production plan validation. Scores are "
        "earned / maximum points, normalized to 0-1. Missing answers and judge "
        "errors are shown by generation rate and judge coverage.",
        "",
        "| Rank | Arm | Answer quality | Evidence support | Request fit "
        "| Schedule/profile safety | Generated | Judge coverage | Median generation |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = payload["ranking"] + ([payload["baseline"]] if payload["baseline"] else [])
    for rank, name in enumerate(order, start=1):
        m = arms[name]["metrics"]
        lines.append(
            f"| {rank if name in payload['ranking'] else 'baseline'} | `{name}` "
            f"| {m['answer_quality'].fmt()} | {m['evidence_support'].fmt()} "
            f"| {m['request_fit'].fmt()} | {m['schedule_profile_safety'].fmt()} "
            f"| {m['answer_generation_rate'].fmt()} | {m['judge_coverage'].fmt()} "
            f"| {ms(arms[name]['latency_ms']['generation_ms'])} |"
        )

    lines += ["", "## Scores by rubric dimension", ""]
    lines += ["| Arm | " + " | ".join(RUBRIC) + " |", "|---|" + "---:|" * len(RUBRIC)]
    for name in order:
        m = arms[name]["metrics"]
        lines.append(f"| `{name}` | " + " | ".join(m[d].fmt() for d in RUBRIC) + " |")

    pairwise = payload["pairwise"]
    if pairwise:
        a, b = pairwise["arm_a"], pairwise["arm_b"]
        wins = pairwise["wins"]
        lines += [
            "",
            "## Head-to-head judge choice",
            "",
            f"A = `{a}`; B = `{b}`. Preferences: `{a}` {wins[a]} · `{b}` {wins[b]} "
            f"· ties {wins['tie']} · judge errors {pairwise['judge_errors']}.",
            "",
            "| Case | Choice | A score | B score | Explanation |",
            "|---|---|---:|---:|---|",
        ]
        for row in pairwise["case_results"]:
            if "judge_error" in row:
                lines.append(f"| {row['case_id']} | judge error | — | — | {row['judge_error']} |")
                continue
            score_a, score_b = (sum(row["scores"][x].values()) for x in ("A", "B"))
            rationale = " ".join(row["rationale"].split()).replace("|", "/")
            lines.append(
                f"| {row['case_id']} | {row['winner']} | {score_a} | {score_b} | {rationale} |"
            )

    if payload["baseline"]:
        rag, no_rag = arms[payload["ranking"][0]]["metrics"], arms[payload["baseline"]]["metrics"]
        lines += [
            "",
            "## RAG vs no RAG",
            "",
            f"`{payload['ranking'][0]}` vs `{payload['baseline']}`: answer quality "
            f"{rag['answer_quality'].fmt()} vs {no_rag['answer_quality'].fmt()}; "
            f"evidence support {rag['evidence_support'].fmt()} vs "
            f"{no_rag['evidence_support'].fmt()}. Without retrieved evidence, an "
            "answer cannot earn evidence support, which counts toward answer quality.",
        ]

    lines += [
        "",
        "## Failed cases",
        "",
    ]
    failed = [
        (name, r)
        for name in order
        for r in arms[name]["case_results"]
        if r["outcome"] != "judged"
    ]
    if failed:
        lines += ["| Arm | Case | Outcome | Detail |", "|---|---|---|---|"]
        for name, r in failed:
            detail = r.get("error") or r.get("judge_error") or "no answer"
            lines.append(f"| `{name}` | {r['case_id']} | {r['outcome']} | {detail} |")
    else:
        lines.append("None: every answer was generated and judged.")
    lines += [
        "",
        "Raw answers, evidence, per-dimension scores and judge rationales are in "
        "`results.json`. This is a fixed-dataset measurement, not a significance test.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from ..observability.observability import configure_logging

    configure_logging("evaluation")
    print(f"Generation report: {run_generation()}")
