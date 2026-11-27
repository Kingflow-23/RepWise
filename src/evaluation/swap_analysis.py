"""Swap analysis: does the LLM judge change its scores when the rubric is reordered?

The judge re-scores answers already saved by the generation suite; nothing is
regenerated. Only the presentation changes: in every rubric dimension the score
levels are listed in one of the six orders of (0, 1, 2), e.g.
``"2 = ...; 0 = ...; 1 = ..."``. Score meanings stay the same, so a stable judge
gives the same scores in every order.

Reported:
- answer quality per order and arm (five general dimensions; evidence support
  is excluded),
- whether the arm ranking changes with the order alone,
- per arm and dimension, how many cases changed score and the largest range.

A case missing any arm/order score is excluded from every comparison; judge
errors are never counted as zero. Calls run in a shuffled order with a
recorded seed.

Run:  python -m src.evaluation.swap_analysis [generation results.json]
      (default: the latest generation run)
"""

import hashlib
import itertools
import json
import logging
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import evaluation_config as config
from config.config import AppConfig

from .datasets import GENERATION_CASES
from .judge import QUALITY_DIMENSIONS, OllamaQualityJudge, judge_input, make_rubric
from .report import write_run

logger = logging.getLogger(__name__)

ORDERS = list(itertools.permutations((0, 1, 2)))  # 6 presentation orders
SCORED = (*QUALITY_DIMENSIONS, "evidence_support")
SEED = 42


def order_name(order) -> str:
    return "-".join(map(str, order))


def latest_generation_results() -> Path:
    """``results.json`` of the most recent generation run."""
    runs = sorted(config.OUT_DIR.glob("*_generation*/results.json"))
    if not runs:
        raise FileNotFoundError(
            "No generation run found: run python -m src.evaluation.generation first."
        )
    return runs[-1]


def load_inputs(source: dict, arms: list[str], case_ids: list[str]) -> list[dict]:
    """Freeze the saved request, answer and evidence of every arm and case."""
    inputs = []
    for arm in arms:
        saved = {r["case_id"]: r for r in source["arms"][arm]["case_results"]}
        for case_id in case_ids:
            result = saved[case_id]
            inputs.append(
                {
                    "case_id": case_id,
                    "arm": arm,
                    "request": judge_input(
                        case_id, result["query"], source["meta"]["profile"]
                    ),
                    "answer": result.get("answer", ""),
                    "evidence": result.get("evidence", ""),
                }
            )
    return inputs


def run_judgments(inputs: list[dict], judge, seed: int = SEED) -> list[dict]:
    """Score every input once per order, in a seeded random execution order."""
    tasks = [(item, order) for item in inputs for order in ORDERS]
    random.Random(seed).shuffle(tasks)
    judgments = []
    for number, (item, order) in enumerate(tasks, start=1):
        row = {"arm": item["arm"], "case_id": item["case_id"], "order": list(order)}
        try:
            row["judge"] = judge.score(
                item["request"], item["answer"], item["evidence"], make_rubric(order)
            )
        except Exception as error:
            row["judge_error"] = f"{type(error).__name__}: {error}"
        logger.info(
            "Swap judgment %d/%d order=%s", number, len(tasks), order_name(order)
        )
        judgments.append(row)
    return judgments


def summarize(judgments: list[dict], arms: list[str], case_ids: list[str]) -> dict:
    """Answer quality per order, ranking changes and per-dimension stability."""
    scores = {
        (j["arm"], j["case_id"], tuple(j["order"])): j["judge"]["scores"]
        for j in judgments
        if "judge" in j
    }
    complete = [
        case_id
        for case_id in case_ids
        if all((arm, case_id, order) in scores for arm in arms for order in ORDERS)
    ]

    by_order = []
    for order in ORDERS:
        stats = {}
        for arm in arms:
            rows = [scores[(arm, case_id, order)] for case_id in complete]
            stats[arm] = {
                "dimension_means": {
                    d: statistics.mean(r[d] for r in rows) if rows else None
                    for d in SCORED
                },
                "answer_quality": (
                    sum(r[d] for r in rows for d in QUALITY_DIMENSIONS)
                    / (2 * len(QUALITY_DIMENSIONS) * len(rows))
                    if rows
                    else None
                ),
            }
        by_order.append(
            {"order": list(order), "arms": stats, "ranking": _ranking(stats)}
        )

    stability = []
    for arm in arms:
        for dimension in SCORED:
            values = [
                [scores[(arm, case_id, order)][dimension] for order in ORDERS]
                for case_id in complete
            ]
            stability.append(
                {
                    "arm": arm,
                    "dimension": dimension,
                    "compared_cases": len(values),
                    "changed_cases": sum(len(set(v)) > 1 for v in values),
                    "max_score_range": max(
                        (max(v) - min(v) for v in values), default=0
                    ),
                }
            )

    return {
        "complete_case_ids": complete,
        "excluded_case_ids": [c for c in case_ids if c not in complete],
        "judge_errors": sum("judge_error" in j for j in judgments),
        "by_order": by_order,
        "stability": stability,
        "ranking_changes": len({json.dumps(o["ranking"]) for o in by_order}) > 1,
    }


def _ranking(stats: dict) -> list[list[str]]:
    """Arms from best to worst quality; equal scores share a rank."""
    groups: dict[float, list[str]] = {}
    for arm, s in stats.items():
        if s["answer_quality"] is not None:
            groups.setdefault(round(s["answer_quality"], 9), []).append(arm)
    return [sorted(groups[q]) for q in sorted(groups, reverse=True)]


def run_swap_analysis(
    source_path: Path | None = None,
    *,
    arms: list[str] | None = None,
    n_cases: int | None = 5,
    seed: int = SEED,
    label: str = "",
):
    """Re-score saved generation answers in all six rubric orders.

    Args:
        source_path: Generation ``results.json``; defaults to the latest run.
        arms: Arms to compare; defaults to the RAG arms of the source run.
        n_cases: Use the first ``n_cases`` cases (``None`` for all); each case
            costs ``6 x len(arms)`` judge calls.
        seed: Seed of the shuffled execution order.
        label: Optional suffix for the run folder.

    Returns:
        The run folder with ``report.md`` and ``results.json``.
    """
    source_path = Path(source_path or latest_generation_results())
    source = json.loads(source_path.read_text(encoding="utf-8"))
    arms = arms or source["ranking"]
    case_ids = source["meta"]["case_ids"][:n_cases]
    settings = AppConfig.from_environment()
    judge = OllamaQualityJudge(settings.ollama_judge_model, settings.ollama_url)

    inputs = load_inputs(source, arms, case_ids)
    judgments = run_judgments(inputs, judge, seed)
    payload = {
        "source": source_path,
        "source_sha256": _sha256(source_path),
        "dataset": GENERATION_CASES,
        "dataset_sha256": _sha256(GENERATION_CASES),
        "case_ids": case_ids,
        "arms": arms,
        "inputs": inputs,
        "selection": f"first {n_cases or len(case_ids)} source case IDs",
        "judge_model": settings.ollama_judge_model,
        "judge_url": settings.ollama_url,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "expected_calls": len(inputs) * len(ORDERS),
        "judgments": judgments,
        "rubrics": {order_name(order): make_rubric(order) for order in ORDERS},
        "summary": summarize(judgments, arms, case_ids),
    }
    return write_run("swap_analysis", payload, render_markdown(payload), label)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_markdown(payload: dict) -> str:
    """Readable summary in the format of the earlier rubric-order experiment."""
    summary = payload["summary"]
    n_cases = len(payload["case_ids"])
    done = payload["expected_calls"] - summary["judge_errors"]
    lines = [
        "# Swap analysis (rubric-order experiment)",
        "",
        f"Judge: {payload['judge_model']}",
        f"Cases: {', '.join(payload['case_ids'])}",
        f"Completed judge calls: {done}/{payload['expected_calls']}",
        f"Judge errors: {summary['judge_errors']}",
        f"Cases eligible for comparison: {len(summary['complete_case_ids'])}/{n_cases}",
        "",
        "The same saved answers, evidence and requests are used in every order.",
        "Score meanings are unchanged. Evidence support is excluded from answer quality.",
        "Cases missing any arm/order score are excluded from all comparisons; "
        "errors are not zero scores.",
        "",
        "| Order | Arm | Answer quality |",
        "|---|---|---|",
    ]
    for row in summary["by_order"]:
        for arm, stats in row["arms"].items():
            quality = stats["answer_quality"]
            lines.append(
                f"| {order_name(row['order'])} | {arm} "
                f"| {'N/A' if quality is None else f'{quality:.4f}'} |"
            )
    lines += [
        "",
        f"Model ranking changes across orders: {summary['ranking_changes']}",
        "",
        "| Arm | Dimension | Cases with changed scores | Largest score range |",
        "|---|---|---|---|",
    ]
    for s in summary["stability"]:
        lines.append(
            f"| {s['arm']} | {s['dimension']} "
            f"| {s['changed_cases']}/{s['compared_cases']} | {s['max_score_range']} |"
        )
    lines += ["", "Per-order rankings (equal scores remain tied):", ""]
    for row in summary["by_order"]:
        ranking = " > ".join(" = ".join(group) for group in row["ranking"]) or "N/A"
        lines.append(f"- {order_name(row['order'])}: {ranking}")
    lines += [
        "",
        f"This {n_cases}-case diagnostic measures order sensitivity. Stable scores "
        "do not prove correctness or absence of other biases.",
        f"Execution order is shuffled with seed {payload['seed']}; there are no "
        "generation calls.",
        "The source generation results and the generation dataset hashes are recorded.",
        "Raw judgments, rationales, exact rubrics, dimension means and frozen inputs "
        "are in results.json.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from ..observability.observability import configure_logging

    configure_logging("evaluation")
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    print(f"Swap analysis report: {run_swap_analysis(source)}")
