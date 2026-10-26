"""Benchmark: evaluate several systems on the same test cases and rank them.

    python -m src.evaluation.benchmark

The systems are listed in BENCHMARK in config/evaluation_config.py. Each one is run
with runner.run_evaluation, exactly like the normal evaluation command, so all
systems are scored on the same cases with the same metrics. The ranking is
written to outputs/benchmark_runs/<date>/ranking.md and ranking.json; the full
results of each system stay in its own run folder (outputs/eval_runs/).
"""

import json
import re
import sys
from datetime import datetime

from config import evaluation_config as config
from . import runner

# Columns of the ranking table: (suite, metric, header). "pass_rate" and the
# latency come from outside the metrics dict, see metric_value().
COLUMNS = [
    ("retrieval", "recall@5", "Recall@5 ↑"),
    ("retrieval", "mrr", "MRR ↑"),
    ("retrieval", "hit@1", "Hit@1 ↑"),
    ("retrieval", "latency_ms_median", "Retrieval ms ↓"),
    ("generation", "pass_rate", "Pass rate ↑"),
    ("generation", "faithfulness", "Faithfulness ↑"),
    ("generation", "citation_precision", "Citation precision ↑"),
    ("generation", "abstention_accuracy", "Abstention accuracy ↑"),
    ("generation", "red_team_pass_rate", "Red team ↑"),
    ("generation", "latency_ms_median", "Generation ms ↓"),
]


def metric_value(payload: dict, suite: str, metric: str) -> float | None:
    """Read one number from a run's results.json content; None when the suite or metric is absent (N/A)."""
    s = payload["suites"].get(suite)
    if s is None:
        return None
    if metric == "pass_rate":
        return s["pass_rate"]["value"]
    if metric in s["metrics"]:
        return s["metrics"][metric]["value"]
    return s["extra"].get(metric)


def rank(rows: list[dict]) -> list[dict]:
    """Sort systems by BENCHMARK_RANK_BY, best first. A missing value (N/A) counts as worse than any number."""
    def key(row):
        return tuple((row[(s, m)] is not None, row[(s, m)] or 0) for s, m in config.BENCHMARK_RANK_BY)
    return sorted(rows, key=key, reverse=True)


def fmt(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}" if isinstance(value, float) and value > 1 else f"{value:.3f}"


def render(ranked: list[dict]) -> str:
    """One Markdown table: one line per system, best first."""
    order = ", then ".join(f"{m} ({s})" for s, m in config.BENCHMARK_RANK_BY)
    lines = [
        "# RepWise benchmark",
        "",
        f"Ranked by {order}. N/A means the suite or metric could not be computed for that system.",
        "",
        "| Rank | System | " + " | ".join(h for _, _, h in COLUMNS) + " |",
        "|---|---|" + "---|" * len(COLUMNS),
    ]
    for i, row in enumerate(ranked, start=1):
        cells = " | ".join(fmt(row[(s, m)]) for s, m, _ in COLUMNS)
        lines.append(f"| {i} | {row['name']} | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = []
    for system in config.BENCHMARK:
        print(f"== {system['name']}")
        label = re.sub(r"[^A-Za-z0-9_-]+", "-", system["name"]).strip("-")
        run = runner.run_evaluation(config.BENCHMARK_SUITE, system["retriever"], system["planner"], label)
        row = {"name": system["name"], "retriever": system["retriever"], "planner": system["planner"],
               "run_dir": str(run["run_dir"])}
        for suite, metric in {(s, m) for s, m, _ in COLUMNS} | set(config.BENCHMARK_RANK_BY):
            row[(suite, metric)] = metric_value(run["payload"], suite, metric)
        rows.append(row)

    ranked = rank(rows)
    out_dir = config.BENCHMARK_OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ranking.md").write_text(render(ranked), encoding="utf-8")
    as_json = [{k if isinstance(k, str) else f"{k[0]}.{k[1]}": v for k, v in row.items()} for row in ranked]
    (out_dir / "ranking.json").write_text(json.dumps(as_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nRanking: {out_dir / 'ranking.md'}")
    for i, row in enumerate(ranked, start=1):
        print(f"  {i}. {row['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
