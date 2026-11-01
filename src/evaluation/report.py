"""Readable summaries of an evaluation run: a Markdown report and a short console view."""

from collections import Counter
from typing import Any

# Direction of each metric: True = higher is better, False = lower is better.
HIGHER_IS_BETTER = {
    "false_abstention_rate": False,
}


def _arrow(name: str, ascii_only: bool = False) -> str:
    """Choose a direction marker for one metric.

    Args:
        name: Metric name used to look up its optimization direction.
        ascii_only: Whether to use console-safe ``(+)``/``(-)`` text.

    Returns:
        Up/higher-is-better or down/lower-is-better marker. Unlisted metrics
        default to higher-is-better.
    """
    up = HIGHER_IS_BETTER.get(name, True)
    if ascii_only:  # Windows consoles often cannot print arrows
        return "(+)" if up else "(-)"
    return "↑" if up else "↓"


def _metric_rows(metrics: dict) -> list[str]:
    """Render a mapping of named ratios as Markdown table rows.

    Args:
        metrics: Metric names mapped to ratio-like values with ``fmt()``.

    Returns:
        One formatted Markdown row per metric, preserving mapping order.
    """
    return [f"| {name} {_arrow(name)} | {r.fmt()} |" for name, r in metrics.items()]


def render(meta: dict, results: dict, gates: list[dict]) -> str:
    """Render a complete Markdown evaluation report.

    Args:
        meta: Run metadata including timestamp, modes, models, corpus, and
            optional warnings.
        results: Mapping of suite names to result objects with pass rate,
            metrics, and optional latency values.
        gates: Release-gate dictionaries containing gate, status, and value.

    Returns:
        Markdown report text. Detailed failures are referenced in
        ``results.json`` rather than expanded inline.
    """
    lines = [
        "# RepWise evaluation report",
        "",
        f"- **Run:** {meta['timestamp']}{' · ' + meta['label'] if meta.get('label') else ''}",
        f"- **Mode:** {meta['suite']}",
        f"- **Retriever:** `{meta['retriever']}` · **Planner:** `{meta['planner']}`",
        f"- **Generation mode:** {meta.get('llm_mode', 'not reported')}",
        f"- **Models:** embedding `{meta.get('embedding_model') or 'not used'}` · generation `{meta.get('generation_model') or 'not used'}`",
        f"- **French query expansion:** {'enabled' if meta.get('query_expansion') else 'disabled'}"
        + (
            f" · model `{meta['query_expansion_model']}`"
            if meta.get("query_expansion_model")
            else ""
        ),
        f"- **Corpus:** {meta['n_corpus_chunks']} chunks from `{meta['chunks_dir']}`",
        "",
        "Values are shown as `value (numerator/denominator)`. `N/A (0/0)` means no applicable case, "
        "not a perfect score. ↑ higher is better, ↓ lower is better.",
        "",
    ]
    for w in meta.get("warnings", []):
        lines += [f"> **Warning:** {w}", ""]

    if gates:
        lines += ["## Release gates", "", "| Gate | Status | Value |", "|---|---|---|"]
        lines += [f"| {g['gate']} | **{g['status']}** | {g['value']} |" for g in gates]
        lines.append("")

    for name, r in results.items():
        lines += [
            f"## {name.capitalize()}",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| pass rate ↑ | {r.pass_rate.fmt()} |",
            *_metric_rows(r.metrics),
        ]
        for key in ("latency_ms_median", "latency_ms_p95"):
            if r.extra.get(key) is not None:
                lines.append(f"| {key} ↓ | {r.extra[key]:.1f} |")
        lines.append("")

    lines += ["Failure details are in `results.json`.", ""]
    return "\n".join(lines)


def render_console(results: dict, gates: list[dict]) -> str:
    """Render a plain-ASCII console summary.

    Args:
        results: Suite names mapped to results with pass rates, metrics, and
            optional failure records.
        gates: Release-gate dictionaries to include after the suite summaries.

    Returns:
        Console-ready text using ``(+)`` for higher-is-better and ``(-)`` for
        lower-is-better.
    """
    out = []
    for name, r in results.items():
        out.append(f"\n== {name} ==  pass rate {r.pass_rate.fmt()}")
        for metric, ratio in r.metrics.items():
            out.append(
                f"  {metric:<30} {_arrow(metric, ascii_only=True)} {ratio.fmt()}"
            )
        if r.failures:
            counts = Counter(f.category for f in r.failures)
            out.append(
                "  failures: " + ", ".join(f"{c}={n}" for c, n in counts.most_common())
            )
    if gates:
        out.append("\n== release gates ==")
        out += [f"  [{g['status']}] {g['gate']} ({g['value']})" for g in gates]
    return "\n".join(out)


def render_comparison(
    meta: dict, retrieval: dict, generation: dict, red_team: Any
) -> str:
    """Render retrieval rankings and the controlled generation comparison.

    Args:
        meta: Comparison metadata including rankings, selected retriever,
            models, corpus, and generation case IDs.
        retrieval: Retriever names mapped to their evaluation results.
        generation: Generation-arm labels mapped to their evaluation results.

    Returns:
        Markdown comparing retrieval strategies and the RAG/no-RAG generation
        arms, including an interpretation caveat.
    """
    main_k = next(
        (
            result.extra["main_k"]
            for result in retrieval.values()
            if isinstance(result.extra.get("main_k"), int)
        ),
        5,
    )
    rewrite_latency = meta.get("query_expansion_latency_ms_median")
    rewrite_latency_text = (
        f"{rewrite_latency:.1f} ms" if rewrite_latency is not None else "not measured"
    )
    generation_rewrite_latency = meta.get(
        "generation_query_expansion_latency_ms_median"
    )
    generation_rewrite_latency_text = (
        f"{generation_rewrite_latency:.1f} ms"
        if generation_rewrite_latency is not None
        else "not measured"
    )
    lines = [
        "# RepWise benchmark comparison",
        "",
        f"- **Run:** {meta['timestamp']}{' · ' + meta['label'] if meta.get('label') else ''}",
        f"- **Corpus:** {meta['n_corpus_chunks']} chunks from `{meta['chunks_dir']}`",
        f"- **Generation cases:** {len(meta['generation_case_ids'])} identical cases per arm",
        f"- **Ollama models:** A `{meta['model_a']}` · B `{meta['model_b']}`",
        f"- **Embedding model:** `{meta['embedding_model']}`",
        f"- **French query expansion:** {'enabled' if meta.get('query_expansion') else 'disabled'}"
        + (
            f" · model `{meta['query_expansion_model']}`"
            if meta.get("query_expansion_model")
            else ""
        ),
        f"- **Median query rewrite latency:** {rewrite_latency_text}",
        f"- **Median plan-query rewrite latency:** {generation_rewrite_latency_text}",
        f"- **Selected retrieval setup:** `{meta['selected_retriever']}`",
        f"- **Best RAG setup:** `{meta['best_rag_setup']['generation_model']} + "
        f"{meta['best_rag_setup']['retriever']}`",
        "",
        "## Retrieval ranking",
        "",
        "All retrievers use the same answerable questions and time-span relevance labels. "
        f"Ranking is by chunk Recall@{main_k}, then MRR, then median retrieval latency. "
        "Rewrite latency is measured separately and shared equally across strategies; "
        "generation uses the configured default strategy unless `--retriever` overrides it.",
        "",
        f"| Rank | Retriever | Recall@{main_k} ↑ | MRR ↑ | Hit@1 ↑ | Median latency ↓ |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, name in enumerate(meta["retrieval_ranking"], start=1):
        result = retrieval[name]
        latency = result.extra.get("latency_ms_median")
        result_main_k = result.extra.get("main_k") or main_k
        recall_key = f"recall@{result_main_k}"
        lines.append(
            f"| {rank} | `{name}` | {result.metrics[recall_key].fmt()} | "
            f"{result.metrics['mrr'].fmt()} | {result.metrics['hit@1'].fmt()} | "
            f"{latency:.1f} ms |"
        )

    lines += [
        "",
        "## Generation comparison",
        "",
        f"Both RAG arms use `{meta['selected_retriever']}`. Only RAG arms are ranked; "
        "the no-RAG baseline uses model A with no retrieved passages or source IDs and "
        "is reported separately. Ranking is by case pass rate, request constraint "
        "compliance, schema validity, then median latency.",
        "",
        "| Rank | System | Pass rate ↑ | Request compliance ↑ | Schema validity ↑ | Citation ID validity ↑ | Median latency ↓ |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, name in enumerate(meta["generation_ranking"], start=1):
        result = generation[name]
        latency = result.extra.get("latency_ms_median")
        lines.append(
            f"| {rank} | {name} | {result.pass_rate.fmt()} | "
            f"{result.metrics['request_constraint_compliance'].fmt()} | "
            f"{result.metrics['output_schema_validity'].fmt()} | "
            f"{result.metrics['citation_id_validity'].fmt()} | "
            f"{latency:.1f} ms |"
        )

    baseline = generation[meta["generation_baseline"]]
    lines += [
        "",
        f"No-RAG baseline (`{meta['generation_baseline']}`): pass rate "
        f"{baseline.pass_rate.fmt()}, request compliance "
        f"{baseline.metrics['request_constraint_compliance'].fmt()}, schema validity "
        f"{baseline.metrics['output_schema_validity'].fmt()}. This is a comparator, "
        "not a candidate for the RAG red-team run.",
        "",
        "## Red-team run on best RAG setup",
        "",
        f"Cases: {red_team.n_cases} · pass rate: {red_team.pass_rate.fmt()} · "
        f"failures: {len(red_team.failures)}. The run used "
        f"`{meta['best_rag_setup']['generation_model']} + "
        f"{meta['best_rag_setup']['retriever']}`.",
    ]

    lines += [
        "",
        "## Interpretation",
        "",
        "- `N/A (0/0)` means a metric was not applicable, not that the system scored perfectly.",
        "- Citation ID validity checks provenance only; it does not prove that a source supports a recommendation.",
        "- The no-RAG arm is a controlled baseline for model A, not a second model comparison.",
        "- Scores describe this fixed dataset only. Review failures and outputs before selecting a release configuration.",
        "- The ranking is descriptive; it does not include confidence intervals or significance tests.",
        "",
        "Per-case failures, numerators, denominators, and run metadata are in `results.json`.",
        "",
    ]
    return "\n".join(lines)
