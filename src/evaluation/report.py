"""Readable summaries of an evaluation run: a Markdown report and a short console view."""

from collections import Counter

# Direction of each metric: True = higher is better, False = lower is better.
HIGHER_IS_BETTER = {
    "false_abstention_rate": False,
}


def _arrow(name: str, ascii_only: bool = False) -> str:
    up = HIGHER_IS_BETTER.get(name, True)
    if ascii_only:  # Windows consoles often cannot print arrows
        return "(+)" if up else "(-)"
    return "↑" if up else "↓"


def _metric_rows(metrics: dict) -> list[str]:
    return [f"| {name} {_arrow(name)} | {r.fmt()} |" for name, r in metrics.items()]


def render(meta: dict, results: dict, gates: list[dict]) -> str:
    lines = [
        "# RepWise evaluation report",
        "",
        f"- **Run:** {meta['timestamp']}{' · ' + meta['label'] if meta.get('label') else ''}",
        f"- **Mode:** {meta['suite']}",
        f"- **Retriever:** `{meta['retriever']}` · **Planner:** `{meta['planner']}`",
        f"- **Generation mode:** {meta.get('llm_mode', 'not reported')}",
        f"- **Models:** embedding `{meta.get('embedding_model') or 'not used'}` · generation `{meta.get('generation_model') or 'not used'}`",
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
    """Plain-ASCII summary; (+) higher is better, (-) lower is better."""
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


def render_comparison(meta: dict, retrieval: dict, generation: dict) -> str:
    """Render the retrieval ranking and controlled RAG/no-RAG generation comparison."""
    lines = [
        "# RepWise benchmark comparison",
        "",
        f"- **Run:** {meta['timestamp']}{' · ' + meta['label'] if meta.get('label') else ''}",
        f"- **Corpus:** {meta['n_corpus_chunks']} chunks from `{meta['chunks_dir']}`",
        f"- **Generation cases:** {len(meta['generation_case_ids'])} identical cases per arm",
        f"- **Ollama models:** A `{meta['model_a']}` · B `{meta['model_b']}`",
        f"- **Embedding model:** `{meta['embedding_model']}`",
        f"- **Selected retrieval setup:** `{meta['selected_retriever']}`",
        "",
        "## Retrieval ranking",
        "",
        "All retrievers use the same answerable questions and time-span relevance labels. "
        "Ranking is by Recall@5, then MRR, then median latency; an explicit `--retriever` "
        "overrides selection for the generation comparison.",
        "",
        "| Rank | Retriever | Recall@5 ↑ | MRR ↑ | Hit@1 ↑ | Median latency ↓ |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, name in enumerate(meta["retrieval_ranking"], start=1):
        result = retrieval[name]
        latency = result.extra.get("latency_ms_median")
        lines.append(
            f"| {rank} | `{name}` | {result.metrics['recall@5'].fmt()} | "
            f"{result.metrics['mrr'].fmt()} | {result.metrics['hit@1'].fmt()} | "
            f"{latency:.1f} ms |"
        )

    lines += [
        "",
        "## Generation comparison",
        "",
        f"Both RAG arms use `{meta['selected_retriever']}`. The no-RAG baseline uses model A "
        "with no retrieved passages or source IDs. Ranking is by case pass rate, request "
        "constraint compliance, schema validity, then median latency.",
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
