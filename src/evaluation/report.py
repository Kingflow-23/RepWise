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
        f"- **Suite / split:** {meta['suite']} / {meta['split']}",
        f"- **Retriever:** `{meta['retriever']}` · **Planner:** `{meta['planner']}`",
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
        lines += [f"## {name.capitalize()}", "",
                "| Metric | Value |", "|---|---|",
                f"| pass rate ↑ | {r.pass_rate.fmt()} |",
                *_metric_rows(r.metrics)]
        for key in ("latency_ms_median", "latency_ms_p95"):
            if r.extra.get(key) is not None:
                lines.append(f"| {key} ↓ | {r.extra[key]:.1f} |")
        lines.append("")
        if r.not_run:
            lines += [f"{len(r.not_run)} adaptation case(s) not run (no adaptation engine yet).", ""]

    lines += ["Failure details are in `results.json`.", ""]
    return "\n".join(lines)




def render_console(results: dict, gates: list[dict]) -> str:
    """Plain-ASCII summary; (+) higher is better, (-) lower is better."""
    out = []
    for name, r in results.items():
        out.append(f"\n== {name} ==  pass rate {r.pass_rate.fmt()}")
        for metric, ratio in r.metrics.items():
            out.append(f"  {metric:<30} {_arrow(metric, ascii_only=True)} {ratio.fmt()}")
        if r.failures:
            counts = Counter(f.category for f in r.failures)
            out.append("  failures: " + ", ".join(f"{c}={n}" for c, n in counts.most_common()))
        if r.not_run:
            out.append(f"  not run: {len(r.not_run)} case(s)")
    if gates:
        out.append("\n== release gates ==")
        out += [f"  [{g['status']}] {g['gate']} ({g['value']})" for g in gates]
    return "\n".join(out)
