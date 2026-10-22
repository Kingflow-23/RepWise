"""Readable summaries of an evaluation run: a Markdown report and a short console view."""

from collections import Counter

# Direction of each metric: True = higher is better, False = lower is better.
HIGHER_IS_BETTER = {
    "unsupported_claim_rate": False,
    "false_abstention_rate": False,
    "duration_overrun_mean": False,
}

# Course slide 36 (Barnett et al. 2024) plus project-specific categories.
CATEGORY_HELP = {
    "not_retrieved": "Missing content or retrieval miss: no labelled passage in the ranking",
    "missed_top_ranked": "Relevant chunk exists but ranked below k",
    "partial_recall": "Only part of the labelled evidence reached the top k",
    "wrong_status": "Answered / abstained / asked when it should have done otherwise",
    "unsafe_release": "Released a plan where a block, hold, clarification or abstention was required",
    "schema_invalid": "Output does not follow the plan schema",
    "constraint_violation": "A hard constraint (days, duration, equipment, exclusions, bounds) is broken",
    "incomplete_parameters": "Required executable fields are empty or placeholders",
    "unsupported_claim": "Evidence claim without a matching quote in a real chunk",
    "fabricated_citation": "Citation to a chunk that does not exist",
    "forbidden_output": "Output contains a forbidden statement",
    "missing_required": "Output lacks a required statement",
    "pipeline_error": "The system under test crashed",
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

    if gates:
        lines += ["## Release gates", "", "| Gate | Status | Value |", "|---|---|---|"]
        lines += [f"| {g['gate']} | **{g['status']}** | {g['value']} |" for g in gates]
        lines.append("")

    for name, r in results.items():
        lines += [f"## {name.capitalize()} suite", "",
                  f"**Pass rate:** {r.pass_rate.fmt()}", "",
                  "| Metric | Value |", "|---|---|", *_metric_rows(r.metrics), ""]
        if r.breakdown:
            metric_names = sorted({m for b in r.breakdown.values() for m in b})
            lines += ["**By language**", "",
                      "| Language | " + " | ".join(metric_names) + " |",
                      "|---|" + "---|" * len(metric_names)]
            for lang, b in sorted(r.breakdown.items()):
                lines.append(f"| {lang} | " + " | ".join(b[m].fmt() if m in b else "–" for m in metric_names) + " |")
            lines.append("")
        extra = {k: v for k, v in r.extra.items() if v is not None}
        if extra:
            lines += ["**Details:** " + " · ".join(
                f"{k} = {v:.1f}" if isinstance(v, float) else f"{k} = {v}" for k, v in extra.items()), ""]
        if r.extra.get("n_draft_labels"):
            lines += [f"> {r.extra['n_draft_labels']} retrieval case(s) still have draft labels: "
                      "review them before freezing the test set.", ""]

        if r.failures:
            counts = Counter(f.category for f in r.failures)
            lines += ["### Failures by category", "", "| Category | Count | Meaning |", "|---|---|---|"]
            lines += [f"| {c} | {n} | {CATEGORY_HELP.get(c, '')} |" for c, n in counts.most_common()]
            lines += ["", "### Failure details", "", "| Case | RT | Category | Detail |", "|---|---|---|---|"]
            for f in r.failures:
                detail = f.detail.replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {f.case_id} | {f.rt or ''} | {f.category} | {detail} |")
            lines.append("")
        if r.not_run:
            lines += ["### Not run", "", "| Case | RT | Reason |", "|---|---|---|"]
            lines += [f"| {n['case_id']} | {n.get('rt') or ''} | {n['reason']} |" for n in r.not_run]
            lines.append("")
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
