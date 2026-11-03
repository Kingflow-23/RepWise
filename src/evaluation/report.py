"""Readable summaries of an evaluation run: a Markdown report and a short console view."""

import json
from collections import Counter
from typing import Any

# Direction of each metric: True = higher is better, False = lower is better.
HIGHER_IS_BETTER = {
    "false_abstention_rate": False,
    "unsafe_in_scope_releases": False,
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


def _summary_metrics(result: Any) -> dict:
    """Select the small set of headline metrics shown in suite summaries."""
    if "answer_quality" in result.metrics:
        names = (
            "answer_quality",
            "answer_generation_rate",
            "request_fit",
            "schedule_profile_safety",
        )
    else:
        names = (
            "output_schema_validity",
            "request_constraint_compliance",
            "citation_id_validity",
            "faithfulness",
        )
    return {name: result.metrics[name] for name in names if name in result.metrics}


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
        *(
            [f"- **Evaluation scope:** {meta['evaluation_scope']}"]
            if meta.get("evaluation_scope")
            else []
        ),
        f"- **Retriever:** `{meta['retriever']}` · **Planner:** `{meta['planner']}`",
        f"- **Generation mode:** {meta.get('llm_mode', 'not reported')}",
        f"- **Models:** embedding `{meta.get('embedding_model') or 'not used'}` · generation `{meta.get('generation_model') or 'not used'}`",
        *(
            [f"- **Scope gate:** `{meta['scope_gate_model']}`"]
            if meta.get("scope_gate_model")
            else []
        ),
        *(
            [f"- **Answer judge:** `{meta['judge_model']}`"]
            if meta.get("judge_model")
            else []
        ),
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
        success_k = r.extra.get("success_k")
        red_team_mode = r.extra.get("red_team_mode", False) or name.startswith(
            "red-team"
        )
        summary_metrics = {} if red_team_mode else _summary_metrics(r)
        metric_rows = _metric_rows(summary_metrics)
        if success_k is not None:
            hit_name = f"hit@{success_k}"
            metric_rows = [row for row in metric_rows if hit_name not in row]
        lines += [
            f"## {name.capitalize()}",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        if success_k is not None:
            hit = r.metrics.get(hit_name)
            hit_summary = "N/A (0/0)"
            if hit is not None and hit.value is not None:
                hit_summary = f"Yes {int(hit.num)}/{int(hit.den)}"
            lines.append(f"| evidence found in top {success_k} | **{hit_summary}** |")
        elif red_team_mode:
            lines.append(f"| pass rate ↑ | {r.pass_rate.fmt()} |")
        elif "answer_quality" not in r.metrics:
            lines.append(f"| pass rate ↑ | {r.pass_rate.fmt()} |")
        lines += metric_rows
        if success_k is not None:
            n_unanswerable = r.extra.get("n_unanswerable_skipped", 0)
            if n_unanswerable:
                lines.append(f"| unanswerable queries | {n_unanswerable} excluded |")
            case_results = r.extra.get("case_results", [])
            if case_results:
                lines += [
                    "",
                    "### Retrieval cases",
                    "",
                    "| Case | Verdict | Evidence coverage | Comment |",
                    "|---|---|---:|---|",
                ]
                for case in case_results:
                    verdict = {
                        "success": "Yes · full",
                        "partial_success": "Yes · partial",
                        "no_evidence": "No / verify",
                        "needs_review": "Needs review",
                        "not_applicable": "N/A",
                    }.get(case["outcome"], case["outcome"])
                    lines.append(
                        f"| {case['case_id']} | {verdict} | "
                        f"{case.get('coverage', '—')} | {case['comment']} |"
                    )
        elif r.extra.get("case_results"):
            lines += [
                "",
                "### Case outcomes",
                "",
                "| Case | Outcome | Status | Details |",
                "|---|---|---|---|",
            ]
            for case in r.extra["case_results"]:
                details = case.get("message") or ""
                if case.get("judge"):
                    details = case["judge"].get("rationale") or details
                diagnostics = case.get("diagnostics") or {}
                model_output = (
                    diagnostics.get("raw_model_output")
                    or diagnostics.get("candidate_plan")
                    or case.get("plan")
                )
                if model_output:
                    if not isinstance(model_output, str):
                        model_output = json.dumps(model_output, ensure_ascii=False)
                    details = f"{details} | model output: {model_output}".strip(" |")
                elif not details:
                    details = case.get("answer") or ""
                details = " ".join(str(details).split())
                if len(details) > 240:
                    details = details[:237] + "..."
                details = details.replace("|", "\\|")
                lines.append(
                    f"| {case['case_id']} | {case.get('outcome', 'needs_review')} | "
                    f"{case.get('status') or '—'} | {details} |"
                )
        if success_k is not None:
            for key in ("latency_ms_median", "latency_ms_p95"):
                if r.extra.get(key) is not None:
                    lines.append(f"| {key} ↓ | {r.extra[key]:.1f} |")
        elif (
            not red_team_mode
            and "answer_quality" in r.metrics
            and r.extra.get("generation_latency_ms_median") is not None
        ):
            lines.append(
                "| generation latency median ↓ | "
                f"{r.extra['generation_latency_ms_median']:.1f} ms |"
            )
        lines.append("")

    lines += [
        "Per-case diagnostics and full model outputs are in `results.json`.",
        "",
    ]
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
        success_k = r.extra.get("success_k")
        red_team_mode = r.extra.get("red_team_mode", False) or name.startswith(
            "red-team"
        )
        if success_k is not None:
            hit = r.metrics.get(f"hit@{success_k}")
            summary = (
                f"Yes {int(hit.num)}/{int(hit.den)}"
                if hit is not None and hit.value is not None
                else "N/A (0/0)"
            )
            out.append(f"\n== {name} ==  evidence in top {success_k}: {summary}")
        elif red_team_mode:
            out.append(f"\n== {name} ==  pass rate {r.pass_rate.fmt()}")
        elif "answer_quality" not in r.metrics:
            out.append(f"\n== {name} ==  pass rate {r.pass_rate.fmt()}")
        metrics = {} if red_team_mode else _summary_metrics(r)
        for metric, ratio in metrics.items():
            out.append(
                f"  {metric:<30} {_arrow(metric, ascii_only=True)} {ratio.fmt()}"
            )
        if r.failures and not red_team_mode:
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
    lines = [
        "# RepWise benchmark comparison",
        "",
        f"- **Run:** {meta['timestamp']}{' · ' + meta['label'] if meta.get('label') else ''}",
        f"- **Corpus:** {meta['n_corpus_chunks']} chunks from `{meta['chunks_dir']}`",
        f"- **Generation cases:** {len(meta['generation_case_ids'])} identical cases per arm",
        f"- **Ollama models:** A `{meta['model_a']}` · B `{meta['model_b']}`",
        f"- **Answer judge:** `{meta.get('judge_model', 'not reported')}`",
        f"- **Red-team scope gate:** `{meta.get('scope_gate_model', 'not reported')}`",
        f"- **Embedding model:** `{meta['embedding_model']}`",
        "- **Query rewriting / generation scope gate:** disabled for model comparison",
        f"- **Selected retrieval setup:** `{meta['selected_retriever']}`",
        f"- **Best RAG setup:** `{meta['best_rag_setup']['generation_model']} + "
        f"{meta['best_rag_setup']['retriever']}`",
        "",
        "## Retrieval ranking",
        "",
        "All retrievers use the same answerable questions and time-span relevance labels. "
        f"Ranking is by chunk Recall@{main_k}, then MRR, then median retrieval latency. "
        "The simple retrieval verdict is Yes when any labeled evidence appears in the top five. "
        "Rewrite latency is measured separately and shared equally across strategies; "
        "generation uses the configured default strategy unless `--retriever` overrides it.",
        "",
        f"| Rank | Retriever | Recall@{main_k} ↑ | Top-5 evidence | MRR ↑ | Hit@1 ↑ | Median latency ↓ |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, name in enumerate(meta["retrieval_ranking"], start=1):
        result = retrieval[name]
        latency = result.extra.get("latency_ms_median")
        result_main_k = result.extra.get("main_k") or main_k
        recall_key = f"recall@{result_main_k}"
        success_k = result.extra.get("success_k", 5)
        hit = result.metrics.get(f"hit@{success_k}")
        evidence_summary = (
            f"Yes {int(hit.num)}/{int(hit.den)}"
            if hit is not None and hit.value is not None
            else "N/A"
        )
        lines.append(
            f"| {rank} | `{name}` | {result.metrics[recall_key].fmt()} | "
            f"{evidence_summary} | "
            f"{result.metrics['mrr'].fmt()} | {result.metrics['hit@1'].fmt()} | "
            f"{latency:.1f} ms |"
        )

    lines += [
        "",
        "## Stage 1: choose the model (both with RAG)",
        "",
        "Every raw answer is scored by the configured local judge. No production "
        "plan-schema or deterministic plan validator is applied. Answer quality "
        "uses judge-scored answers only; answer-generation rate and judge coverage "
        "are reported separately so generation and judge failures are not mistaken "
        "for low-quality answers. Scores are normalized to 0-1; the displayed "
        "numerator/denominator are earned/max points.",
        "",
        "| Rank | Model + RAG | Answer quality ↑ | Answer generated ↑ | Judge coverage ↑ | Model median ↓ |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, name in enumerate(meta["generation_ranking"], start=1):
        result = generation[name]
        latency = result.extra.get("generation_latency_ms_median")
        latency_text = f"{latency:.1f} ms" if latency is not None else "N/A"
        lines.append(
            f"| {rank} | {name} | {result.metrics['answer_quality'].fmt()} | "
            f"{result.metrics['answer_generation_rate'].fmt()} | "
            f"{result.metrics['judge_coverage'].fmt()} | {latency_text} |"
        )

    def judged_metric(result, name: str) -> str:
        """Format a judged ratio, showing an explicit marker when unavailable.

        Args:
            result: Evaluation result containing named ratio metrics.
            name: Metric key to retrieve from the result.

        Returns:
            The metric's formatted numerator/denominator, or ``N/A (0/0)``
            when that metric was not calculated.
        """
        ratio = result.metrics.get(name)
        return ratio.fmt() if ratio is not None else "N/A (0/0)"

    pairwise = meta.get("pairwise_comparison", {})
    wins = pairwise.get("wins", {})
    lines += [
        "",
        "### Per-case head-to-head judge choice",
        "",
        f"A = `{pairwise.get('arm_a', 'model A')}`; "
        f"B = `{pairwise.get('arm_b', 'model B')}`.",
        "",
        f"Judge preference counts: `{pairwise.get('arm_a', 'A')}` "
        f"{wins.get(pairwise.get('arm_a'), 0)} · "
        f"`{pairwise.get('arm_b', 'B')}` "
        f"{wins.get(pairwise.get('arm_b'), 0)} · ties {wins.get('tie', 0)} · "
        f"judge errors {pairwise.get('judge_errors', 0)}.",
        "",
        "| Case | Judge choice | A score | B score | Explanation |",
        "|---|---|---:|---:|---|",
    ]
    for item in pairwise.get("case_results", []):
        scores = item.get("scores", {})
        score_a = sum(scores.get("A", {}).values())
        score_b = sum(scores.get("B", {}).values())
        choice = item.get("winner", "judge error")
        rationale = item.get("rationale") or item.get("judge_error", "")
        rationale = rationale.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item['case_id']} | {choice} | {score_a} | {score_b} | {rationale} |"
        )

    baseline = generation[meta["generation_baseline"]]
    best_rag_result = generation[meta["best_rag_setup"]["generation_arm"]]
    rag_latency = best_rag_result.extra.get("generation_latency_ms_median")
    no_rag_latency = baseline.extra.get("generation_latency_ms_median")
    rag_latency_text = f"{rag_latency:.1f} ms" if rag_latency is not None else "N/A"
    no_rag_latency_text = (
        f"{no_rag_latency:.1f} ms" if no_rag_latency is not None else "N/A"
    )
    lines += [
        "",
        "## Stage 2: RAG vs no RAG for the selected model",
        "",
        f"Selected model: `{meta['best_rag_setup']['generation_model']}`. "
        f"RAG answer quality: {best_rag_result.metrics['answer_quality'].fmt()}; "
        f"no-RAG answer quality: {baseline.metrics['answer_quality'].fmt()}. "
        f"Answer generated: RAG "
        f"{best_rag_result.metrics['answer_generation_rate'].fmt()}, no-RAG "
        f"{baseline.metrics['answer_generation_rate'].fmt()}. "
        f"Request fit: RAG {judged_metric(best_rag_result, 'request_fit')}, "
        f"no-RAG {judged_metric(baseline, 'request_fit')}. "
        f"Schedule/profile safety: RAG "
        f"{judged_metric(best_rag_result, 'schedule_profile_safety')}, no-RAG "
        f"{judged_metric(baseline, 'schedule_profile_safety')}. "
        f"Median generation latency: RAG "
        f"{rag_latency_text}; no-RAG {no_rag_latency_text}.",
        "",
        "| System | Answer quality | Answer generated | Judge coverage | Request fit | Schedule/profile safety |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {meta['best_rag_setup']['generation_model']} + RAG | "
        f"{judged_metric(best_rag_result, 'answer_quality')} | "
        f"{judged_metric(best_rag_result, 'answer_generation_rate')} | "
        f"{judged_metric(best_rag_result, 'judge_coverage')} | "
        f"{judged_metric(best_rag_result, 'request_fit')} | "
        f"{judged_metric(best_rag_result, 'schedule_profile_safety')} |",
        f"| {meta['best_rag_setup']['generation_model']} without RAG | "
        f"{judged_metric(baseline, 'answer_quality')} | "
        f"{judged_metric(baseline, 'answer_generation_rate')} | "
        f"{judged_metric(baseline, 'judge_coverage')} | "
        f"{judged_metric(baseline, 'request_fit')} | "
        f"{judged_metric(baseline, 'schedule_profile_safety')} |",
        "",
        "## Red-team plan check",
        "",
        f"Pass rate: {red_team.pass_rate.fmt()}. Any returned plan is a failure; "
        "no plan is a pass. The run used "
        f"`{meta['best_rag_setup']['generation_model']} + "
        f"{meta['best_rag_setup']['retriever']}`. Per-case outputs are in "
        "`results.json`.",
    ]

    lines += [
        "",
        "## Interpretation",
        "",
        "- `N/A (0/0)` means a metric was not applicable, not that the system scored perfectly.",
        "- Per-dimension judge scores and evidence diagnostics are retained in `results.json` but omitted from the headline report and model ranking.",
        "- The no-RAG arm reuses the winning RAG model's benchmark outputs and is scored on the same cases.",
        "- Scores describe this fixed dataset only. Review failures and outputs before selecting a release configuration.",
        "- The ranking is descriptive; it does not include confidence intervals or significance tests.",
        "",
        "Per-case failures, numerators, denominators, and run metadata are in `results.json`.",
        "",
    ]
    return "\n".join(lines)


def render_rag_vs_no_rag(meta: dict, generation: dict) -> str:
    """Render the focused, fixed-model RAG ablation report."""
    rag = generation[meta["rag_arm"]]
    no_rag = generation[meta["no_rag_arm"]]

    def metric(result: Any, name: str) -> str:
        ratio = result.metrics.get(name)
        return ratio.fmt() if ratio is not None else "N/A (0/0)"

    def latency(result: Any) -> str:
        value = result.extra.get("generation_latency_ms_median")
        return f"{value:.1f} ms" if value is not None else "N/A"

    return "\n".join(
        [
            "# RepWise RAG vs no-RAG benchmark",
            "",
            f"- **Run:** {meta['timestamp']}{' · ' + meta['label'] if meta.get('label') else ''}",
            f"- **Generation model:** `{meta['generation_model']}`",
            f"- **Answer judge:** `{meta['judge_model']}`",
            f"- **RAG retriever:** `{meta['selected_retriever']}`",
            f"- **Corpus:** {meta['n_corpus_chunks']} chunks from `{meta['chunks_dir']}`",
            f"- **Generation cases:** {len(meta['generation_case_ids'])} identical cases per arm",
            f"- **Query expansion in RAG arm:** `{meta['query_expansion']}`",
            "",
            "Both arms use the same model, unconstrained response prompt, and nominal cases. The no-RAG arm receives no retrieved evidence. Answer JSON and plan-schema formatting are not required or scored.",
            "",
            "| System | Answer quality | Evidence support | Answer generated | Judge coverage | Request fit | Schedule/profile safety | Median generation latency |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| `{meta['rag_arm']}` | {metric(rag, 'answer_quality')} | {metric(rag, 'evidence_support')} | {metric(rag, 'answer_generation_rate')} | {metric(rag, 'judge_coverage')} | {metric(rag, 'request_fit')} | {metric(rag, 'schedule_profile_safety')} | {latency(rag)} |",
            f"| `{meta['no_rag_arm']}` | {metric(no_rag, 'answer_quality')} | {metric(no_rag, 'evidence_support')} | {metric(no_rag, 'answer_generation_rate')} | {metric(no_rag, 'judge_coverage')} | {metric(no_rag, 'request_fit')} | {metric(no_rag, 'schedule_profile_safety')} | {latency(no_rag)} |",
            "",
            "## Interpretation",
            "",
            "- Answer quality averages all five general quality dimensions plus evidence support. With no retrieved evidence, the judge cannot award evidence support; raw answers and per-dimension scores are retained in `results.json`.",
            "- This is a fixed-dataset ablation, not a significance test or a complete safety evaluation.",
            "- No production plan validation or retry is applied to either arm.",
            "",
        ]
    )
