"""Retrieval evaluation: does search put the labelled evidence near the top?

Questions are labelled with video time spans, not chunk IDs. A chunk is
relevant when it overlaps a span by at least one second, so labels survive
re-chunking. Every strategy answers the same questions; they are ranked by
Recall@MAIN_K, then MRR, then median latency.

A question succeeds when any relevant chunk is in the top
``RETRIEVAL_SUCCESS_K``; it is a full success when every labelled span is
covered there, partial otherwise.

Run:  python -m src.evaluation.retrieval
"""

import logging
import time

from config import evaluation_config as config
from config.config import AppConfig

from ..retrieval.base import Retriever
from ..retrieval.semantic import load_chunks
from .datasets import RetrievalCase, load_retrieval_cases
from .metrics.common import Ratio, mean_ratio
from .metrics.retrieval import judge_chunk_ranking, judge_ranking, relevant_chunk_ids
from .report import median, ms, percentile, write_run

logger = logging.getLogger(__name__)


def build_retrievers(
    names: list[str], chunks: list, *, query_expansion: bool = False
) -> dict[str, Retriever]:
    """Build the production BM25, semantic and/or hybrid retrievers.

    With ``query_expansion`` the query is rewritten in French first; the scope
    gate is disabled because evaluation questions are curated.
    """
    from ..retrieval.bm25 import BM25Retriever
    from ..retrieval.hybrid import HybridRetriever
    from ..retrieval.query_expansion import OllamaQueryExpander, QueryExpansionRetriever
    from ..retrieval.semantic import SemanticRetriever

    settings = AppConfig.from_environment()
    bm25 = semantic = None
    if {"bm25", "hybrid"} & set(names):
        bm25 = BM25Retriever(chunks)
    if {"semantic", "hybrid"} & set(names):
        semantic = SemanticRetriever(
            chunks, settings.embedding_index, settings.embedding_model
        )
    built = {"bm25": bm25, "semantic": semantic}
    if "hybrid" in names:
        built["hybrid"] = HybridRetriever(bm25, semantic)
    retrievers = {name: built[name] for name in names}
    if query_expansion:
        expander = OllamaQueryExpander(
            settings.query_expansion_model, settings.ollama_url
        )
        retrievers = {
            name: QueryExpansionRetriever(r, expander, enforce_scope_gate=False)
            for name, r in retrievers.items()
        }
    return retrievers


def evaluate_retriever(
    cases: list[RetrievalCase], retriever: Retriever, chunks: list[dict]
) -> dict:
    """Score one retriever on every answerable question.

    Args:
        cases: Retrieval questions with their labelled spans.
        retriever: Object with ``search(query, top_k)``.
        chunks: Full corpus as dictionaries, used to map spans to chunk IDs.

    Returns:
        ``metrics`` (Ratios), ``latency_ms`` and per-question ``case_results``.
    """
    success_k = config.RETRIEVAL_SUCCESS_K
    depth = max(*config.K_VALUES, success_k, config.RETRIEVAL_DEPTH)
    per_query: dict[str, list[float]] = {}
    latencies, case_results = [], []
    n_evaluated = n_found = 0

    for case in cases:
        if not case.answerable:
            continue
        relevant_ids = relevant_chunk_ids(case.spans, chunks)
        if not relevant_ids:
            case_results.append(
                {
                    "case_id": case.id,
                    "outcome": "needs_review",
                    "comment": "No corpus chunk matches the labelled evidence; "
                    "excluded from the denominator.",
                }
            )
            continue

        started = time.perf_counter()
        ranking = [c.model_dump() for c in retriever.search(case.query, depth)]
        latencies.append((time.perf_counter() - started) * 1000)
        n_evaluated += 1

        judged = judge_chunk_ranking(relevant_ids, ranking)
        for k in config.K_VALUES:
            per_query.setdefault(f"recall@{k}", []).append(judged.recall_at_k(k))
            per_query.setdefault(f"precision@{k}", []).append(judged.precision_at_k(k))
        per_query.setdefault("hit@1", []).append(judged.hit_at_k(1))
        per_query.setdefault("mrr", []).append(judged.reciprocal_rank())

        spans = judge_ranking(case.spans, ranking).matches[:success_k]
        covered = {index for matches in spans for index in matches}
        found = bool(judged.hit_at_k(success_k))
        n_found += found
        first_rank = judged.first_relevant_rank()
        if not found:
            outcome = "no_evidence"
            comment = f"No labelled evidence in the top {success_k}" + (
                f"; first match at rank {first_rank}." if first_rank else "."
            )
        elif len(covered) == len(case.spans):
            outcome = "success"
            comment = f"All {len(case.spans)} labelled span(s) in the top {success_k}."
        else:
            outcome = "partial_success"
            comment = (
                f"Evidence in the top {success_k}, but only "
                f"{len(covered)}/{len(case.spans)} labelled spans."
            )
        case_results.append(
            {
                "case_id": case.id,
                "query": case.query,
                "outcome": outcome,
                "top_k_hit": found,
                "coverage": f"{len(covered)}/{len(case.spans)} spans",
                "chunk_coverage": (
                    f"{len(set().union(*judged.matches[:success_k]))}/{len(relevant_ids)}"
                ),
                "relevant_ranks": [
                    rank for rank, m in enumerate(judged.matches, start=1) if m
                ],
                "comment": comment,
                "top_chunks": [chunk["chunk_id"] for chunk in ranking[:3]],
            }
        )

    metrics = {name: mean_ratio(values) for name, values in per_query.items()}
    metrics[f"top{success_k}_evidence"] = Ratio(n_found, n_evaluated)
    return {
        "metrics": metrics,
        "latency_ms": {
            "median": median(latencies),
            "p95": percentile(latencies, 0.95),
        },
        "case_results": case_results,
    }


def run_retrieval(
    strategies: tuple[str, ...] = config.RETRIEVERS,
    *,
    query_expansion: bool = False,
    label: str = "",
):
    """Compare retrieval strategies and write ``report.md`` + ``results.json``.

    Returns:
        The run folder.
    """
    chunks = load_chunks(config.CHUNKS_DIR)
    chunk_dicts = [chunk.model_dump() for chunk in chunks]
    cases = load_retrieval_cases()
    retrievers = build_retrievers(
        list(strategies), chunks, query_expansion=query_expansion
    )
    results = {}
    for name, retriever in retrievers.items():
        logger.info("Retrieval evaluation started retriever=%s", name)
        results[name] = evaluate_retriever(cases, retriever, chunk_dicts)

    def rank_key(name: str) -> tuple[float, float, float]:
        metrics, latency = results[name]["metrics"], results[name]["latency_ms"]
        return (
            metrics[f"recall@{config.MAIN_K}"].value or 0.0,
            metrics["mrr"].value or 0.0,
            -(latency["median"] if latency["median"] is not None else float("inf")),
        )

    payload = {
        "meta": {
            "suite": "retrieval",
            "label": label,
            "n_cases": len(cases),
            "n_answerable": sum(case.answerable for case in cases),
            "n_corpus_chunks": len(chunks),
            "chunks_dir": config.CHUNKS_DIR,
            "embedding_model": AppConfig.from_environment().embedding_model,
            "query_expansion": query_expansion,
            "cutoffs": list(config.K_VALUES),
            "main_k": config.MAIN_K,
            "success_k": config.RETRIEVAL_SUCCESS_K,
            "retrieval_depth": config.RETRIEVAL_DEPTH,
        },
        "ranking": sorted(results, key=rank_key, reverse=True),
        "retrievers": results,
    }
    return write_run("retrieval", payload, render_markdown(payload), label)


def render_markdown(payload: dict) -> str:
    """Readable summary: ranking table, metrics at every k, missed questions."""
    meta, results = payload["meta"], payload["retrievers"]
    main_k, success_k = meta["main_k"], meta["success_k"]
    lines = [
        "# Retrieval evaluation",
        "",
        f"- **Questions:** {meta['n_answerable']} answerable / {meta['n_cases']}",
        f"- **Corpus:** {meta['n_corpus_chunks']} chunks",
        f"- **Embedding model:** `{meta['embedding_model']}`",
        f"- **Query expansion:** {meta['query_expansion']}",
        "",
        f"Ranked by Recall@{main_k}, then MRR, then median latency. "
        f"Top-{success_k} evidence is Yes when any labelled evidence is in the "
        f"top {success_k}.",
        "",
        f"| Rank | Retriever | Recall@{main_k} | Top-{success_k} evidence | MRR "
        "| Hit@1 | Median latency | p95 latency |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, name in enumerate(payload["ranking"], start=1):
        m, latency = results[name]["metrics"], results[name]["latency_ms"]
        lines.append(
            f"| {rank} | `{name}` | {m[f'recall@{main_k}'].fmt()} "
            f"| {m[f'top{success_k}_evidence'].fmt()} | {m['mrr'].fmt()} "
            f"| {m['hit@1'].fmt()} | {ms(latency['median'])} | {ms(latency['p95'])} |"
        )

    lines += ["", "## Recall and precision at each cutoff", ""]
    cutoffs = meta["cutoffs"]
    lines.append(
        "| Retriever | "
        + " | ".join(f"R@{k}" for k in cutoffs)
        + " | "
        + " | ".join(f"P@{k}" for k in cutoffs)
        + " |"
    )
    lines.append("|---|" + "---:|" * (2 * len(cutoffs)))
    for name in payload["ranking"]:
        m = results[name]["metrics"]
        values = [m[f"recall@{k}"] for k in cutoffs] + [
            m[f"precision@{k}"] for k in cutoffs
        ]
        lines.append(
            f"| `{name}` | "
            + " | ".join(f"{v.value:.3f}" if v.value is not None else "N/A" for v in values)
            + " |"
        )

    for name in payload["ranking"]:
        missed = [
            case
            for case in results[name]["case_results"]
            if case["outcome"] != "success"
        ]
        lines += ["", f"## `{name}`: questions not fully covered ({len(missed)})", ""]
        if not missed:
            lines.append("None.")
            continue
        lines += ["| Case | Outcome | Coverage | Comment |", "|---|---|---|---|"]
        for case in missed:
            lines.append(
                f"| {case['case_id']} | {case['outcome']} "
                f"| {case.get('coverage', '—')} | {case['comment']} |"
            )
    lines += ["", "Per-question rankings and coverage are in `results.json`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    from ..observability.observability import configure_logging

    configure_logging("evaluation")
    print(f"Retrieval report: {run_retrieval()}")
