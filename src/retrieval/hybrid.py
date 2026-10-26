"""Simple Reciprocal Rank Fusion for BM25 and semantic search."""

from __future__ import annotations

from ..schemas.models import RetrievedChunk


class HybridRetriever:
    """Combine two ranked lists without comparing unlike score scales."""

    def __init__(self, lexical, semantic, rrf_constant: int = 60):
        self.lexical = lexical
        self.semantic = semantic
        self.rrf_constant = rrf_constant

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Score each chunk as 1/(constant + rank) in each list where it appears."""
        # Ask each retriever for enough candidates to give fusion useful options.
        candidate_count = max(top_k * 4, top_k)
        lists = [
            self.lexical.search(query, candidate_count),
            self.semantic.search(query, candidate_count),
        ]
        fused: dict[str, tuple[float, RetrievedChunk]] = {}
        for ranking in lists:
            for rank, chunk in enumerate(ranking, start=1):
                stable_id = chunk.chunk_id or chunk.passage_ids[0]
                score, _ = fused.get(stable_id, (0.0, chunk))
                fused[stable_id] = (score + 1 / (self.rrf_constant + rank), chunk)
        ordered = sorted(fused.values(), key=lambda item: -item[0])[:top_k]
        return [
            chunk.model_copy(update={"rank": rank, "score": score})
            for rank, (score, chunk) in enumerate(ordered, start=1)
        ]
