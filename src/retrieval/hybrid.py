"""Simple Reciprocal Rank Fusion for BM25 and semantic search."""

from __future__ import annotations

import logging

from ..schemas.models import RetrievedChunk

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Fuse lexical and semantic ranked lists with Reciprocal Rank Fusion.

    RRF uses each retriever's rank rather than its raw score, allowing BM25 and
    cosine-similarity results to be combined despite their different scales.
    """

    def __init__(self, lexical, semantic, rrf_constant: int = 60):
        """Store the two retrievers and rank-fusion smoothing constant.

        Args:
            lexical: Retriever supplying the lexical ranking (typically BM25).
            semantic: Retriever supplying the semantic ranking.
            rrf_constant: Offset added to each one-based rank in the RRF
                contribution.
        """
        self.lexical = lexical
        self.semantic = semantic
        self.rrf_constant = rrf_constant

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Fuse both rankings and return the highest-scoring unique chunks.

        Each source contributes ``1 / (rrf_constant + rank)`` for a chunk it
        returns. To provide candidates for fusion, each retriever is asked for
        four times ``top_k`` results (or ``top_k`` when larger).

        Args:
            query: Query passed unchanged to each underlying retriever.
            top_k: Maximum number of fused chunks to return.

        Returns:
            Unique chunks ordered by descending RRF score, with fused one-based
            ranks and scores attached.

        Raises:
            AttributeError: If a returned chunk has neither ``chunk_id`` nor a
                passage ID usable as its identity.
            IndexError: If a chunk has no passage IDs and no chunk ID.
        """
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
        results = [
            chunk.model_copy(update={"rank": rank, "score": score})
            for rank, (score, chunk) in enumerate(ordered, start=1)
        ]
        logger.debug(
            "Hybrid search completed bm25_candidates=%s semantic_candidates=%s fused=%s returned=%s",
            len(lists[0]),
            len(lists[1]),
            len(fused),
            len(results),
        )
        return results
