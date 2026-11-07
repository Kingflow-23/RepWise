"""Build the configured retriever and plan pipeline in one place.

The API, the MCP server and the scripts all call these functions, so they all
search the corpus the same way (``REPWISE_RETRIEVAL``: bm25, semantic or hybrid).
"""

from __future__ import annotations

from pathlib import Path

from config.config import AppConfig

from .base import Retriever
from .bm25 import BM25Retriever
from .hybrid import HybridRetriever
from .semantic import SemanticRetriever, load_chunks


def build_retriever(config: AppConfig) -> Retriever:
    """Load the corpus and build the configured retriever.

    Args:
        config: Settings with ``chunk_file``, ``retrieval_strategy``,
            ``embedding_model`` and ``embedding_index``.

    Returns:
        A BM25, semantic or hybrid retriever.

    Raises:
        FileNotFoundError: If ``REPWISE_CHUNK_FILE`` is not set or is missing.
        ValueError: If the strategy name is unknown.
        OSError: If the semantic index cannot be read.
    """
    if not config.chunk_file:
        raise FileNotFoundError(
            "Set REPWISE_CHUNK_FILE to the prepared JSONL corpus before starting."
        )
    chunks = load_chunks(Path(config.chunk_file).parent)
    if config.retrieval_strategy == "bm25":
        return BM25Retriever(chunks)
    if config.retrieval_strategy == "semantic":
        return SemanticRetriever(chunks, config.embedding_index, config.embedding_model)
    if config.retrieval_strategy == "hybrid":
        semantic = SemanticRetriever(
            chunks, config.embedding_index, config.embedding_model
        )
        return HybridRetriever(BM25Retriever(chunks), semantic)
    raise ValueError("REPWISE_RETRIEVAL must be bm25, semantic or hybrid.")


def build_plan_pipeline(config: AppConfig):
    """Build the full RAG plan pipeline: retriever, scope gate and generator.

    Args:
        config: Application settings.

    Returns:
        A :class:`~src.generation.pipeline.PlanPipeline` using Ollama.

    Raises:
        FileNotFoundError, ValueError, OSError: See :func:`build_retriever`.
    """
    from ..generation.pipeline import PlanPipeline
    from ..llm.ollama import OllamaGenerator
    from .query_expansion import OllamaQueryExpander, QueryExpansionRetriever

    retriever = QueryExpansionRetriever(
        build_retriever(config),
        OllamaQueryExpander(config.query_expansion_model, config.ollama_url),
        rewrite=config.query_expansion,
    )
    return PlanPipeline(
        retriever, OllamaGenerator(config.ollama_model, config.ollama_url), config.top_k
    )
