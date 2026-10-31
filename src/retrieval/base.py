"""The stable interface between generation and the knowledge base."""

from __future__ import annotations

from typing import Protocol

from ..schemas.models import RetrievedChunk


class Retriever(Protocol):
    """Interface implemented by ranked corpus retrievers.

    Implementations return source chunks in descending relevance order, with
    rank and score populated on each result.
    """

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return at most ``top_k`` chunks ranked for a query.

        Args:
            query: User query to match against the corpus.
            top_k: Maximum number of ranked chunks to return.

        Returns:
            Retrieved source chunks, best match first.
        """


class MissingRetriever:
    """Keep the API available while preserving a clear corpus/index setup error."""

    def __init__(self, message: str | None = None):
        """Create a placeholder retriever that reports a setup problem.

        Args:
            message: Optional explanation raised by :meth:`search`; a default
                corpus-setup message is used when omitted or empty.
        """
        self.message = (
            message or "Retrieval is not connected yet. Configure a local corpus."
        )

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Raise the configured setup error instead of returning results.

        Args:
            query: Accepted for compatibility with the :class:`Retriever`
                protocol; it is not used.
            top_k: Accepted for protocol compatibility; it is not used.

        Raises:
            RuntimeError: Always, with the stored setup message.
        """
        raise RuntimeError(self.message)
