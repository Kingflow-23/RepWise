"""The stable interface between generation and the knowledge base."""

from __future__ import annotations

from typing import Protocol

from ..schemas.models import RetrievedChunk


class Retriever(Protocol):
    """Return chunks relevant to one query."""

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Read from the existing knowledge base without re-ingesting data."""


class MissingRetriever:
    """Clear placeholder until the vector-index adapter is connected."""

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise RuntimeError(
            "Retrieval is not connected yet. Add the knowledge-base adapter."
        )
