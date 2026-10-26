"""The stable interface between generation and the knowledge base."""

from __future__ import annotations

from typing import Protocol

from ..schemas.models import RetrievedChunk


class Retriever(Protocol):
    """Return chunks relevant to one query."""

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Read from the existing knowledge base without re-ingesting data."""


class MissingRetriever:
    """Keep the API available while preserving a clear corpus/index setup error."""

    def __init__(self, message: str | None = None):
        self.message = (
            message or "Retrieval is not connected yet. Configure a local corpus."
        )

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise RuntimeError(self.message)
