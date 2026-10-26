"""A transparent keyword baseline for local demonstrations."""

from __future__ import annotations

import json
from pathlib import Path

from ..schemas.models import RetrievedChunk


class JsonlRetriever:
    """Small word-overlap baseline retained for tests and one-file demos.

    The API uses the BM25, semantic, or hybrid retriever in ``src/retrieval``.
    """

    def __init__(self, chunk_file: str | Path):
        path = Path(chunk_file)
        if not path.is_file():
            raise FileNotFoundError(f"Chunk file not found: {path}")
        self.chunks = [
            RetrievedChunk.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        words = {word.casefold() for word in query.split() if len(word) > 2}

        def score(chunk: RetrievedChunk) -> int:
            text = f"{chunk.title} {chunk.text} {' '.join(chunk.topics)}".casefold()
            return sum(word in text for word in words)

        ranked = sorted(self.chunks, key=score, reverse=True)
        return [
            chunk.model_copy(update={"rank": index + 1, "score": float(score(chunk))})
            for index, chunk in enumerate(ranked[:top_k])
            if score(chunk) > 0
        ]
