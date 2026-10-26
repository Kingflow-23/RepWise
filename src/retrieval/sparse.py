"""A small BM25 implementation for the local JSONL corpus."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from ..schemas.models import RetrievedChunk


def terms(text: str) -> list[str]:
    """Tokenize French and English text without introducing a large dependency."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character for character in folded if not unicodedata.combining(character)
    )
    return re.findall(r"[\w]+", without_accents, flags=re.UNICODE)


class BM25Retriever:
    """Rank text with BM25 (k1=1.5, b=0.75), a lexical baseline."""

    def __init__(self, chunks: list[RetrievedChunk], k1: float = 1.5, b: float = 0.75):
        # Commercial passages should never be candidate evidence.
        self.chunks = [
            chunk for chunk in chunks if not {"sponsor", "cta"} & set(chunk.flags)
        ]
        self.k1 = k1
        self.b = b
        self.documents = [
            Counter(terms(f"{c.title} {c.text} {' '.join(c.topics)}"))
            for c in self.chunks
        ]
        self.lengths = [sum(document.values()) for document in self.documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        document_frequency = Counter(
            term for document in self.documents for term in document
        )
        count = len(self.documents)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top chunks and attach the BM25 score and rank."""
        scores = []
        for position, document in enumerate(self.documents):
            score = 0.0
            length_factor = self.k1 * (
                1
                - self.b
                + self.b * self.lengths[position] / max(self.average_length, 1)
            )
            for term in terms(query):
                frequency = document.get(term, 0)
                if frequency:
                    score += (
                        self.idf.get(term, 0)
                        * frequency
                        * (self.k1 + 1)
                        / (frequency + length_factor)
                    )
            if score > 0:
                scores.append((score, position))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [
            self.chunks[position].model_copy(update={"rank": rank, "score": score})
            for rank, (score, position) in enumerate(scores[:top_k], start=1)
        ]
