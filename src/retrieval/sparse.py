"""A small BM25 implementation for the local JSONL corpus."""

from __future__ import annotations

import math
import logging
import re
import unicodedata
from collections import Counter

from ..schemas.models import RetrievedChunk

logger = logging.getLogger(__name__)


def terms(text: str) -> list[str]:
    """Tokenize text into case- and accent-folded word terms.

    Args:
        text: French or English text to tokenize.

    Returns:
        Unicode word tokens with accents removed and original punctuation
        omitted.
    """
    folded = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character for character in folded if not unicodedata.combining(character)
    )
    return re.findall(r"[\w]+", without_accents, flags=re.UNICODE)


class BM25Retriever:
    """Rank text with BM25 (k1=1.5, b=0.75), a lexical baseline.
    k1 controls the term frequency saturation, and b controls the length normalization.
    """

    def __init__(self, chunks: list[RetrievedChunk], k1: float = 1.5, b: float = 0.75):
        """Index chunk text and metadata for BM25 search.

        The searchable text combines title, body, topics, and body-part labels.
        IDF is computed once from document frequency; ``k1`` and ``b`` retain
        the usual term-frequency saturation and length-normalization roles.

        Args:
            chunks: Corpus chunks to rank; their order breaks score ties.
            k1: BM25 term-frequency saturation parameter.
            b: BM25 document-length normalization parameter, conventionally
                between zero and one.
        """
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.documents = [
            Counter(
                terms(
                    f"{c.title} {c.text} {' '.join(c.topics)} "
                    f"{' '.join(c.body_parts)}"
                )
            )
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
        """Return positive-scoring chunks ranked by BM25.

        Args:
            query: Query tokenized with :func:`terms`.
            top_k: Maximum number of results to return.

        Returns:
            At most ``top_k`` copies of matching chunks, ordered by descending
            BM25 score and then original corpus order, with rank and score
            attached. Chunks with no query-term match are omitted.
        """
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
        results = [
            self.chunks[position].model_copy(update={"rank": rank, "score": score})
            for rank, (score, position) in enumerate(scores[:top_k], start=1)
        ]
        logger.debug("BM25 search completed candidates=%s returned=%s", len(scores), len(results))
        return results
