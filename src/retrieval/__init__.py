"""Public retrieval protocol and implementations.

The package exports the BM25 lexical baseline, local semantic retriever,
Reciprocal Rank Fusion wrapper, and placeholder retriever interface.
"""

from .base import MissingRetriever, Retriever
from .hybrid import HybridRetriever
from .query_expansion import OllamaQueryExpander, QueryExpansionRetriever
from .semantic import SemanticRetriever
from .bm25 import BM25Retriever

__all__ = [
    "BM25Retriever",
    "HybridRetriever",
    "MissingRetriever",
    "OllamaQueryExpander",
    "QueryExpansionRetriever",
    "Retriever",
    "SemanticRetriever",
]
