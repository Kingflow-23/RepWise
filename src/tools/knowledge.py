"""Evidence search over the curated fitness corpus (Fitness Knowledge Agent).

``search_documents`` uses the same retriever as plan generation (BM25, semantic
or hybrid, as configured by ``REPWISE_RETRIEVAL``), so evidence quality is the
same everywhere. It returns passages and their sources, nothing generated:
interpretation is the agent's job, and the agent must keep the two apart.

It deliberately searches *inside* the plan pipeline's scope gate. That gate
only accepts plan requests, so it would refuse knowledge questions such as
"how long should I rest between sets?". Health questions are refused here
instead, by the health check, because RepWise does not answer them.
"""

from __future__ import annotations

import logging
from typing import Literal

from src.retrieval.query_expansion import ScopeRefusalError
from src.schemas.models import Model
from src.tools.context import ToolContext
from src.tools.results import ToolFailure, failure

logger = logging.getLogger(__name__)

EXCERPT_CHARACTERS = 600


class Document(Model):
    """One passage found in the fitness corpus, with its source."""

    source_id: str
    title: str
    excerpt: str
    url: str
    speaker: str | None = None
    score: float | None = None


class SearchResult(Model):
    """Output of ``search_documents``, best match first."""

    ok: Literal[True] = True
    query: str
    evidence_found: bool
    documents: list[Document]
    note: str


def search_documents(
    context: ToolContext, query: str, top_k: int = 3
) -> SearchResult | ToolFailure:
    """Search the fitness knowledge base. Nothing is generated or saved.

    Args:
        context: Trusted tool context.
        query: What to look up, in English or French.
        top_k: How many passages to return.

    Returns:
        Passages with source IDs, titles and links, best first. When nothing
        matches, ``evidence_found`` is ``False`` and ``documents`` is empty, so
        the agent can say the corpus does not cover the question. Failures:
        ``health_question``, ``search_unavailable``, ``out_of_scope``.
    """
    if context.health.check(query).health_concern:
        return failure(
            "health_question",
            "RepWise does not answer pain, injury or medical questions. "
            "Suggest seeing a health professional.",
        )
    try:
        pipeline = context.pipeline()
        retriever = getattr(pipeline.retriever, "retriever", pipeline.retriever)
        chunks = retriever.search(query, top_k)
    except ScopeRefusalError as error:
        return failure("out_of_scope", error.user_message)
    except (OSError, RuntimeError, ValueError) as error:
        logger.exception("Document search failed")
        return failure("search_unavailable", str(error))
    documents = [
        Document(
            source_id=chunk.chunk_id,
            title=chunk.title,
            excerpt=chunk.text[:EXCERPT_CHARACTERS],
            url=chunk.start_url or chunk.video_url,
            speaker=chunk.speaker,
            score=chunk.score,
        )
        for chunk in chunks[:top_k]
    ]
    return SearchResult(
        query=query,
        evidence_found=bool(documents),
        documents=documents,
        note=(
            "These are retrieved transcript excerpts. Quote them as evidence and "
            "label anything else as interpretation."
            if documents
            else "The corpus has no passage for this question."
        ),
    )
