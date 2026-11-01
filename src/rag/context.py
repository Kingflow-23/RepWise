"""Create a retrieval query and a safe evidence block for the model."""

from __future__ import annotations

import re
import unicodedata

from ..body_parts import detect_body_parts
from ..schemas.models import PlanRequest, RetrievedChunk, Source

HIGHLIGHT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "au",
    "aux",
    "avec",
    "be",
    "but",
    "by",
    "ce",
    "ces",
    "cette",
    "dans",
    "de",
    "des",
    "du",
    "en",
    "et",
    "est",
    "for",
    "from",
    "in",
    "is",
    "it",
    "la",
    "le",
    "les",
    "of",
    "on",
    "or",
    "par",
    "pour",
    "que",
    "qui",
    "sur",
    "the",
    "this",
    "to",
    "un",
    "une",
    "with",
    "you",
    "your",
}


def _normalize_highlight_token(token: str) -> str:
    """Normalize one token for accent-insensitive, case-insensitive matching.

    Args:
        token: Original word token from a query or evidence excerpt.

    Returns:
        A case-folded token without combining accents, with a trailing plural
        ``s`` removed when the token is longer than four characters.
    """
    folded = unicodedata.normalize("NFKD", token.casefold())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    if len(folded) > 4 and folded.endswith("s"):
        folded = folded[:-1]
    return folded


def _highlight_terms(query: str, text: str) -> list[str]:
    """Find query terms present in text while preserving their source spelling.

    Args:
        query: Text whose meaningful words define the terms to match.
        text: Evidence excerpt in which to find those terms.

    Returns:
        Unique matching words from ``text`` in their first-occurrence order.
    """
    query_tokens = {
        normalized
        for token in re.findall(r"\w+", query, flags=re.UNICODE)
        if (normalized := _normalize_highlight_token(token))
        and normalized not in HIGHLIGHT_STOP_WORDS
    }
    return list(
        dict.fromkeys(
            match.group(0)
            for match in re.finditer(r"\w+", text, flags=re.UNICODE)
            if _normalize_highlight_token(match.group(0)) in query_tokens
        )
    )


def make_query(request: PlanRequest) -> str:
    """Build a descriptive retrieval request from current intent and profile.

    Args:
        request: Plan request whose focus, goal, experience, equipment, and
            constraints should guide retrieval.

    Returns:
        A labeled query containing focus, goal, experience, preferences,
        equipment, constraints, and refinement instructions; detected body
        parts are added as retrieval hints.
    """
    profile = request.profile
    parts = [
        f"Training focus: {request.focus}",
        f"Goal: {profile.goal}",
        f"Experience level: {profile.experience}",
    ]
    if profile.preferences:
        parts.append(f"Training preferences: {profile.preferences}")
    if profile.equipment:
        parts.append(f"Available equipment: {', '.join(profile.equipment)}")
    if request.extra_constraints:
        parts.append(f"Additional constraints: {'; '.join(request.extra_constraints)}")
    if request.refinement_instructions:
        parts.append(f"Requested plan changes: {request.refinement_instructions}")
    body_parts = detect_body_parts(" ".join(parts))
    if body_parts:
        parts.append("body parts: " + " ".join(body_parts))
    return " | ".join(part for part in parts if part)


def make_highlight_query(
    request: PlanRequest, *, retrieval_query: str | None = None
) -> str:
    """Build terms for highlighting evidence relevant to the stated request.

    Args:
        request: Plan request supplying the focus, profile, and constraints.
        retrieval_query: Exact rewritten query submitted to the retriever, when
            query expansion is enabled.

    Returns:
        A space-separated string containing original request/profile intent and
        the actual query used for retrieval.

    Notes:
        Schedule and experience metadata are excluded because they do not
        establish topical relevance. Equipment and the effective retrieval query
        are included because they may be relevant constraints or transcript terms.
    """
    parts = [
        request.focus,
        request.profile.goal,
        request.profile.preferences or "",
        *request.profile.equipment,
        *request.extra_constraints,
        request.refinement_instructions or "",
        retrieval_query or "",
    ]
    body_parts = detect_body_parts(" ".join(parts))
    if body_parts:
        parts.extend(body_parts)
    return " ".join(part for part in parts if part)


def select_evidence(
    chunks: list[RetrievedChunk], *, query: str = ""
) -> tuple[str, list[Source]]:
    """Format all retrieved chunks as model context and source records.

    Args:
        chunks: Retrieved passages to preserve in their supplied order.
        query: Optional query used to mark matching terms in each excerpt.

    Returns:
        A pair containing the joined evidence block and one :class:`Source` for
        each chunk. Each source retains the full normalized evidence text while
        its display excerpt is limited to 300 characters.

    Notes:
        A source ID prefers the first unseen passage ID, then the chunk ID, and
        finally a generated fallback ID, so repeated IDs do not merge evidence.
    """
    context_parts: list[str] = []
    sources: list[Source] = []
    seen_source_ids: set[str] = set()

    for chunk in chunks:
        source_id = next(
            (
                passage_id
                for passage_id in chunk.passage_ids
                if passage_id not in seen_source_ids
            ),
            None,
        )
        if source_id is None:
            source_id = chunk.chunk_id
        if not source_id or source_id in seen_source_ids:
            source_id = f"retrieved_chunk_{len(sources) + 1}"
        seen_source_ids.add(source_id)
        excerpt = " ".join(chunk.text.split())
        evidence_block = f"SOURCE {source_id}: {excerpt}"
        context_parts.append(evidence_block)
        sources.append(
            Source(
                source_id=source_id,
                title=chunk.title,
                url=chunk.start_url or chunk.video_url,
                excerpt=excerpt[:300],
                chunk_id=chunk.chunk_id,
                passage_ids=chunk.passage_ids,
                video_id=chunk.video_id,
                start=chunk.start,
                end=chunk.end,
                video_url=chunk.video_url,
                start_url=chunk.start_url,
                evidence_text=excerpt,
                matched_terms=_highlight_terms(query, excerpt),
            )
        )

    return "\n\n".join(context_parts), sources
