"""Create a retrieval query and a safe evidence block for the model."""

from __future__ import annotations

from ..schemas.models import PlanRequest, RetrievedChunk, Source


def make_query(request: PlanRequest) -> str:
    """Combine the current focus with profile details relevant to retrieval."""
    profile = request.profile
    parts = [
        request.focus,
        profile.goal,
        profile.experience,
        *profile.equipment,
        *request.extra_constraints,
    ]
    return " | ".join(part for part in parts if part)


def select_evidence(
    chunks: list[RetrievedChunk], max_characters: int = 8_000
) -> tuple[str, list[Source]]:
    """Filter unusable chunks and keep source references for the final plan."""
    context_parts: list[str] = []
    sources: list[Source] = []
    seen_ids: set[str] = set()
    used_characters = 0

    for chunk in chunks:
        if {"sponsor", "cta"}.intersection(chunk.flags):
            continue
        source_id = chunk.passage_ids[0]
        if source_id in seen_ids:
            continue
        excerpt = " ".join(chunk.text.split())
        evidence_block = f"SOURCE {source_id}: {excerpt}"
        if used_characters + len(evidence_block) > max_characters:
            continue
        seen_ids.add(source_id)
        used_characters += len(evidence_block)
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
            )
        )

    return "\n\n".join(context_parts), sources
