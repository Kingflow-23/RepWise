"""Retrieval metrics: Recall@k, Precision@k, Hit@k, and MRR.

Two levels are supported:

* ID level (``*_ids`` helpers): relevant items and the ranking are plain IDs.
  This is the textbook definition (course slides 30-32).
* Span level (``judge_ranking``): source evidence is labelled as a time span
  inside a video. This is useful for source coverage and survives re-chunking.
* Chunk level (``judge_chunk_ranking``): each chunk overlapping labelled source
  evidence is a relevant retrieval item, so recall is measured over chunks.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------- ID level


def recall_at_k_ids(relevant: set, ranking: list, k: int) -> float | None:
    """Compute ID-level recall among the first ``k`` ranked items.

    Args:
        relevant: Set of relevant item IDs.
        ranking: Ranked item IDs, best first.
        k: Number of leading positions to evaluate.

    Returns:
        Fraction of relevant IDs present in the top ``k``; ``None`` when there
        are no relevant IDs.
    """
    if not relevant:
        return None
    return len(relevant & set(ranking[:k])) / len(relevant)


def precision_at_k_ids(relevant: set, ranking: list, k: int) -> float:
    """Compute precision at ``k`` using ``k`` as the denominator.

    Args:
        relevant: Set of relevant item IDs.
        ranking: Ranked item IDs, best first.
        k: Number of leading positions used as the denominator.

    Returns:
        Relevant IDs among the available top-ranked items divided by ``k``.
        Missing results in a short ranking count as non-relevant positions.
    """
    top = ranking[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / k


def hit_at_k_ids(relevant: set, ranking: list, k: int) -> float:
    """Return whether any relevant ID appears in the first ``k`` positions.

    Args:
        relevant: Set of relevant item IDs.
        ranking: Ranked item IDs, best first.
        k: Number of leading positions to inspect.

    Returns:
        ``1.0`` if at least one top-``k`` item is relevant, otherwise ``0.0``.
    """
    return float(any(d in relevant for d in ranking[:k]))


def reciprocal_rank_ids(relevant: set, ranking: list) -> float:
    """Compute the reciprocal rank of the first relevant ID.

    Args:
        relevant: Set of relevant item IDs.
        ranking: Ranked item IDs, best first.

    Returns:
        ``1 / rank`` for the first relevant item, or zero when none is ranked.
    """
    for rank, d in enumerate(ranking, start=1):
        if d in relevant:
            return 1 / rank
    return 0.0


# -------------------------------------------------------------- span level


@dataclass(frozen=True)
class Span:
    """A labelled relevant interval in a source video.

    Attributes:
        video_id: Stable source video identifier.
        start: Inclusive start time in seconds.
        end: End time in seconds.
    """

    video_id: str
    start: float
    end: float


@dataclass
class RankingJudgement:
    """Relevance of one ranked list against labelled retrieval targets.

    ``matches`` maps each retrieved rank to indexes of relevant targets.
    """

    n_relevant: int
    # For each rank: indexes of labelled targets matched by the retrieved item.
    matches: list[list[int]] = field(default_factory=list)

    def recall_at_k(self, k: int) -> float | None:
        """Compute target recall over the top ``k`` ranked items.

        Args:
            k: Number of leading ranked chunks to evaluate.

        Returns:
            Fraction of relevant targets retrieved at least once, or ``None``
            when no targets were labelled.
        """
        if self.n_relevant == 0:
            return None
        covered = {i for m in self.matches[:k] for i in m}
        return len(covered) / self.n_relevant

    def precision_at_k(self, k: int) -> float:
        """Compute the share of top-``k`` ranks matching a relevant target.

        Args:
            k: Number of top ranks used as the denominator.

        Returns:
            Relevant ranks among available matches divided by ``k``.
        """
        return sum(1 for m in self.matches[:k] if m) / k

    def hit_at_k(self, k: int) -> float:
        """Return whether any of the first ``k`` ranks matches a target.

        Args:
            k: Number of leading ranks to inspect.

        Returns:
            ``1.0`` if any inspected rank is relevant, otherwise ``0.0``.
        """
        return float(any(self.matches[:k]))

    def reciprocal_rank(self) -> float:
        """Return the reciprocal rank of the first relevant ranked item.

        Returns:
            ``1 / rank`` for the first relevant chunk, or zero if none is
            relevant.
        """
        for rank, m in enumerate(self.matches, start=1):
            if m:
                return 1 / rank
        return 0.0

    def first_relevant_rank(self) -> int | None:
        """Find the one-based rank of the first relevant item.

        Returns:
            One-based first relevant rank, or ``None`` if there is no match.
        """
        for rank, m in enumerate(self.matches, start=1):
            if m:
                return rank
        return None


def overlap_seconds(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    """Measure the intersection length of two time intervals.

    Args:
        a_start: Start time of the first interval.
        a_end: End time of the first interval.
        b_start: Start time of the second interval.
        b_end: End time of the second interval.

    Returns:
        Non-negative overlap duration in seconds; disjoint intervals return
        zero.
    """
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def judge_ranking(
    spans: list[Span], ranking: list[dict], min_overlap_s: float = 1.0
) -> RankingJudgement:
    """Mark ranked chunks that sufficiently overlap each labelled span.

    Args:
        spans: Ground-truth relevant intervals.
        ranking: Retrieved chunk dictionaries in best-first order, each with
            ``video_id``, ``start``, and ``end``.
        min_overlap_s: Minimum intersection duration, inclusive, for a chunk to
            match a span. The default filters incidental boundary contact.

    Returns:
        A judgement whose ``matches`` entry for each rank contains indexes of
        labelled spans overlapped by at least this many seconds.
    """
    judgement = RankingJudgement(n_relevant=len(spans))
    for chunk in ranking:
        hits = [
            i
            for i, s in enumerate(spans)
            if chunk.get("video_id") == s.video_id
            and overlap_seconds(chunk["start"], chunk["end"], s.start, s.end)
            >= min_overlap_s
        ]
        judgement.matches.append(hits)
    return judgement


def relevant_chunk_ids(
    spans: list[Span], chunks: list[dict], min_overlap_s: float = 1.0
) -> set[str]:
    """Map source-evidence spans to the corpus chunks that contain them.

    Args:
        spans: Human-labelled evidence intervals.
        chunks: Full corpus records; each must contain ``chunk_id``, ``video_id``,
            ``start``, and ``end``.
        min_overlap_s: Minimum intersection duration, inclusive, for relevance.

    Returns:
        IDs of all chunks that overlap at least one source-evidence span.

    Raises:
        KeyError: If a corpus chunk lacks a required field.
    """
    return {
        chunk["chunk_id"]
        for chunk in chunks
        if any(
            chunk["video_id"] == span.video_id
            and overlap_seconds(chunk["start"], chunk["end"], span.start, span.end)
            >= min_overlap_s
            for span in spans
        )
    }


def judge_chunk_ranking(
    relevant_ids: set[str], ranking: list[dict]
) -> RankingJudgement:
    """Judge a ranked list of retrieved chunks against relevant chunk IDs.

    Args:
        relevant_ids: IDs of relevant chunks in the full corpus.
        ranking: Retrieved chunk dictionaries in best-first order.

    Returns:
        A judgement whose recall denominator is the number of relevant chunks.
    """
    target_indexes = {chunk_id: index for index, chunk_id in enumerate(relevant_ids)}
    judgement = RankingJudgement(n_relevant=len(relevant_ids))
    for chunk in ranking:
        chunk_id = chunk["chunk_id"]
        index = target_indexes.get(chunk_id)
        judgement.matches.append([] if index is None else [index])
    return judgement
