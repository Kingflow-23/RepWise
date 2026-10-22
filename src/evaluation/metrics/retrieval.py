"""Retrieval metrics: Recall@k, Precision@k, Hit@k, MRR, nDCG@k.

Two levels are supported:

* ID level (``*_ids`` helpers): relevant items and the ranking are plain IDs.
  This is the textbook definition (course slides 30-32).
* Span level (``judge_ranking``): relevance is labelled as a time span inside a
  video (video_id, start, end, grade). A retrieved chunk is relevant when it
  overlaps a labelled span. Labels therefore survive any re-chunking of the
  corpus, and a span covered by two overlapping chunks is counted once.
"""

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------- ID level

def recall_at_k_ids(relevant: set, ranking: list, k: int) -> float | None:
    if not relevant:
        return None
    return len(relevant & set(ranking[:k])) / len(relevant)


def precision_at_k_ids(relevant: set, ranking: list, k: int) -> float:
    top = ranking[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / k


def hit_at_k_ids(relevant: set, ranking: list, k: int) -> float:
    return float(any(d in relevant for d in ranking[:k]))


def reciprocal_rank_ids(relevant: set, ranking: list) -> float:
    for rank, d in enumerate(ranking, start=1):
        if d in relevant:
            return 1 / rank
    return 0.0


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))


def ndcg_at_k_ids(grades: dict, ranking: list, k: int) -> float | None:
    """grades maps item ID -> graded relevance (0 = not relevant)."""
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    if not ideal:
        return None
    gains = [grades.get(d, 0) for d in ranking[:k]]
    return dcg(gains) / dcg(ideal)


# -------------------------------------------------------------- span level

@dataclass(frozen=True)
class Span:
    video_id: str
    start: float
    end: float
    grade: int = 1


@dataclass
class RankingJudgement:
    """Relevance of one ranked list against labelled spans."""

    n_spans: int
    # For each rank: indices of labelled spans that the chunk overlaps.
    matches: list[list[int]] = field(default_factory=list)
    grades: list[int] = field(default_factory=list)

    def recall_at_k(self, k: int) -> float | None:
        if self.n_spans == 0:
            return None
        covered = {i for m in self.matches[:k] for i in m}
        return len(covered) / self.n_spans

    def precision_at_k(self, k: int) -> float:
        return sum(1 for m in self.matches[:k] if m) / k

    def hit_at_k(self, k: int) -> float:
        return float(any(self.matches[:k]))

    def reciprocal_rank(self) -> float:
        for rank, m in enumerate(self.matches, start=1):
            if m:
                return 1 / rank
        return 0.0

    def first_relevant_rank(self) -> int | None:
        for rank, m in enumerate(self.matches, start=1):
            if m:
                return rank
        return None

    def ndcg_at_k(self, k: int) -> float | None:
        ideal = sorted((g for g in self.grades if g > 0), reverse=True)[:k]
        if not ideal:
            return None
        # Each span earns its gain once, at the first chunk that covers it.
        credited: set[int] = set()
        gains = []
        for m in self.matches[:k]:
            fresh = [i for i in m if i not in credited]
            credited.update(fresh)
            gains.append(max((self.grades[i] for i in fresh), default=0))
        return dcg(gains) / dcg(ideal)


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def judge_ranking(spans: list[Span], ranking: list[dict], min_overlap_s: float = 1.0) -> RankingJudgement:
    """ranking: retrieved chunks, best first, each with video_id, start and end."""
    judgement = RankingJudgement(n_spans=len(spans), grades=[s.grade for s in spans])
    for chunk in ranking:
        hits = [
            i for i, s in enumerate(spans)
            if s.grade > 0
            and chunk.get("video_id") == s.video_id
            and overlap_seconds(chunk["start"], chunk["end"], s.start, s.end) >= min_overlap_s
        ]
        judgement.matches.append(hits)
    return judgement
