import pytest

from src.evaluation.metrics.common import Ratio, mean_ratio
from src.evaluation.metrics.retrieval import (
    Span,
    hit_at_k_ids,
    judge_ranking,
    precision_at_k_ids,
    recall_at_k_ids,
    reciprocal_rank_ids,
)

# Worked example from the course (GenAI, 7 Oct, slide 32).
RELEVANT = {"d2", "d5", "d9"}
RANKING = ["d7", "d2", "d4", "d5", "d1", "d9", "d3"]


def test_course_example_recall_at_5():
    assert recall_at_k_ids(RELEVANT, RANKING, 5) == pytest.approx(2 / 3)


def test_course_example_precision_at_5():
    assert precision_at_k_ids(RELEVANT, RANKING, 5) == pytest.approx(0.40)


def test_course_example_hit_at_1():
    assert hit_at_k_ids(RELEVANT, RANKING, 1) == 0


def test_course_example_mrr():
    assert reciprocal_rank_ids(RELEVANT, RANKING) == pytest.approx(0.5)


def test_slide_30_example():
    ranking = ["art13", "art12", "art4", "faq", "art7"]
    assert hit_at_k_ids({"art12"}, ranking, 1) == 0
    assert recall_at_k_ids({"art12"}, ranking, 3) == 1
    assert reciprocal_rank_ids({"art12"}, ranking) == 0.5


def test_mrr_is_zero_when_nothing_relevant_is_returned():
    assert reciprocal_rank_ids({"x"}, ["a", "b"]) == 0.0


def test_recall_is_not_applicable_without_relevant_items():
    assert recall_at_k_ids(set(), RANKING, 5) is None


# ------------------------------------------------------------- span level


def chunk(video, start, end):
    return {"video_id": video, "start": start, "end": end}


def test_span_overlap_marks_chunk_relevant():
    spans = [Span("v1", 100, 130)]
    j = judge_ranking(spans, [chunk("v2", 100, 130), chunk("v1", 90, 180)])
    assert j.recall_at_k(1) == 0
    assert j.recall_at_k(2) == 1
    assert j.reciprocal_rank() == 0.5
    assert j.first_relevant_rank() == 2


def test_same_span_in_two_overlapping_chunks_counts_once():
    spans = [Span("v1", 100, 130), Span("v1", 400, 420)]
    ranking = [
        chunk("v1", 0, 120),
        chunk("v1", 110, 200),
    ]  # both overlap the first span
    j = judge_ranking(spans, ranking)
    assert j.recall_at_k(2) == 0.5
    assert j.precision_at_k(2) == 1.0


def test_touching_boundaries_is_not_an_overlap():
    j = judge_ranking([Span("v1", 100, 130)], [chunk("v1", 130, 200)])
    assert j.hit_at_k(1) == 0


def test_ratio_reports_na_for_zero_denominator():
    assert Ratio(0, 0).value is None
    assert Ratio(0, 0).fmt() == "N/A (0/0)"
    assert mean_ratio([1.0, 0.5]).value == 0.75
