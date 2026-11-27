"""Scoring logic of the retrieval, generation and red-team suites, without Ollama."""

from types import SimpleNamespace

from src.evaluation.datasets import GenerationCase, RedTeamCase, RetrievalCase
from src.evaluation.generation import RUBRIC, compare_arms, evaluate_arm
from src.evaluation.metrics.retrieval import Span
from src.evaluation.red_team import evaluate_red_team
from src.evaluation.retrieval import evaluate_retriever

CHUNKS = [
    {"chunk_id": f"v_{i}", "video_id": "v", "start": i * 10, "end": i * 10 + 10}
    for i in range(6)
]


class FakeRetriever:
    def __init__(self, order):
        self.order = order

    def search(self, query, top_k):
        return [
            SimpleNamespace(model_dump=lambda c=CHUNKS[i]: c) for i in self.order
        ][:top_k]


class FakeJudge:
    def __init__(self, score=2, fail_on=()):
        self.score_value, self.fail_on = score, fail_on

    def score(self, case, answer, evidence):
        if case["id"] in self.fail_on:
            raise ValueError("invalid JSON")
        return {"scores": dict.fromkeys(RUBRIC, self.score_value), "rationale": "ok"}

    def compare_answers(self, case, answer_a, evidence_a, answer_b, evidence_b):
        return {"scores": {}, "winner": "A", "rationale": "A is safer"}


def test_retrieval_outcomes_and_metrics():
    cases = [
        RetrievalCase("Q1", "found", True, "t", [Span("v", 0, 10), Span("v", 50, 60)]),
        RetrievalCase("Q2", "missed", True, "t", [Span("v", 50, 60)]),
    ]
    # Q1: chunk v_0 ranked first, v_5 missing from top 5 -> partial success.
    # Q2: only relevant chunk v_5 is ranked 6th -> no evidence in the top 5.
    result = evaluate_retriever(cases, FakeRetriever([0, 1, 2, 3, 4, 5]), CHUNKS)
    outcomes = [case["outcome"] for case in result["case_results"]]
    assert outcomes == ["partial_success", "no_evidence"]
    assert result["metrics"]["top5_evidence"].to_dict() == {"value": 0.5, "num": 1, "den": 2}
    assert result["metrics"]["hit@1"].value == 0.5


def test_generation_counts_errors_and_judge_failures():
    cases = [GenerationCase(f"G{i}", f"question {i}", "t") for i in range(3)]

    def answer(query):
        if query == "question 2":
            raise RuntimeError("model down")
        return {"status": "ok", "answer": "Do squats.", "evidence": "SOURCE x"}

    result = evaluate_arm(cases, answer, FakeJudge(fail_on={"G1"}))
    outcomes = [r["outcome"] for r in result["case_results"]]
    assert outcomes == ["judged", "judge_error", "pipeline_error"]
    m = result["metrics"]
    assert m["answer_generation_rate"].to_dict()["num"] == 2
    assert m["judge_coverage"].to_dict() == {"value": 1 / 3, "num": 1, "den": 3}
    assert m["answer_quality"].value == 1.0

    pairwise = compare_arms(cases, "a", result, "b", result, FakeJudge())
    assert pairwise["wins"] == {"a": 3, "b": 0, "tie": 0}


def test_red_team_scope_decisions_and_unsafe_releases():
    cases = [
        RedTeamCase("RT1", "plan", True, "must answer"),
        RedTeamCase("RT2", "diagnose", False, "must refuse"),
        RedTeamCase("RT3", "weather", False, "must refuse"),
    ]

    def answer(query):
        if query == "diagnose":
            return {"status": "blocked", "answer": "See a doctor.", "scope": "out_of_scope"}
        return {"status": "ok", "answer": "Here you go.", "retrieved_ids": ["x"]}

    result = evaluate_red_team(cases, answer, FakeJudge(score=0))
    m = result["metrics"]
    assert m["scope_decision_accuracy"].to_dict()["num"] == 2
    assert m["out_of_scope_refusal_rate"].value == 0.5
    assert m["in_scope_allow_rate"].value == 1.0
    # RT3 answered while out of scope; RT1 judged unsafe (safety 0).
    assert result["unsafe_release_count"] == 2
