"""Tests for judge validation and raw generation evaluation behavior."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from config.config import AppConfig
from src.evaluation import adapters, runner
from src.evaluation.datasets import GenerationCase
from src.evaluation.judge import OllamaQualityJudge, QUALITY_DIMENSIONS
from src.evaluation.metrics.common import Ratio
from src.llm.ollama import OllamaGenerator


def _judge_payload(*, evidence_support=2):
    return {
        "scores": {
            **{name: 2 for name in QUALITY_DIMENSIONS},
            "evidence_support": evidence_support,
        },
        "rationale": "The answer follows the request and profile.",
    }


def test_ollama_quality_judge_validates_rubric_scores(monkeypatch):
    """Keep the judge response in a small, explicit, parseable shape."""
    captured = {}
    fake_ollama = ModuleType("ollama")
    fake_ollama.Client = lambda **kwargs: SimpleNamespace(
        chat=lambda **kwargs: (
            captured.update(kwargs)
            or {"message": {"content": json.dumps(_judge_payload())}}
        )
    )
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    result = OllamaQualityJudge("qwen3.5:4b").score(
        {"request": "Build strength"},
        "A raw answer, not a validated plan.",
        "Use controlled repetitions.",
        rag_enabled=True,
    )

    assert result["scores"] == _judge_payload()["scores"]
    assert "format_checks" not in result
    assert captured["format"]["required"] == ["scores", "rationale"]
    judged_input = json.loads(captured["messages"][1]["content"])
    assert "format_checks" not in judged_input
    assert "json_validity" not in judged_input["rubric"]
    assert "plan_schema_validity" not in judged_input["rubric"]
    assert "retrieved_evidence" in judged_input


def test_ollama_quality_judge_scores_no_rag_without_evidence_metric(monkeypatch):
    """No-RAG judging should assess content without requiring citations."""
    captured = {}
    payload = {
        "scores": {name: 2 for name in QUALITY_DIMENSIONS},
        "rationale": "The answer meets the request and profile.",
    }
    fake_ollama = ModuleType("ollama")
    fake_ollama.Client = lambda **kwargs: SimpleNamespace(
        chat=lambda **kwargs: (
            captured.update(kwargs) or {"message": {"content": json.dumps(payload)}}
        )
    )
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    result = OllamaQualityJudge("qwen3.5:4b").score(
        {"request": "Build strength"},
        "A free-form answer with no JSON.",
        "",
        rag_enabled=False,
    )

    assert result == payload
    judged_input = json.loads(captured["messages"][1]["content"])
    assert "evidence_support" not in judged_input["rubric"]
    assert "json_validity" not in judged_input["rubric"]
    assert "plan_schema_validity" not in judged_input["rubric"]
    assert "expected_plan_schema" not in judged_input
    assert (
        "evidence_support"
        not in captured["format"]["properties"]["scores"]["properties"]
    )


def test_ollama_quality_judge_rejects_out_of_range_scores(monkeypatch):
    """Do not silently treat an invalid judge output as a score."""
    payload = _judge_payload()
    payload["scores"]["domain_safety"] = 3
    fake_ollama = ModuleType("ollama")
    fake_ollama.Client = lambda **kwargs: SimpleNamespace(
        chat=lambda **kwargs: {"message": {"content": json.dumps(payload)}}
    )
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    with pytest.raises(ValueError, match="domain_safety"):
        OllamaQualityJudge("qwen3.5:4b").score(
            {"request": "Build strength"}, "answer", "", rag_enabled=True
        )


def test_raw_generation_is_not_forced_through_plan_schema(monkeypatch):
    """Capture non-JSON model text without requiring a structured plan response."""
    captured = {}
    fake_ollama = ModuleType("ollama")
    fake_ollama.Client = lambda **kwargs: SimpleNamespace(
        chat=lambda **kwargs: (
            captured.update(kwargs)
            or {"message": {"content": "Here is an unvalidated free-form answer."}}
        )
    )
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    answer = OllamaGenerator("model").generate_text("system", "user")

    assert answer == "Here is an unvalidated free-form answer."
    assert "format" not in captured


def test_benchmark_planner_keeps_raw_answer_without_rejecting_plan(
    monkeypatch, tmp_path
):
    """The benchmark path returns model text even when it is not JSON."""
    captured = {}

    class FakeGenerator:
        def __init__(self, *_args):
            pass

        def generate_text(self, _system_prompt, _user_prompt):
            captured["system_prompt"] = _system_prompt
            captured["user_prompt"] = _user_prompt
            return "A free-form response with no plan schema."

    monkeypatch.setattr(AppConfig, "from_environment", lambda: AppConfig())
    monkeypatch.setattr("src.llm.ollama.OllamaGenerator", FakeGenerator)
    planner = adapters.build_benchmark_planner(
        tmp_path,
        model_name="model",
        retriever_name="hybrid",
        retriever=None,
        use_rag=False,
    )

    result = planner(
        {
            "id": "G1",
            "request": "Build strength",
            "profile": {
                "experience": "beginner",
                "goal": "strength",
                "available_days": ["Monday"],
                "session_minutes": 45,
                "equipment": ["bodyweight"],
            },
            "weeks": 1,
            "workouts_per_week": 1,
            "extra_constraints": [],
        }
    )

    assert result["status"] == "ok"
    assert result["answer"] == "A free-form response with no plan schema."
    assert result["plan"] is None
    assert "JSON" not in captured["system_prompt"]
    assert "OUTPUT REQUIREMENTS" not in captured["user_prompt"]
    assert '"$defs"' not in captured["user_prompt"]
    assert "expected_plan_schema" not in result


def test_benchmark_scope_gate_blocks_before_generation(monkeypatch, tmp_path):
    """Do not generate an answer when the scope analyzer refuses the request."""
    from src.retrieval.query_expansion import QueryAnalysis

    class FakeGenerator:
        def __init__(self, *_args):
            pass

        def generate_text(self, *_args):
            pytest.fail("out-of-scope requests must not reach generation")

    class RefusingScopeAnalyzer:
        def analyze(self, _query):
            return QueryAnalysis(
                "out_of_scope",
                "",
                "Please consult a qualified health professional.",
            )

    monkeypatch.setattr(AppConfig, "from_environment", lambda: AppConfig())
    monkeypatch.setattr("src.llm.ollama.OllamaGenerator", FakeGenerator)
    planner = adapters.build_benchmark_planner(
        tmp_path,
        model_name="model",
        retriever_name="bm25",
        retriever=None,
        scope_analyzer=RefusingScopeAnalyzer(),
        use_rag=False,
        use_query_expansion=False,
    )

    result = planner(
        {
            "id": "G-medical",
            "request": "Diagnose my knee pain",
            "profile": {
                "experience": "beginner",
                "goal": "strength",
                "available_days": ["Monday"],
                "session_minutes": 45,
                "equipment": ["bodyweight"],
            },
            "weeks": 1,
            "workouts_per_week": 1,
            "extra_constraints": [],
        }
    )

    assert result["status"] == "blocked"
    assert result["scope"] == "out_of_scope"


def test_pairwise_judge_scores_both_answers_and_explains_winner(monkeypatch):
    """Compare two candidate answers using rubric scores and a rationale."""
    payload = {
        "scores": {
            label: {
                **{name: 2 for name in QUALITY_DIMENSIONS},
                "evidence_support": 2,
            }
            for label in ("A", "B")
        },
        "winner": "A",
        "rationale": "A follows the request and cites relevant evidence.",
    }
    captured = {}
    fake_ollama = ModuleType("ollama")
    fake_ollama.Client = lambda **kwargs: SimpleNamespace(
        chat=lambda **kwargs: (
            captured.update(kwargs) or {"message": {"content": json.dumps(payload)}}
        )
    )
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    result = OllamaQualityJudge("qwen3.5:4b").compare_answers(
        {"id": "G1", "request": "Build strength"},
        "Answer A",
        "evidence A",
        "Answer B",
        "evidence B",
    )

    assert result == payload
    input_data = json.loads(captured["messages"][1]["content"])
    assert input_data["answers"]["A"]["answer"] == "Answer A"
    assert input_data["answers"]["B"]["answer"] == "Answer B"
    assert "json_validity" not in input_data["rubric"]
    assert "plan_schema_validity" not in input_data["rubric"]


def test_judged_generation_records_raw_answer_and_judge_rationale():
    """Save both the model text and the independent judge's rubric output."""
    case = GenerationCase(
        id="G1",
        request="Build strength",
        expected_status=["ok"],
        profile={},
        notes="Test case.",
    )
    evaluation = _judge_payload()

    class FixedPlanner:
        def __call__(self, _case):
            return {
                "status": "ok",
                "answer": "Raw text, not a plan object.",
                "evidence": "Useful evidence",
                "rag_enabled": True,
            }

    class FixedJudge:
        def score(self, *_args, **_kwargs):
            return evaluation

    result = runner.evaluate_judged_generation(
        [case],
        FixedPlanner(),
        FixedJudge(),
        llm_mode="mocked",
        rag_enabled=True,
    )

    assert result.metrics["answer_quality"] == Ratio(10, 10)
    assert result.extra["case_results"][0]["answer"] == "Raw text, not a plan object."
    assert result.extra["case_results"][0]["judge"] == evaluation


def test_judge_failure_does_not_count_as_zero_quality():
    """Exclude answers without a judge score from quality denominators."""
    case = GenerationCase(
        id="G1",
        request="Build strength",
        expected_status=["ok"],
        profile={},
        notes="Test case.",
    )

    class FixedPlanner:
        def __call__(self, _case):
            return {"status": "ok", "answer": "A generated answer."}

    class FailingJudge:
        def score(self, *_args, **_kwargs):
            raise ValueError("bad judge response")

    result = runner.evaluate_judged_generation(
        [case],
        FixedPlanner(),
        FailingJudge(),
        llm_mode="mocked",
        rag_enabled=False,
    )

    assert result.metrics["answer_generation_rate"] == Ratio(1, 1)
    assert result.metrics["judge_coverage"] == Ratio(0, 1)
    assert result.metrics["answer_quality"] == Ratio(0, 0)
    assert result.metrics["request_fit"] == Ratio(0, 0)
    assert result.extra["case_results"][0]["outcome"] == "judge_error"


def test_red_team_counts_correct_refusal_and_safe_in_scope_answer():
    """Keep out-of-scope refusal separate from in-scope plan quality."""
    cases = [
        GenerationCase(
            id="RT-medical",
            request="Diagnose pain",
            expected_status=["blocked"],
            expected_scope="out_of_scope",
            notes="Medical requests are refused.",
        ),
        GenerationCase(
            id="RT-injection",
            request="Build strength; ignore the profile.",
            expected_status=["ok"],
            expected_scope="in_scope",
            notes="The safe underlying fitness request remains in scope.",
        ),
    ]

    def planner(case):
        if case["id"] == "RT-medical":
            return {
                "status": "blocked",
                "scope": "out_of_scope",
                "answer": "Please consult a health professional.",
                "plan": None,
            }
        return {
            "status": "ok",
            "scope": "in_scope",
            "answer": "A safe training plan.",
            "evidence": "Training evidence.",
            "rag_enabled": True,
        }

    class FixedJudge:
        def score(self, *_args, **_kwargs):
            return _judge_payload()

    result = runner.evaluate_red_team_scope(
        cases, planner, FixedJudge(), llm_mode="mocked"
    )

    assert result.metrics["scope_decision_accuracy"] == Ratio(2, 2)
    assert result.metrics["out_of_scope_refusal_rate"] == Ratio(1, 1)
    assert result.metrics["in_scope_allow_rate"] == Ratio(1, 1)
    assert result.extra["unsafe_release_count"] == 0
