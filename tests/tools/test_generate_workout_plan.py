"""Exercise the generation tool with real validation and SQLite persistence."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src.db.repository import StudentRepository
from src.generation.pipeline import PlanPipeline
from src.retrieval.query_expansion import ScopeRefusalError
from src.schemas.models import (
    PlanOptions,
    PlanRequest,
    RetrievedChunk,
    TrainingPlan,
    UserProfile,
)
from src.tools.generate_workout_plan import (
    GenerateWorkoutPlanResult,
    generate_workout_plan,
)


@pytest.fixture
def account(tmp_path):
    repository = StudentRepository(tmp_path / "plans.sqlite3")
    repository.initialize()
    user_id = repository.create_user("Test User", "not-a-real-password-hash")
    repository.save_profile(
        user_id,
        UserProfile(
            experience="beginner",
            goal="Build leg muscles",
            available_days=["Monday"],
            session_minutes=45,
            equipment=["bodyweight"],
        ),
    )
    return repository, user_id


def make_pipeline():
    chunks = [
        RetrievedChunk(
            chunk_id=f"chunk-{i}",
            video_id=f"video-{i}",
            passage_ids=[f"passage-{i}"],
            text="Use controlled squats and progress gradually.",
            title="Leg training",
            video_url="https://example.com/legs",
        )
        for i in range(5)
    ]
    plan = TrainingPlan.model_validate(
        {
            "title": "Beginner leg training",
            "summary": "One bodyweight session.",
            "safety_note": "Stop if an exercise causes pain.",
            "weeks": [
                {
                    "number": 1,
                    "objective": "Practice controlled repetitions.",
                    "workouts": [
                        {
                            "day": "Monday",
                            "focus": "Leg muscles",
                            "estimated_minutes": 30,
                            "exercises": [
                                {
                                    "name": "Bodyweight squat",
                                    "sets": 3,
                                    "repetitions": "8-12",
                                    "rest_seconds": 60,
                                    "estimated_minutes": 10,
                                    "coaching_note": "Use controlled repetitions.",
                                    "equipment": ["bodyweight"],
                                    "source_ids": ["passage-0"],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )
    # Avoid Mock dynamically claiming the optional query-aware protocol.
    retriever = Mock(spec=["search"])
    retriever.search.return_value = chunks
    generator = Mock()
    generator.generate.return_value = plan
    return PlanPipeline(retriever, generator), retriever, generator


def test_leg_program_returns_json_and_saves_profile_request_and_sources(account):
    repository, user_id = account
    pipeline, retriever, generator = make_pipeline()
    options = PlanOptions(
        focus="Generate a program to build leg muscles", workouts_per_week=1
    )
    result = generate_workout_plan(
        options, student_id=user_id, repository=repository, pipeline=pipeline
    )
    assert isinstance(result, GenerateWorkoutPlanResult)
    assert result.status == "completed"
    assert result.plan_id
    assert result.plan.weeks[0].workouts[0].focus == "Leg muscles"
    assert len(result.sources) == 5
    assert (
        GenerateWorkoutPlanResult.model_validate_json(result.model_dump_json())
        == result
    )
    saved = repository.list_plans(user_id)[0]
    assert saved["id"] == result.plan_id
    assert saved["request"]["focus"] == options.focus
    assert saved["request"]["weeks"] == 1
    assert saved["sources"][0]["source_id"] == "passage-0"
    assert "leg muscles" in retriever.search.call_args.args[0]
    generator.generate.assert_called_once()


def test_refinement_receives_previous_saved_plan(account):
    repository, user_id = account
    pipeline, _, generator = make_pipeline()
    first = generate_workout_plan(
        PlanOptions(focus="Build leg muscles"),
        student_id=user_id,
        repository=repository,
        pipeline=pipeline,
    )
    generator.reset_mock()
    generate_workout_plan(
        PlanOptions(focus="Build leg muscles", refinement_instructions="More variety"),
        student_id=user_id,
        repository=repository,
        pipeline=pipeline,
    )
    prompt = generator.generate.call_args.args[1]
    assert "More variety" in prompt
    assert first.plan.title in prompt
    assert len(repository.list_plans(user_id)) == 2


def test_missing_profile_stops_before_generation(account):
    repository, _ = account
    user_id = repository.create_user("No Profile", "unused")
    pipeline = Mock()
    result = generate_workout_plan(
        PlanOptions(focus="Build leg muscles"),
        student_id=user_id,
        repository=repository,
        pipeline=pipeline,
    )
    assert result.status == "missing_profile"
    assert result.plan_id is None
    pipeline.create_plan.assert_not_called()


def test_impossible_schedule_does_not_generate_or_save(account):
    repository, user_id = account
    pipeline = Mock()
    result = generate_workout_plan(
        PlanOptions(focus="Build leg muscles", workouts_per_week=2),
        student_id=user_id,
        repository=repository,
        pipeline=pipeline,
    )
    assert result.status == "invalid_request"
    assert "cannot exceed" in result.message
    pipeline.create_plan.assert_not_called()
    assert repository.list_plans(user_id) == []


@pytest.mark.parametrize(
    "failure",
    ["out_of_scope", "retrieval_error", "insufficient_evidence", "generation_error"],
)
def test_pipeline_failures_return_structured_result_without_saving(account, failure):
    repository, user_id = account
    pipeline, retriever, generator = make_pipeline()
    if failure == "out_of_scope":
        retriever.search.side_effect = ScopeRefusalError(
            "out_of_scope", "Unsupported request"
        )
    elif failure == "retrieval_error":
        retriever.search.side_effect = RuntimeError("Index unavailable")
    elif failure == "insufficient_evidence":
        retriever.search.return_value = []
    else:
        generator.generate.side_effect = RuntimeError("Model unavailable")
    result = generate_workout_plan(
        PlanOptions(focus="Build leg muscles"),
        student_id=user_id,
        repository=repository,
        pipeline=pipeline,
    )
    assert result.status == failure
    assert result.plan_id is None
    assert repository.list_plans(user_id) == []
    if failure == "generation_error":
        assert generator.generate.call_count == 3
    else:
        generator.generate.assert_not_called()


def test_tool_catalog_uses_actual_input_schema():
    catalog = json.loads(
        (Path(__file__).resolve().parents[2] / "docs/agent/tools.json").read_text()
    )
    tool = next(
        item for item in catalog["tools"] if item["name"] == "generate_workout_plan"
    )
    assert tool["parameters"] == PlanOptions.model_json_schema()
    assert "student_id" not in tool["parameters"]["properties"]
    with pytest.raises(ValidationError):
        PlanOptions.model_validate(
            {"focus": "Build legs", "student_id": "another-account"}
        )


def test_completed_result_cannot_claim_success_without_a_saved_id():
    pipeline, _, _ = make_pipeline()
    result = pipeline.create_plan(
        PlanRequest(
            focus="Build legs",
            profile=UserProfile(
                experience="beginner",
                goal="Build legs",
                available_days=["Monday"],
                session_minutes=45,
                equipment=["bodyweight"],
            ),
        )
    )
    with pytest.raises(ValidationError, match="saved plan ID"):
        GenerateWorkoutPlanResult(**result.model_dump())


def test_database_failure_does_not_return_a_completed_result(account, monkeypatch):
    repository, user_id = account
    pipeline, _, _ = make_pipeline()
    monkeypatch.setattr(
        repository, "save_plan", Mock(side_effect=OSError("Database unavailable"))
    )
    with pytest.raises(OSError, match="Database unavailable"):
        generate_workout_plan(
            PlanOptions(focus="Build leg muscles"),
            student_id=user_id,
            repository=repository,
            pipeline=pipeline,
        )
    assert repository.list_plans(user_id) == []
