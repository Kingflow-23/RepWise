"""Exercise the generation tool with real validation and SQLite persistence."""

from datetime import datetime, time, timedelta
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src.agents import approvals
from src.db.repository import StudentRepository
from src.generation.pipeline import PlanPipeline
from src.retrieval.query_expansion import ScopeRefusalError
from src.schemas.models import (
    AgentPlanOptions,
    PlanOptions,
    PlanRequest,
    RetrievedChunk,
    TrainingPlan,
    UserProfile,
)
from src.tools import calendar as calendar_tools
from src.tools.catalog import get_tool_spec
from src.tools.registry import PlanArguments
from src.tools.planning import (
    GenerateWorkoutPlanResult,
    generate_plan_for_agent,
    generate_workout_plan,
)
from tests.conftest import NOW, PARIS, make_context


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


def test_requested_session_duration_is_capped_and_plan_remains_one_week(account):
    repository, user_id = account
    pipeline, _, _ = make_pipeline()
    original_create_plan = pipeline.create_plan
    pipeline.create_plan = Mock(wraps=original_create_plan)

    result = generate_workout_plan(
        PlanOptions(
            focus="Build leg muscles",
            workouts_per_week=1,
            max_session_minutes=30,
        ),
        student_id=user_id,
        repository=repository,
        pipeline=pipeline,
    )

    request = pipeline.create_plan.call_args.args[0]
    assert request.weeks == 1
    assert request.profile.session_minutes == 30
    assert result.status == "completed"
    assert result.plan.weeks[0].workouts[0].estimated_minutes <= 30


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
    schema = get_tool_spec("generate_workout_plan").input_schema
    assert schema["properties"] == AgentPlanOptions.model_json_schema()["properties"]
    assert schema["properties"] == PlanArguments.model_json_schema()["properties"]
    assert "student_id" not in schema["properties"]
    with pytest.raises(ValidationError):
        PlanOptions.model_validate(
            {"focus": "Build legs", "student_id": "another-account"}
        )


def test_agent_plan_is_active_first_then_a_draft_that_keeps_the_current_plan(account):
    repository, user_id = account
    pipeline, _, _ = make_pipeline()
    context = make_context(repository, user_id, pipeline_factory=lambda: pipeline)

    first = generate_plan_for_agent(
        context, focus="Build leg muscles", schedule_scope="weekly"
    )
    second = generate_plan_for_agent(
        context, focus="Build leg muscles again", schedule_scope="weekly"
    )

    assert (first.status, first.plan_status) == ("completed", "active")
    assert (second.status, second.plan_status) == ("completed", "draft")
    assert repository.get_current_plan(user_id)["id"] == first.plan_id


def test_agent_plan_without_a_pipeline_is_unavailable(account):
    repository, user_id = account

    result = generate_plan_for_agent(
        make_context(repository, user_id),
        focus="Build legs",
        schedule_scope="weekly",
    )

    assert result.status == "unavailable"
    assert repository.list_plans(user_id) == []


def test_this_week_plan_uses_only_free_profile_days_and_reduces_workout_count(account):
    repository, user_id = account
    repository.save_profile(
        user_id,
        UserProfile(
            experience="beginner",
            goal="Build leg muscles",
            available_days=[
                "Monday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            session_minutes=45,
            equipment=["bodyweight"],
        ),
    )
    pipeline, _, generator = make_pipeline()
    plan = generator.generate.return_value
    week = plan.weeks[0]
    generator.generate.return_value = plan.model_copy(
        update={
            "weeks": [
                week.model_copy(
                    update={
                        "workouts": [
                            week.workouts[0].model_copy(update={"day": "Saturday"})
                        ]
                    }
                )
            ]
        }
    )
    busy = [
        calendar_tools.BusyPeriod(
            start=datetime.combine(NOW.date() + timedelta(days=offset), time(6), PARIS),
            end=datetime.combine(NOW.date() + timedelta(days=offset), time(22), PARIS),
        )
        for offset in (0, 1, 2, 4)
    ]
    context = make_context(
        repository,
        user_id,
        calendar=calendar_tools.FakeCalendar(busy),
        pipeline_factory=lambda: pipeline,
    )

    result = generate_plan_for_agent(
        context,
        focus="Build leg muscles",
        schedule_scope="this_week",
        workouts_per_week=3,
    )

    assert result.status == "completed"
    assert result.plan_status == "draft"
    assert result.schedule_week_start.isoformat() == "2030-01-07"
    assert [workout.day for workout in result.plan.weeks[0].workouts] == ["Saturday"]
    saved_request = repository.list_plans(user_id)[0]["request"]
    assert saved_request["profile"]["available_days"] == ["Saturday"]
    assert saved_request["workouts_per_week"] == 1
    assert saved_request["schedule_week_start"] == "2030-01-07"
    assert "Available profile days: Saturday" in result.schedule_note
    assert "reduced to 1" in result.schedule_note


def test_this_week_plan_does_not_generate_when_calendar_is_unavailable(account):
    repository, user_id = account
    pipeline, _, generator = make_pipeline()

    class UnavailableCalendar:
        def busy_periods(self, start: datetime, end: datetime):
            if end <= start:
                return []
            raise calendar_tools.CalendarError("Calendar read failed")

    context = make_context(
        repository,
        user_id,
        calendar=UnavailableCalendar(),
        pipeline_factory=lambda: pipeline,
    )

    result = generate_plan_for_agent(
        context, focus="Build legs", schedule_scope="this_week"
    )

    assert result.status == "calendar_unavailable"
    assert "Calendar read failed" in result.message
    generator.generate.assert_not_called()
    assert repository.list_plans(user_id) == []


def test_this_week_plan_requires_a_connected_calendar(account):
    repository, user_id = account
    pipeline, _, generator = make_pipeline()

    result = generate_plan_for_agent(
        make_context(repository, user_id, pipeline_factory=lambda: pipeline),
        focus="Build legs",
    )

    assert result.status == "calendar_not_connected"
    assert "Connect Google Calendar" in result.message
    generator.generate.assert_not_called()
    assert repository.list_plans(user_id) == []


def test_this_week_plan_is_not_generated_when_all_profile_days_are_busy(account):
    repository, user_id = account
    repository.save_profile(
        user_id,
        UserProfile(
            experience="beginner",
            goal="Build leg muscles",
            available_days=["Saturday"],
            session_minutes=45,
            equipment=["bodyweight"],
        ),
    )
    pipeline, _, generator = make_pipeline()
    saturday = NOW.date() + timedelta(days=3)
    calendar = calendar_tools.FakeCalendar(
        [
            calendar_tools.BusyPeriod(
                start=datetime.combine(saturday, time(6), PARIS),
                end=datetime.combine(saturday, time(22), PARIS),
            )
        ]
    )

    result = generate_plan_for_agent(
        make_context(
            repository,
            user_id,
            calendar=calendar,
            pipeline_factory=lambda: pipeline,
        ),
        focus="Build legs",
    )

    assert result.status == "no_available_days"
    assert result.plan_id is None
    generator.generate.assert_not_called()
    assert repository.list_plans(user_id) == []


def test_first_this_week_plan_waits_for_approval_of_exact_calendar_slots(account):
    repository, user_id = account
    repository.save_profile(
        user_id,
        UserProfile(
            experience="beginner",
            goal="Build leg muscles",
            available_days=["Saturday"],
            session_minutes=45,
            equipment=["bodyweight"],
        ),
    )
    pipeline, _, generator = make_pipeline()
    plan = generator.generate.return_value
    week = plan.weeks[0]
    generator.generate.return_value = plan.model_copy(
        update={
            "weeks": [
                week.model_copy(
                    update={
                        "workouts": [
                            week.workouts[0].model_copy(update={"day": "Saturday"})
                        ]
                    }
                )
            ]
        }
    )
    calendar = calendar_tools.FakeCalendar()
    context = make_context(
        repository,
        user_id,
        calendar=calendar,
        pipeline_factory=lambda: pipeline,
    )

    run = approvals.run_tool_with_approval(
        context,
        "generate_workout_plan",
        {"focus": "Build leg muscles"},
        "planning",
    )

    assert run.result.plan_status == "draft"
    assert repository.get_current_plan(user_id) is None
    assert len(run.pending) == 1
    action = repository.get_pending_action(user_id, run.pending[0].action_id)
    assert action["tool_name"] == "activate_plan"
    booking = action["arguments"]["bookings"][0]
    assert booking["start"].startswith("2030-01-12")

    decision = approvals.approve(context, run.pending[0].action_id)

    assert decision.action.result["activated"] is True
    assert decision.action.result["plan_status"] == "active"
    assert repository.get_current_plan(user_id)["id"] == run.result.plan_id
    assert len(calendar.events) == 1


def test_this_week_plan_is_not_approved_if_slots_disappear_before_preparation(account):
    repository, user_id = account
    repository.save_profile(
        user_id,
        UserProfile(
            experience="beginner",
            goal="Build leg muscles",
            available_days=["Saturday"],
            session_minutes=45,
            equipment=["bodyweight"],
        ),
    )
    pipeline, _, generator = make_pipeline()
    plan = generator.generate.return_value
    week = plan.weeks[0]
    generator.generate.return_value = plan.model_copy(
        update={
            "weeks": [
                week.model_copy(
                    update={
                        "workouts": [
                            week.workouts[0].model_copy(update={"day": "Saturday"})
                        ]
                    }
                )
            ]
        }
    )

    class CalendarWithChangingAvailability:
        reads = 0

        def busy_periods(self, start: datetime, end: datetime):
            self.reads += 1
            if self.reads == 1:
                return []
            return [calendar_tools.BusyPeriod(start=start, end=end)]

    context = make_context(
        repository,
        user_id,
        calendar=CalendarWithChangingAvailability(),
        pipeline_factory=lambda: pipeline,
    )

    run = approvals.run_tool_with_approval(
        context,
        "generate_workout_plan",
        {"focus": "Build leg muscles"},
        "planning",
    )

    assert run.result.plan_status == "draft"
    assert "Activation was not proposed" in run.result.schedule_note
    assert run.pending == []
    assert repository.get_current_plan(user_id) is None


def test_this_week_plan_stays_draft_if_approved_slot_becomes_busy(account):
    repository, user_id = account
    repository.save_profile(
        user_id,
        UserProfile(
            experience="beginner",
            goal="Build leg muscles",
            available_days=["Saturday"],
            session_minutes=45,
            equipment=["bodyweight"],
        ),
    )
    pipeline, _, generator = make_pipeline()
    plan = generator.generate.return_value
    week = plan.weeks[0]
    generator.generate.return_value = plan.model_copy(
        update={
            "weeks": [
                week.model_copy(
                    update={
                        "workouts": [
                            week.workouts[0].model_copy(update={"day": "Saturday"})
                        ]
                    }
                )
            ]
        }
    )
    calendar = calendar_tools.FakeCalendar()
    context = make_context(
        repository,
        user_id,
        calendar=calendar,
        pipeline_factory=lambda: pipeline,
    )
    run = approvals.run_tool_with_approval(
        context,
        "generate_workout_plan",
        {"focus": "Build leg muscles"},
        "planning",
    )
    action = repository.get_pending_action(user_id, run.pending[0].action_id)
    approved_start = datetime.fromisoformat(action["arguments"]["bookings"][0]["start"])
    calendar.busy.append(
        calendar_tools.BusyPeriod(
            start=approved_start, end=approved_start + timedelta(minutes=45)
        )
    )

    decision = approvals.approve(context, run.pending[0].action_id)

    assert decision.action.result["activated"] is False
    assert decision.action.result["plan_status"] == "draft"
    assert repository.get_current_plan(user_id) is None
    assert calendar.events == {}


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
