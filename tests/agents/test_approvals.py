"""The approval gate: what runs, what waits, and who can approve it."""

from datetime import timedelta
from types import SimpleNamespace

import pytest

from src.agents import approvals
from src.schemas.models import PlanResult, TrainingPlan
from tests.conftest import NOW, make_context

MOVE = {"moves": [{"week_number": 1, "workout_index": 0, "new_day": "Saturday"}]}


def first_day(demo):
    return demo.repository.get_plan(demo.student_id, demo.plan_id)["plan"]["weeks"][0][
        "workouts"
    ][0]["day"]


@pytest.fixture
def saturday(demo):
    """Make Saturday an available day, so the demo move is valid."""
    profile = demo.repository.get_profile(demo.student_id)
    demo.repository.save_profile(
        demo.student_id,
        profile.model_copy(
            update={"available_days": [*profile.available_days, "Saturday"]}
        ),
    )


def test_a_read_runs_immediately(demo):
    run = approvals.run_tool_with_approval(
        demo.context, "get_timetable", {}, "orchestrator"
    )

    assert run.succeeded and run.pending == []
    assert run.result.plan_id == demo.plan_id


def test_a_consequential_write_is_held_and_changes_nothing(demo, saturday):
    run = approvals.run_tool_with_approval(
        demo.context, "organize_schedule", MOVE, "schedule"
    )

    assert run.result.status == "pending_approval"
    assert first_day(demo) == "Monday"
    stored = demo.repository.get_pending_action(demo.student_id, run.result.action_id)
    assert (stored["tool_name"], stored["proposed_by"], stored["status"]) == (
        "organize_schedule",
        "schedule",
        "pending",
    )
    assert stored["arguments"]["moves"][0]["new_day"] == "Saturday"


def test_a_catch_up_is_held_as_the_exact_moves_code_chose(demo, saturday):
    # Monday is logged as missed, then a catch-up is asked.
    approvals.run_tool_with_approval(
        demo.context,
        "save_workout_feedback",
        {
            "week_number": 1,
            "workout_index": 0,
            "completion_status": "not_completed",
            "difficulty": "just_right",
        },
        "performance",
    )

    run = approvals.run_tool_with_approval(
        demo.context, "organize_schedule", {"catch_up_week": 1}, "schedule"
    )

    stored = demo.repository.get_pending_action(demo.student_id, run.result.action_id)
    assert stored["arguments"]["catch_up_week"] is None
    assert stored["arguments"]["moves"][0] == {
        "week_number": 1,
        "workout_index": 0,
        "new_day": "Wednesday",
    }
    assert "Monday (missed) → Wednesday" in run.result.reason
    assert first_day(demo) == "Monday"


def test_approval_runs_exactly_the_stored_action_once(demo, saturday):
    held = approvals.run_tool_with_approval(
        demo.context, "organize_schedule", MOVE, "schedule"
    ).result

    decision = approvals.approve(demo.context, held.action_id)

    assert decision.executed and decision.action.status == "executed"
    assert first_day(demo) == "Saturday"
    with pytest.raises(approvals.ApprovalError) as error:
        approvals.approve(demo.context, held.action_id)
    assert error.value.code == "already_decided"


def test_another_student_cannot_approve_or_see_the_action(
    demo, other_student, saturday
):
    held = approvals.run_tool_with_approval(
        demo.context, "organize_schedule", MOVE, "schedule"
    ).result
    other = make_context(demo.repository, other_student)

    with pytest.raises(approvals.ApprovalError) as error:
        approvals.approve(other, held.action_id)

    assert error.value.code == "not_found"
    assert approvals.list_actions(other) == []
    assert first_day(demo) == "Monday"


def test_an_expired_action_cannot_be_approved(demo, saturday):
    held = approvals.run_tool_with_approval(
        demo.context, "organize_schedule", MOVE, "schedule"
    ).result
    later = make_context(
        demo.repository, demo.student_id, clock=lambda: NOW + timedelta(hours=2)
    )

    with pytest.raises(approvals.ApprovalError) as error:
        approvals.approve(later, held.action_id)

    assert error.value.code == "expired"
    assert approvals.list_actions(later, None)[0].status == "expired"


def test_a_tampered_action_is_refused(demo, saturday):
    held = approvals.run_tool_with_approval(
        demo.context, "organize_schedule", MOVE, "schedule"
    ).result
    with demo.repository._connect() as connection:
        connection.execute(
            "UPDATE pending_actions SET arguments_json=? WHERE id=?",
            (
                '{"moves": [{"week_number": 1, "workout_index": 0, "new_day": "Friday"}]}',
                held.action_id,
            ),
        )

    with pytest.raises(approvals.ApprovalError) as error:
        approvals.approve(demo.context, held.action_id)

    assert error.value.code == "tampered"


def test_a_declined_action_never_runs(demo, saturday):
    held = approvals.run_tool_with_approval(
        demo.context, "organize_schedule", MOVE, "schedule"
    ).result

    decision = approvals.decline(demo.context, held.action_id)

    assert decision.action.status == "declined" and not decision.executed
    assert first_day(demo) == "Monday"


def test_a_failed_approved_action_is_recorded_as_failed(demo):
    # Sunday is not an available day, so the approved move fails validation.
    sunday = {"moves": [{"week_number": 1, "workout_index": 0, "new_day": "Sunday"}]}
    held = approvals.run_tool_with_approval(
        demo.context, "organize_schedule", sunday, "schedule"
    ).result

    decision = approvals.approve(demo.context, held.action_id)

    assert decision.action.status == "failed"
    assert decision.action.result["code"] == "day_not_available"


def test_invalid_arguments_are_rejected_before_anything_is_stored(demo):
    run = approvals.run_tool_with_approval(
        demo.context, "organize_schedule", {"moves": []}, "schedule"
    )

    assert run.result.code == "invalid_arguments"
    assert demo.repository.list_pending_actions(demo.student_id) == []


def test_feedback_mentioning_pain_needs_approval_and_plain_feedback_does_not(demo):
    painful = {
        "week_number": 1,
        "workout_index": 0,
        "completion_status": "completed",
        "difficulty": "just_right",
        "comment": "My knee hurts",
    }
    plain = {**painful, "workout_index": 1, "comment": "Felt strong"}

    held = approvals.run_tool_with_approval(
        demo.context, "save_workout_feedback", painful, "performance"
    )
    saved = approvals.run_tool_with_approval(
        demo.context, "save_workout_feedback", plain, "performance"
    )

    assert held.result.status == "pending_approval"
    assert saved.result.saved
    assert [
        f.workout_index
        for f in demo.repository.list_workout_feedback(demo.student_id, demo.plan_id)
    ] == [1]


def test_a_draft_plan_gets_an_activation_waiting_for_approval(demo):
    record = demo.repository.get_plan(demo.student_id, demo.plan_id)

    class Pipeline:
        retriever = generator = None

        def create_plan(self, request, previous_plan=None):
            return PlanResult(
                status="completed",
                message="ok",
                plan=TrainingPlan.model_validate(record["plan"]),
            )

    context = make_context(demo.repository, demo.student_id, pipeline_factory=Pipeline)
    run = approvals.run_tool_with_approval(
        context, "generate_workout_plan", {"focus": "Legs"}, "planning"
    )

    assert run.result.plan_status == "draft"
    assert [item.tool_name for item in run.pending] == ["activate_plan"]
    assert demo.repository.get_current_plan(demo.student_id)["id"] == demo.plan_id

    approvals.approve(context, run.pending[0].action_id)
    assert demo.repository.get_current_plan(demo.student_id)["id"] == run.result.plan_id


def test_an_unknown_tool_is_a_structured_failure(demo):
    assert (
        approvals.run_tool_with_approval(
            demo.context, "drop_database", {}, "x"
        ).result.code
        == "unknown_tool"
    )


def test_a_crashing_tool_is_reported_not_raised(demo, monkeypatch):
    from src.tools import registry

    broken = SimpleNamespace(**{**registry.IMPLEMENTATIONS["get_timetable"].__dict__})
    broken.run = lambda context: 1 / 0
    monkeypatch.setitem(registry.IMPLEMENTATIONS, "get_timetable", broken)

    assert (
        approvals.run_tool_with_approval(
            demo.context, "get_timetable", {}, "x"
        ).result.code
        == "tool_crashed"
    )
