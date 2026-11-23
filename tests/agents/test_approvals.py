"""The approval gate: what runs, what waits, and who can approve it."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from src.agents import approvals
from src.schemas.models import PlanResult, TrainingPlan
from src.tools import calendar as calendar_tools
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


def test_a_move_onto_an_occupied_day_is_rejected_before_approval(demo):
    plan = demo.repository.get_plan(demo.student_id, demo.plan_id)["plan"]
    occupied_day = plan["weeks"][0]["workouts"][2]["day"]

    run = approvals.run_tool_with_approval(
        demo.context,
        "organize_schedule",
        {"moves": [{"week_number": 1, "workout_index": 1, "new_day": occupied_day}]},
        "schedule",
    )

    assert run.result.code == "day_conflict"
    assert demo.repository.list_pending_actions(demo.student_id) == []


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


def test_empty_booking_request_prepares_all_times_for_one_approval(demo):
    calendar = calendar_tools.FakeCalendar()
    context = make_context(demo.repository, demo.student_id, calendar=calendar)
    run = approvals.run_tool_with_approval(
        context, "add_workouts_to_calendar", {}, "schedule"
    )

    assert run.result.status == "pending_approval"
    assert calendar.events == {}
    pending = demo.repository.get_pending_action(demo.student_id, run.result.action_id)
    assert pending["status"] == "pending"
    bookings = pending["arguments"]["bookings"]
    assert len(bookings) == 2
    starts = [item["start"] for item in bookings]
    assert len(starts) == len(set(starts))
    assert "Week 1:" in pending["reason"]

    booked = approvals.approve(context, run.result.action_id)

    assert booked.executed
    assert booked.action.result["outcomes"]
    assert all(item["status"] == "booked" for item in booked.action.result["outcomes"])
    assert len(calendar.events) == len(booked.action.result["outcomes"])
    assert all(
        item["workout"]["exercises"] for item in booked.action.result["outcomes"]
    )


def test_empty_booking_request_is_rejected_if_any_workout_has_no_slot(demo):
    class FullCalendar(calendar_tools.FakeCalendar):
        def busy_periods(self, start, end):
            return [calendar_tools.BusyPeriod(start=start, end=end)]

    context = make_context(demo.repository, demo.student_id, calendar=FullCalendar())

    run = approvals.run_tool_with_approval(
        context, "add_workouts_to_calendar", {}, "schedule"
    )

    assert run.result.code == "no_free_slots"
    assert demo.repository.list_pending_actions(demo.student_id) == []


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


def test_schedule_is_revalidated_when_approval_is_applied(demo, saturday):
    held = approvals.run_tool_with_approval(
        demo.context, "organize_schedule", MOVE, "schedule"
    ).result

    profile = demo.repository.get_profile(demo.student_id)
    demo.repository.save_profile(
        demo.student_id,
        profile.model_copy(
            update={
                "available_days": [
                    day for day in profile.available_days if day != "Saturday"
                ]
            }
        ),
    )
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
        context,
        "generate_workout_plan",
        {"focus": "Legs", "schedule_scope": "weekly"},
        "planning",
    )

    assert run.result.plan_status == "draft"
    assert [item.tool_name for item in run.pending] == ["activate_plan"]
    assert demo.repository.get_current_plan(demo.student_id)["id"] == demo.plan_id

    approvals.approve(context, run.pending[0].action_id)
    assert demo.repository.get_current_plan(demo.student_id)["id"] == run.result.plan_id


def test_approving_a_draft_plan_schedules_it_in_the_connected_calendar(demo):
    record = demo.repository.get_plan(demo.student_id, demo.plan_id)

    class Pipeline:
        retriever = generator = None

        def create_plan(self, request, previous_plan=None):
            return PlanResult(
                status="completed",
                message="ok",
                plan=TrainingPlan.model_validate(record["plan"]),
            )

    calendar = calendar_tools.FakeCalendar()
    context = make_context(
        demo.repository,
        demo.student_id,
        calendar=calendar,
        pipeline_factory=Pipeline,
    )
    run = approvals.run_tool_with_approval(
        context,
        "generate_workout_plan",
        {"focus": "Legs", "schedule_scope": "weekly"},
        "planning",
    )
    pending = demo.repository.get_pending_action(
        demo.student_id, run.pending[0].action_id
    )
    prepared = pending["arguments"]["bookings"]
    assert prepared
    assert "Calendar booking:" in pending["reason"]
    assert "Europe/Paris" in pending["reason"]

    decision = approvals.approve(context, run.pending[0].action_id)

    assert decision.executed
    assert decision.action.result["calendar_status"] == "scheduled"
    assert decision.action.result["calendar_workouts"] == len(prepared)
    assert len(calendar.events) == decision.action.result["calendar_workouts"]


def test_activation_uses_approved_times_and_does_not_silently_reschedule(demo):
    record = demo.repository.get_plan(demo.student_id, demo.plan_id)

    class Pipeline:
        retriever = generator = None

        def create_plan(self, request, previous_plan=None):
            return PlanResult(
                status="completed",
                message="ok",
                plan=TrainingPlan.model_validate(record["plan"]),
            )

    calendar = calendar_tools.FakeCalendar()
    context = make_context(
        demo.repository,
        demo.student_id,
        calendar=calendar,
        pipeline_factory=Pipeline,
    )
    run = approvals.run_tool_with_approval(
        context,
        "generate_workout_plan",
        {"focus": "Legs", "schedule_scope": "weekly"},
        "planning",
    )
    action_id = run.pending[0].action_id
    pending = demo.repository.get_pending_action(demo.student_id, action_id)
    approved_start = datetime.fromisoformat(
        pending["arguments"]["bookings"][0]["start"]
    )
    calendar.busy.append(
        calendar_tools.BusyPeriod(
            start=approved_start, end=approved_start + timedelta(hours=2)
        )
    )

    decision = approvals.approve(context, action_id)

    assert decision.action.result["calendar_status"] == "partial"
    assert decision.action.result["calendar_workouts"] == (
        len(pending["arguments"]["bookings"]) - 1
    )
    assert len(calendar.events) == decision.action.result["calendar_workouts"]
    assert any(
        outcome["status"] == "conflict"
        for outcome in decision.action.result["calendar_result"]["outcomes"]
    )


def test_activation_with_any_missing_slot_approves_without_partial_booking(demo):
    record = demo.repository.get_plan(demo.student_id, demo.plan_id)

    class Pipeline:
        retriever = generator = None

        def create_plan(self, request, previous_plan=None):
            return PlanResult(
                status="completed",
                message="ok",
                plan=TrainingPlan.model_validate(record["plan"]),
            )

    class FullCalendar(calendar_tools.FakeCalendar):
        def busy_periods(self, start, end):
            return [calendar_tools.BusyPeriod(start=start, end=end)]

    calendar = FullCalendar()
    context = make_context(
        demo.repository,
        demo.student_id,
        calendar=calendar,
        pipeline_factory=Pipeline,
    )
    run = approvals.run_tool_with_approval(
        context,
        "generate_workout_plan",
        {"focus": "Legs", "schedule_scope": "weekly"},
        "planning",
    )
    pending = demo.repository.get_pending_action(
        demo.student_id, run.pending[0].action_id
    )

    assert pending["arguments"]["bookings"] == []
    assert "No calendar events will be added" in pending["reason"]
    assert "no free time" in pending["reason"].lower()

    decision = approvals.approve(context, run.pending[0].action_id)

    assert decision.action.result["calendar_status"] == "no_slots"
    assert decision.action.result["calendar_workouts"] == 0
    assert calendar.events == {}


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
