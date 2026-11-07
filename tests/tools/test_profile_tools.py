"""The orchestrator's read tools: profile, current plan, timetable and one workout."""

from src.schemas.models import PlanRequest, PlanResult, TrainingPlan
from src.tools import profile
from tests.conftest import make_context


def save_copy_of_plan(demo, status):
    """Save another plan with the demo plan's content."""
    record = demo.repository.get_plan(demo.student_id, demo.plan_id)
    return demo.repository.save_plan(
        demo.student_id,
        PlanRequest.model_validate(record["request"]),
        PlanResult(
            status="completed",
            message="Copy",
            plan=TrainingPlan.model_validate(record["plan"]),
        ),
        status=status,
    )


def test_profile_is_the_saved_profile(demo):
    result = profile.get_user_profile(demo.context)

    assert result.ok
    assert result.profile == demo.repository.get_profile(demo.student_id)


def test_timetable_gives_ids_other_tools_need(demo):
    result = profile.get_timetable(demo.context)

    assert result.plan_id == demo.plan_id
    assert [
        (w.week_number, w.workout_index, w.day) for w in result.planned_workouts
    ] == [
        (1, 0, "Monday"),
        (1, 1, "Wednesday"),
        (1, 2, "Friday"),
    ]
    assert all(w.plan_id == demo.plan_id for w in result.planned_workouts)


def test_current_plan_ignores_drafts_and_archives_on_activation(demo):
    draft_id = save_copy_of_plan(demo, "draft")

    assert profile.get_current_plan(demo.context).plan_id == demo.plan_id
    demo.repository.activate_plan(demo.student_id, draft_id)

    current = profile.get_current_plan(demo.context)
    assert (current.plan_id, current.status) == (draft_id, "active")
    assert (
        demo.repository.get_plan(demo.student_id, demo.plan_id)["status"] == "archived"
    )


def test_get_workout_returns_feedback_and_performance(demo):
    demo.repository.save_workout_performance(
        demo.student_id, demo.plan_id, 1, 0, "Goblet squat", 3, "10", 12.0
    )

    result = profile.get_workout(demo.context, 1, 0)

    assert result.workout.day == "Monday"
    assert result.performance[0].weight_kg == 12.0


def test_unknown_workout_and_plan_are_structured_failures(demo):
    assert profile.get_workout(demo.context, 9, 0).code == "workout_not_found"
    assert (
        profile.get_workout(demo.context, 1, 0, plan_id="nope").code == "plan_not_found"
    )


def test_a_student_never_reads_another_students_data(demo, other_student):
    other = make_context(demo.repository, other_student)

    # The other account's own plan is returned, never the demo account's.
    assert profile.get_timetable(other).plan_id != demo.plan_id
    assert (
        profile.get_workout(other, 1, 0, plan_id=demo.plan_id).code == "plan_not_found"
    )


def test_a_student_without_a_profile_or_plan(demo):
    nobody = make_context(demo.repository, "nobody")

    assert profile.get_user_profile(nobody).code == "student_not_found"
    assert profile.get_current_plan(nobody).code == "plan_not_found"


def test_a_database_failure_is_reported_not_raised(demo, tmp_path):
    from src.db.repository import StudentRepository

    broken = make_context(
        StudentRepository(tmp_path / "missing" / "dir" / "x.sqlite3"), "s"
    )

    assert profile.get_user_profile(broken).code == "storage_error"
