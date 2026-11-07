"""Feedback, history, performance, adaptation, schedule and recommendation tools."""

import json
from types import SimpleNamespace

import pytest

from src.schemas.models import PlanRequest, PlanResult, TrainingPlan, Workout
from src.tools import training
from src.tools.profile import PerformanceRecord, get_timetable
from tests.conftest import make_context

SQUAT = PerformanceRecord(
    exercise_name="goblet squat", sets_completed=3, reps_per_set="10", weight_kg=12
)


class EasierGenerator:
    """Stand in for the model: cut one set from each exercise of the target."""

    def __init__(self):
        self.calls = 0

    def generate_workout(self, _system, user_prompt, _schema=None):
        self.calls += 1
        target = Workout.model_validate(json.loads(user_prompt)["target_workout"])
        return target.model_copy(
            update={
                "exercises": [
                    item.model_copy(update={"sets": item.sets - 1})
                    for item in target.exercises
                ]
            }
        )


def with_generator(demo, generator):
    """Give the demo context a pipeline whose generator is ``generator``."""
    pipeline = SimpleNamespace(generator=generator, retriever=None)
    return make_context(
        demo.repository, demo.student_id, pipeline_factory=lambda: pipeline
    )


def give_feedback(demo, index=0, comment=None, status="completed"):
    return training.save_workout_feedback(
        demo.context, 1, index, status, "slightly_too_hard", comment
    )


# --- feedback and the stored health flag ---------------------------------


def test_feedback_is_saved_with_the_health_decision(demo):
    plain = give_feedback(demo, 0, "The last set was difficult")
    painful = give_feedback(demo, 1, "J'ai une douleur au genou")
    no_comment = give_feedback(demo, 2)

    assert plain.feedback.health_concern is False
    assert painful.feedback.health_concern is True
    assert no_comment.feedback.health_concern is None


def test_a_missed_workout_is_recorded(demo):
    result = give_feedback(demo, 0, status="not_completed")

    assert result.feedback.completion_status == "not_completed"
    assert training.get_workout_history(demo.context).summary.not_completed == 1


def test_feedback_for_an_unknown_workout_saves_nothing(demo):
    result = training.save_workout_feedback(
        demo.context, 1, 9, "completed", "just_right"
    )

    assert result.code == "workout_not_found"
    assert demo.repository.list_workout_feedback(demo.student_id, demo.plan_id) == []


def test_feedback_cannot_target_another_students_plan(demo, other_student):
    other = make_context(demo.repository, other_student)

    result = training.save_workout_feedback(
        other, 1, 0, "completed", "just_right", plan_id=demo.plan_id
    )

    assert result.code == "plan_not_found"


# --- performance and history ---------------------------------------------


def test_performance_uses_the_plan_exercise_name_and_appears_in_history(demo):
    result = training.log_workout_performance(demo.context, 1, 0, [SQUAT])

    assert result.records[0].exercise_name == "Goblet squat"
    assert (
        training.get_workout_history(demo.context).workouts[0].performance[0].weight_kg
        == 12
    )


def test_one_wrong_exercise_saves_nothing(demo):
    wrong = SQUAT.model_copy(update={"exercise_name": "Bench press"})

    result = training.log_workout_performance(demo.context, 1, 0, [SQUAT, wrong])

    assert result.code == "exercise_not_found"
    assert training.get_workout_history(demo.context).workouts == []


def test_history_summary_is_counted_by_code(demo):
    give_feedback(demo, 0, "J'ai une douleur au genou")
    give_feedback(demo, 1)

    summary = training.get_workout_history(demo.context).summary

    assert (summary.workouts_logged, summary.completed) == (2, 2)
    assert summary.difficulty_counts == {"slightly_too_hard": 2}
    assert summary.health_concern_reports == 1


def test_history_of_a_new_student_is_empty_not_an_error(demo):
    record = demo.repository.get_plan(demo.student_id, demo.plan_id)
    fresh = demo.repository.create_user("Fresh", "!no-login")
    demo.repository.save_profile(
        fresh, PlanRequest.model_validate(record["request"]).profile
    )

    history = training.get_workout_history(make_context(demo.repository, fresh))

    assert (
        history.ok and history.workouts == [] and history.summary.workouts_logged == 0
    )


# --- recommendations -------------------------------------------------------


def test_recommendation_is_kept_without_changing_the_plan(demo):
    before = demo.repository.get_plan(demo.student_id, demo.plan_id)["plan"]

    result = training.save_recommendation(
        demo.context, "Add a squat set", "To progress", source_ids=["s1"]
    )

    assert result.saved and result.plan_id == demo.plan_id
    pending = training.get_workout_history(demo.context).pending_recommendations
    assert pending[0].source_ids == ["s1"]
    assert demo.repository.get_plan(demo.student_id, demo.plan_id)["plan"] == before


# --- adaptation guardrails ---------------------------------------------------


def test_next_workout_is_adapted_and_saved(demo):
    give_feedback(demo)
    context = with_generator(demo, EasierGenerator())
    logged_before = demo.repository.get_plan(demo.student_id, demo.plan_id)["plan"][
        "weeks"
    ][0]["workouts"][0]

    result = training.update_upcoming_workout(context, 1, 0)

    assert result.changed and result.workout_index == 1
    saved = demo.repository.get_plan(demo.student_id, demo.plan_id)["plan"]
    assert (
        saved["weeks"][0]["workouts"][1]["exercises"][0]["sets"]
        == result.after.exercises[0].sets
    )
    # The logged workout is history: it never changes.
    assert saved["weeks"][0]["workouts"][0] == logged_before


def test_same_feedback_cannot_adapt_twice(demo):
    give_feedback(demo)
    context = with_generator(demo, EasierGenerator())
    training.update_upcoming_workout(context, 1, 0)

    assert training.update_upcoming_workout(context, 1, 0).code == "already_adapted"


@pytest.mark.parametrize("comment", ["My knee hurts", "J'ai mal au genou"])
def test_a_health_concern_stops_adaptation_without_calling_the_model(demo, comment):
    give_feedback(demo, comment=comment)
    generator = EasierGenerator()

    result = training.update_upcoming_workout(with_generator(demo, generator), 1, 0)

    assert result.code == "health_concern"
    assert "medical advice" in result.error
    assert generator.calls == 0


def test_adaptation_needs_feedback_first(demo):
    result = training.update_upcoming_workout(
        with_generator(demo, EasierGenerator()), 1, 0
    )

    assert result.code == "feedback_not_found"


def test_last_workout_has_nothing_to_adapt(demo):
    give_feedback(demo, 2)

    result = training.update_upcoming_workout(
        with_generator(demo, EasierGenerator()), 1, 2
    )

    assert result.code == "no_upcoming_workout"


def test_a_model_that_renames_exercises_changes_nothing(demo):
    class Renaming(EasierGenerator):
        def generate_workout(self, *arguments):
            workout = super().generate_workout(*arguments)
            renamed = workout.exercises[0].model_copy(update={"name": "Something else"})
            return workout.model_copy(
                update={"exercises": [renamed, *workout.exercises[1:]]}
            )

    give_feedback(demo)
    before = demo.repository.get_plan(demo.student_id, demo.plan_id)["plan"]

    result = training.update_upcoming_workout(with_generator(demo, Renaming()), 1, 0)

    assert result.code == "adaptation_failed"
    assert demo.repository.get_plan(demo.student_id, demo.plan_id)["plan"] == before


def test_without_a_model_adaptation_is_unavailable(demo):
    give_feedback(demo)

    assert training.update_upcoming_workout(demo.context, 1, 0).code == "unavailable"


# --- schedule ----------------------------------------------------------------


def days(demo):
    return [item.day for item in get_timetable(demo.context).planned_workouts]


def move(index, day):
    return training.WorkoutMove(week_number=1, workout_index=index, new_day=day)


def test_two_workouts_can_swap_days(demo):
    result = training.organize_schedule(
        demo.context, [move(0, "friday"), move(2, "Monday")]
    )

    assert result.ok
    assert days(demo) == ["Friday", "Wednesday", "Monday"]


def test_a_day_clash_changes_nothing(demo):
    assert (
        training.organize_schedule(demo.context, [move(0, "Friday")]).code
        == "day_conflict"
    )
    assert days(demo) == ["Monday", "Wednesday", "Friday"]


def test_an_unavailable_day_is_refused(demo):
    assert (
        training.organize_schedule(demo.context, [move(0, "Sunday")]).code
        == "day_not_available"
    )


def test_schedule_changes_stay_within_the_student_plan(demo, other_student):
    other = make_context(demo.repository, other_student)

    result = training.organize_schedule(
        other, [move(0, "Saturday")], plan_id=demo.plan_id
    )

    assert result.code == "plan_not_found"


def add_saturday(demo):
    """Make Saturday available too, so a week has a day left to catch up on."""
    profile = demo.repository.get_profile(demo.student_id)
    demo.repository.save_profile(
        demo.student_id,
        profile.model_copy(
            update={"available_days": [*profile.available_days, "Saturday"]}
        ),
    )


def test_missed_workouts_go_first_on_the_days_left(demo):
    add_saturday(demo)
    give_feedback(demo, 0, status="not_completed")
    give_feedback(demo, 1, status="not_completed")

    moves = training.catch_up_moves(demo.context, week_number=1)
    described = training.describe_moves(demo.context, moves)
    result = training.organize_schedule(demo.context, catch_up_week=1)

    # Monday's and Wednesday's missed workouts take Friday and Saturday;
    # Friday's workout no longer fits and takes a day that has passed.
    assert [(m.workout_index, m.new_day) for m in moves] == [
        (0, "Friday"),
        (1, "Saturday"),
        (2, "Monday"),
    ]
    assert result.ok and days(demo) == ["Friday", "Saturday", "Monday"]
    assert "Monday (missed) → Friday" in described
    assert "skipped this week" in described


def test_catch_up_starts_no_earlier_than_from_day(demo):
    add_saturday(demo)
    give_feedback(demo, 0, status="not_completed")

    moves = training.catch_up_moves(demo.context, 1, from_day="Saturday")

    assert [(m.workout_index, m.new_day) for m in moves] == [(0, "Saturday")]


def test_catch_up_needs_a_logged_missed_workout_and_a_day_left(demo):
    assert training.catch_up_moves(demo.context, 1).code == "nothing_to_catch_up"
    give_feedback(demo, 2, status="not_completed")
    assert training.catch_up_moves(demo.context, 1).code == "no_day_left"


def test_an_archived_plan_is_not_the_one_changed(demo):
    record = demo.repository.get_plan(demo.student_id, demo.plan_id)
    newer = demo.repository.save_plan(
        demo.student_id,
        PlanRequest.model_validate(record["request"]),
        PlanResult(
            status="completed",
            message="New",
            plan=TrainingPlan.model_validate(record["plan"]),
        ),
    )

    result = training.organize_schedule(
        demo.context, [move(0, "Friday"), move(2, "Monday")]
    )

    assert result.plan_id == newer
