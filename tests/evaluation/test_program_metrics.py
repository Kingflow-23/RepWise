import pytest

from src.evaluation.metrics.program import (
    check_program, duration_overrun, estimate_session_minutes, reps_upper, tempo_seconds,
)

PROFILE = {"days_per_week": 3, "max_session_min": 45, "equipment": ["dumbbells", "bench"]}


def exercise(name="Développé haltères", sets=3, reps="8-12", rest_s=90, equipment=("dumbbells", "bench"), **kw):
    return {"name": name, "sets": sets, "reps": reps, "rest_s": rest_s,
            "effort": {"scale": "RIR", "value": "1-2"}, "equipment": list(equipment), **kw}


def plan(*sessions):
    return {"sessions": list(sessions)}


@pytest.mark.parametrize("reps, expected", [("8-12", 12), ({"min": 6, "max": 8}, 8), (10, 10), ("AMRAP", None)])
def test_reps_upper(reps, expected):
    assert reps_upper(reps) == expected


def test_tempo_seconds():
    assert tempo_seconds("3-0-1-0") == 4
    assert tempo_seconds("2-1-X-0") == 4
    assert tempo_seconds("lent") is None


def test_duration_formula_by_hand():
    # warm-up 10 min
    # ex1: 3 sets x 12 reps x 4 s = 144 s + 2 rests x 90 s = 180 s -> 324 s
    # ex2: 4 sets x 10 reps x 5 s (tempo 3-1-1-0) = 200 s + 3 x 60 s = 180 s -> 380 s
    # one transition 60 s
    session = {"warmup_min": 10, "exercises": [
        exercise(), exercise("Rowing", sets=4, reps=10, rest_s=60, tempo="3-1-1-0")]}
    est = estimate_session_minutes(session)
    assert est.minutes == pytest.approx(10 + (324 + 380 + 60) / 60)


def test_missing_warmup_is_an_explicit_assumption():
    est = estimate_session_minutes({"exercises": [exercise()]})
    assert any("warm-up" in a for a in est.assumptions)


def test_duration_overrun_does_not_penalise_short_sessions():
    assert duration_overrun(30, 45) == 0
    assert duration_overrun(54, 45) == pytest.approx(0.2)


def test_valid_plan_has_no_violation():
    report = check_program(plan({"name": "A", "exercises": [exercise()]}), PROFILE)
    assert report.zero_violations
    assert report.parameter_completeness.value == 1


def test_too_many_sessions_and_missing_equipment():
    s = {"name": "A", "exercises": [exercise("Squat barre", equipment=["barbell", "rack"])]}
    report = check_program(plan(s, s, s, s), PROFILE)
    names = {v.name.split(":")[0] for v in report.violations}
    assert {"available_days", "equipment"} <= names
    assert report.hard_constraint_adherence.value < 1


def test_overlong_session_violates_duration():
    big = {"name": "Long", "exercises": [exercise(sets=5, rest_s=180) for _ in range(6)]}
    report = check_program(plan(big), PROFILE)
    assert any(v.name.startswith("duration") for v in report.violations)
    assert report.overruns[0] > 0


def test_placeholders_do_not_count_as_parameters():
    ex = exercise(rest_s=None, effort="TBD")
    report = check_program(plan({"name": "A", "exercises": [ex]}), PROFILE)
    assert report.parameter_completeness.fmt() == "0.600 (3/5)"


def test_excluded_exercises_are_matched_without_accents():
    s = {"name": "A", "exercises": [exercise("Soulevé de terre", equipment=["dumbbells"])]}
    report = check_program(plan(s), PROFILE, forbidden_exercises=["souleve de terre"])
    assert any(v.name == "excluded_exercises" for v in report.violations)


def test_invalid_sets_and_unknown_effort_scale():
    ex = exercise(sets=0, effort={"scale": "percent", "value": 80})
    report = check_program(plan({"name": "A", "exercises": [ex]}), PROFILE)
    names = {v.name.split(":")[0] for v in report.violations}
    assert {"valid_sets", "effort_scale"} <= names


def test_preferences():
    profile = {**PROFILE, "preferences": {"preferred_exercises": ["développé"], "disliked_exercises": ["curl"]}}
    report = check_program(plan({"name": "A", "exercises": [exercise(), exercise("Curl marteau")]}), profile)
    assert report.preference_satisfaction.fmt() == "0.500 (1/2)"
