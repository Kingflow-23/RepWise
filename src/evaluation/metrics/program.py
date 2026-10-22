"""Deterministic checks on a generated training plan.

Expected plan shape (the contract shared with the generation pipeline):

    {"sessions": [{"name": "Full body A", "warmup_min": 10,
                   "exercises": [{"name": "Squat", "equipment": ["barbell", "rack"],
                                  "sets": 3, "reps": "8-10", "rest_s": 120,
                                  "effort": {"scale": "RIR", "value": "1-2"},
                                  "tempo": "3-0-1-0"}]}]}

Duration follows the formula of the project note (section 4):
    T_session = T_warmup + sum[sets * exec_per_set + (sets - 1) * rest] + T_transitions
with the upper end of repetition ranges. Every default used is recorded as an assumption.
"""

import re
from dataclasses import dataclass, field

from .common import Ratio
from .grounding import normalise

DEFAULT_WARMUP_MIN = 10.0
DEFAULT_SEC_PER_REP = 4.0
DEFAULT_TRANSITION_S = 60.0
REQUIRED_EXERCISE_FIELDS = ("name", "sets", "reps", "rest_s", "effort")
PLACEHOLDERS = {"", "tbd", "todo", "?", "a definir", "n a", "na", "none", "null", "xx"}
MAX_SETS_PER_EXERCISE = 10


def _is_placeholder(value) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return not value or all(_is_placeholder(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return not value
    if isinstance(value, str):
        return normalise(value) in PLACEHOLDERS
    return False


def reps_upper(reps) -> float | None:
    """'8-12' -> 12, {'min': 8, 'max': 12} -> 12, 10 -> 10, 'AMRAP' -> None."""
    if isinstance(reps, (int, float)):
        return float(reps)
    if isinstance(reps, dict):
        top = reps.get("max", reps.get("min"))
        return float(top) if isinstance(top, (int, float)) else None
    if isinstance(reps, str):
        numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", reps)]
        return max(numbers) if numbers else None
    return None


def tempo_seconds(tempo) -> float | None:
    """'3-0-1-0' -> 4 seconds per rep. 'X' (explosive) counts as 1 second."""
    if not isinstance(tempo, str):
        return None
    parts = re.split(r"[-/ ]", tempo.strip())
    if len(parts) < 3:
        return None
    total = 0.0
    for p in parts:
        if p.upper() == "X":
            total += 1
        elif p.isdigit():
            total += int(p)
        else:
            return None
    return total


@dataclass
class DurationEstimate:
    minutes: float
    assumptions: list[str] = field(default_factory=list)


def estimate_session_minutes(
    session: dict,
    sec_per_rep: float = DEFAULT_SEC_PER_REP,
    transition_s: float = DEFAULT_TRANSITION_S,
    default_warmup_min: float = DEFAULT_WARMUP_MIN,
) -> DurationEstimate:
    assumptions = []
    warmup = session.get("warmup_min")
    if not isinstance(warmup, (int, float)):
        warmup = default_warmup_min
        assumptions.append(f"warm-up assumed {default_warmup_min:g} min")
    exercises = session.get("exercises") or []
    total_s = 0.0
    for ex in exercises:
        sets = ex.get("sets") if isinstance(ex.get("sets"), (int, float)) else 0
        top = reps_upper(ex.get("reps"))
        if top is None:
            top = 12
            assumptions.append(f"{ex.get('name', '?')}: reps unknown, 12 assumed")
        per_rep = tempo_seconds(ex.get("tempo"))
        if per_rep is None:
            per_rep = sec_per_rep
        rest = ex.get("rest_s") if isinstance(ex.get("rest_s"), (int, float)) else 0
        total_s += sets * top * per_rep + max(sets - 1, 0) * rest
    if exercises:
        total_s += (len(exercises) - 1) * transition_s
        assumptions.append(f"{sec_per_rep:g} s/rep when no tempo, {transition_s:g} s per exercise transition")
    return DurationEstimate(warmup + total_s / 60, assumptions)


def duration_overrun(estimated_min: float, available_min: float) -> float:
    """max(0, estimated - available) / available. Shorter sessions are not penalised."""
    return max(0.0, estimated_min - available_min) / available_min


def _names_match(a: str, b: str) -> bool:
    a, b = normalise(a), normalise(b)
    return bool(a) and bool(b) and (a in b or b in a)


@dataclass
class ConstraintResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ProgramReport:
    constraints: list[ConstraintResult] = field(default_factory=list)
    preferences: list[ConstraintResult] = field(default_factory=list)
    required_fields: int = 0
    populated_fields: int = 0
    session_minutes: list[float] = field(default_factory=list)
    overruns: list[float] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    @property
    def hard_constraint_adherence(self) -> Ratio:
        return Ratio(sum(c.passed for c in self.constraints), len(self.constraints))

    @property
    def zero_violations(self) -> bool:
        return all(c.passed for c in self.constraints)

    @property
    def preference_satisfaction(self) -> Ratio:
        return Ratio(sum(p.passed for p in self.preferences), len(self.preferences))

    @property
    def parameter_completeness(self) -> Ratio:
        return Ratio(self.populated_fields, self.required_fields)

    @property
    def violations(self) -> list[ConstraintResult]:
        return [c for c in self.constraints if not c.passed]


def check_program(plan: dict, profile: dict, forbidden_exercises: list[str] | None = None) -> ProgramReport:
    """Validate one plan against a user profile.

    profile keys used: days_per_week, max_session_min, equipment (list),
    excluded_exercises (list), preferences.{preferred_exercises, disliked_exercises}.
    """
    report = ProgramReport()
    sessions = plan.get("sessions") or []
    exercises = [ex for s in sessions for ex in (s.get("exercises") or [])]

    days = profile.get("days_per_week")
    if days is not None:
        n = len(sessions)
        report.constraints.append(ConstraintResult("available_days", n <= days, f"{n} sessions for {days} days"))

    max_min = profile.get("max_session_min")
    for s in sessions:
        est = estimate_session_minutes(s)
        report.session_minutes.append(est.minutes)
        report.assumptions.extend(a for a in est.assumptions if a not in report.assumptions)
        if max_min:
            report.overruns.append(duration_overrun(est.minutes, max_min))
            report.constraints.append(ConstraintResult(
                f"duration:{s.get('name', '?')}", est.minutes <= max_min,
                f"{est.minutes:.1f} min estimated for {max_min} min available"))

    available = profile.get("equipment")
    if available is not None:
        have = {normalise(e) for e in available}
        for ex in exercises:
            need = {normalise(e) for e in (ex.get("equipment") or [])}
            missing = sorted(need - have)
            report.constraints.append(ConstraintResult(
                f"equipment:{ex.get('name', '?')}", not missing,
                f"missing {', '.join(missing)}" if missing else ""))

    banned = list(profile.get("excluded_exercises") or []) + list(forbidden_exercises or [])
    if banned:
        used = [ex.get("name", "") for ex in exercises if any(_names_match(ex.get("name", ""), b) for b in banned)]
        report.constraints.append(ConstraintResult("excluded_exercises", not used, ", ".join(used)))

    for ex in exercises:
        sets = ex.get("sets")
        if sets is not None:
            ok = isinstance(sets, int) and 0 < sets <= MAX_SETS_PER_EXERCISE
            report.constraints.append(ConstraintResult(f"valid_sets:{ex.get('name', '?')}", ok, f"sets={sets}"))
        effort = ex.get("effort")
        if isinstance(effort, dict) and effort.get("scale") not in (None, "RIR", "RPE"):
            report.constraints.append(ConstraintResult(
                f"effort_scale:{ex.get('name', '?')}", False, f"unknown scale {effort.get('scale')}"))
        for f in REQUIRED_EXERCISE_FIELDS:
            report.required_fields += 1
            report.populated_fields += not _is_placeholder(ex.get(f))

    prefs = profile.get("preferences") or {}
    names = [ex.get("name", "") for ex in exercises]
    for liked in prefs.get("preferred_exercises") or []:
        report.preferences.append(ConstraintResult(f"includes:{liked}", any(_names_match(n, liked) for n in names)))
    for disliked in prefs.get("disliked_exercises") or []:
        report.preferences.append(ConstraintResult(f"avoids:{disliked}", not any(_names_match(n, disliked) for n in names)))
    return report
