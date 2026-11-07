"""Calendar tools that fit the training plan into the student's week.

RepWise has two calendars, and this module never confuses them:

- the **internal** calendar: the weekday of each workout in the plan, plus the
  ``calendar_events`` table recording which workouts RepWise booked and when;
- the **external** calendar: the student's Google Calendar, used for free/busy
  times and for creating events, only after the student connects it
  (``python -m scripts.connect_google_calendar``).

Without an external calendar, ``get_calendar_busy``, ``propose_training_slots``
and ``add_workouts_to_calendar`` return ``calendar_not_connected``: RepWise does
not invent free times. ``organize_schedule`` (in ``training.py``) still works,
because it only changes the internal plan.

Slot finding is three small pure functions, so a wrong slot can be traced:

    plan weekday --> real date              (date_of_workout)
    real date + busy times --> free starts  (free_start_times)
    free starts --> a few options           (spread_options)

All times are timezone-aware. Dates are computed in the student's time zone
and converted by the provider, so daylight-saving changes are handled.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, time, timedelta
from typing import Literal, Protocol
from zoneinfo import ZoneInfo

from pydantic import Field

from src.schemas.models import WEEKDAYS, Model, TrainingPlan, Workout, weekday_number
from src.tools.context import ToolContext
from src.tools.profile import find_workout, resolve_plan
from src.tools.results import (
    STUDENT_NOT_FOUND,
    ToolFailure,
    failure,
    report_storage_errors,
)

logger = logging.getLogger(__name__)

# Candidate start times are this far apart, and each workout gets this many options.
SLOT_STEP_MINUTES = 30
OPTIONS_PER_WORKOUT = 3
NOT_CONNECTED_MESSAGE = (
    "No external calendar is connected, so RepWise cannot see when the student is "
    "busy. Connect Google Calendar, or move workouts within the plan instead."
)


class CalendarError(Exception):
    """The calendar could not answer, so free times are unknown."""


class BusyPeriod(Model):
    """A time when the student is busy. Calendars give no titles or details."""

    start: datetime
    end: datetime


class CalendarProvider(Protocol):
    """What the tools need from an external calendar."""

    def busy_periods(self, start: datetime, end: datetime) -> list[BusyPeriod]:
        """Return busy times in the range. Raise ``CalendarError`` if unknown."""

    def create_event(
        self, key: str, title: str, description: str, start: datetime, end: datetime
    ) -> str:
        """Create an event and return its ID. The same ``key`` must not duplicate."""


class FakeCalendar:
    """An in-memory calendar for tests and demos. Nothing leaves the process."""

    def __init__(self, busy: list[BusyPeriod] | None = None):
        """Start with some busy periods.

        Args:
            busy: Periods the student is busy, or ``None`` for a free calendar.
        """
        self.busy = list(busy or [])
        self.events: dict[str, dict] = {}

    def busy_periods(self, start: datetime, end: datetime) -> list[BusyPeriod]:
        """Return the busy periods that overlap ``start``-``end``.

        Args:
            start: Start of the range.
            end: End of the range.

        Returns:
            Overlapping periods.
        """
        return [p for p in self.busy if p.start < end and p.end > start]

    def create_event(
        self, key: str, title: str, description: str, start: datetime, end: datetime
    ) -> str:
        """Create an event once per ``key`` and mark its time as busy.

        Args:
            key: Idempotency key; a second call with it changes nothing.
            title: Event title.
            description: Event description.
            start: Event start.
            end: Event end.

        Returns:
            The event ID, equal to ``key``.
        """
        if key not in self.events:
            self.events[key] = {
                "title": title,
                "description": description,
                "start": start,
                "end": end,
            }
            self.busy.append(BusyPeriod(start=start, end=end))
        return key


class Slot(Model):
    """A free time that fits one workout.

    ``label`` is written by code ("Monday 2026-11-09 07:00", in the student's
    time zone), so a model can copy it instead of converting a timestamp itself.
    """

    start: datetime
    end: datetime
    day: str
    label: str


class WorkoutSlots(Model):
    """The options for one workout."""

    plan_id: str
    week_number: int
    workout_index: int
    planned_day: str
    focus: str
    minutes: int
    options: list[Slot]
    moved_to_another_day: bool = False
    note: str | None = None


class BusyResult(Model):
    """Output of ``get_calendar_busy``."""

    ok: Literal[True] = True
    timezone: str
    start: datetime
    end: datetime
    busy: list[BusyPeriod]


class SlotsResult(Model):
    """Output of ``propose_training_slots``. Nothing has been booked."""

    ok: Literal[True] = True
    timezone: str
    week_start: date
    workouts: list[WorkoutSlots]


class Booking(Model):
    """Book one workout at a time the student approved."""

    week_number: int = Field(ge=1)
    workout_index: int = Field(ge=0)
    start: datetime


class BookingOutcome(Model):
    """What happened to one booking."""

    week_number: int
    workout_index: int
    status: Literal["booked", "already_booked", "conflict", "failed"]
    event_id: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    plan_day_differs: bool = False
    detail: str | None = None


class BookingResult(Model):
    """Output of ``add_workouts_to_calendar``. Check each outcome's status."""

    ok: Literal[True] = True
    plan_id: str
    outcomes: list[BookingOutcome]


def monday_of(day: date) -> date:
    """Return the Monday of the week containing ``day``.

    Args:
        day: Any date.

    Returns:
        The Monday on or before ``day``.
    """
    return day - timedelta(days=day.weekday())


def default_week_start(plan_days: list[str], today: date) -> date:
    """Choose which week to schedule: this one, unless a workout day has passed.

    Args:
        plan_days: Weekday names used by the plan.
        today: Today's date in the student's time zone.

    Returns:
        The Monday of this week, or of next week when a planned day is past.
    """
    known = [n for n in (weekday_number(day) for day in plan_days) if n is not None]
    this_monday = monday_of(today)
    if known and min(known) < today.weekday():
        return this_monday + timedelta(days=7)
    return this_monday


def date_of_workout(week_start: date, week_number: int, day: str) -> date | None:
    """Turn "week 1, Monday" into a real date.

    Args:
        week_start: Monday of the plan's first week.
        week_number: Plan week, starting at 1.
        day: English weekday name.

    Returns:
        The date, or ``None`` for an unknown day name.
    """
    number = weekday_number(day)
    if number is None:
        return None
    return week_start + timedelta(days=7 * (week_number - 1) + number)


def free_start_times(
    day: date,
    minutes: int,
    busy: list[BusyPeriod],
    zone: ZoneInfo,
    now: datetime,
    earliest: time,
    latest: time,
) -> list[datetime]:
    """List start times on one day when the whole workout is free and in the future.

    Args:
        day: The date to check.
        minutes: Workout length.
        busy: Busy periods that may overlap the day.
        zone: Student's time zone.
        now: Current time; past starts are skipped.
        earliest: First allowed start time.
        latest: Latest allowed end time.

    Returns:
        Candidate starts, every ``SLOT_STEP_MINUTES``.
    """
    start = datetime.combine(day, earliest, zone)
    last_end = datetime.combine(day, latest, zone)
    length, step = timedelta(minutes=minutes), timedelta(minutes=SLOT_STEP_MINUTES)
    starts = []
    while start + length <= last_end:
        overlaps = any(start < p.end and start + length > p.start for p in busy)
        if start > now and not overlaps:
            starts.append(start)
        start += step
    return starts


def spread_options(starts: list[datetime]) -> list[datetime]:
    """Choose the first, middle and last times, so the options really differ.

    Args:
        starts: Candidate start times, in order.

    Returns:
        At most ``OPTIONS_PER_WORKOUT`` starts.
    """
    if len(starts) <= OPTIONS_PER_WORKOUT:
        return starts
    return [starts[0], starts[len(starts) // 2], starts[-1]]


def event_key(
    student_id: str, plan_id: str, week_number: int, workout_index: int
) -> str:
    """Build a stable event key, so a retried booking cannot create a duplicate.

    Args:
        student_id: Account the event belongs to.
        plan_id: Plan of the workout.
        week_number: Week of the workout.
        workout_index: Position of the workout.

    Returns:
        A hexadecimal SHA-256 digest (valid as a Google event ID).
    """
    text = f"{student_id}:{plan_id}:{week_number}:{workout_index}"
    return hashlib.sha256(text.encode()).hexdigest()


def event_description(workout: Workout) -> str:
    """Describe a workout for the calendar. This text is sent to the provider.

    Args:
        workout: The workout being booked.

    Returns:
        Focus and exercises, one per line.
    """
    lines = [
        f"{item.name}: {item.sets} x {item.repetitions}" for item in workout.exercises
    ]
    return "\n".join([workout.focus, *lines, "", "Added by RepWise."])


def get_calendar_busy(
    context: ToolContext, start_date: date | None = None, days: int = 7
) -> BusyResult | ToolFailure:
    """Read when the student is busy. Times only, never event titles.

    Args:
        context: Trusted tool context.
        start_date: First day to read, or ``None`` for today.
        days: How many days to read.

    Returns:
        Busy periods, or ``calendar_not_connected`` / ``calendar_unavailable``.
    """
    if context.calendar is None:
        return failure("calendar_not_connected", NOT_CONNECTED_MESSAGE)
    zone = ZoneInfo(context.timezone)
    first_day = start_date or context.clock().astimezone(zone).date()
    start = datetime.combine(first_day, time.min, zone)
    end = start + timedelta(days=days)
    try:
        busy = context.calendar.busy_periods(start, end)
    except CalendarError as error:
        logger.warning("Calendar read failed: %s", error)
        return failure("calendar_unavailable", str(error))
    return BusyResult(timezone=context.timezone, start=start, end=end, busy=busy)


@report_storage_errors
def propose_training_slots(
    context: ToolContext,
    week_start: date | None = None,
    earliest: time = time(7, 0),
    latest: time = time(21, 0),
) -> SlotsResult | ToolFailure:
    """Find free times for the current plan's unbooked, unlogged workouts.

    A workout stays on its planned day when possible. Otherwise its options are
    on another available day that no other workout uses that week. The first
    option of each workout is held while the next workout is placed, so two
    workouts are never offered the same time. Nothing is booked.

    Args:
        context: Trusted tool context.
        week_start: A date in the first week to schedule, or ``None`` to choose
            this week (or next week if a planned day already passed).
        earliest: Earliest start time.
        latest: Latest end time.

    Returns:
        Options per workout, or a failure (``calendar_not_connected``,
        ``calendar_unavailable``, ``student_not_found``, ``plan_not_found``).
    """
    if context.calendar is None:
        return failure("calendar_not_connected", NOT_CONNECTED_MESSAGE)
    profile = context.repository.get_profile(context.student_id)
    if profile is None:
        return failure(STUDENT_NOT_FOUND, "Save a profile before asking RepWise.")
    record = resolve_plan(context, None)
    if isinstance(record, ToolFailure):
        return record
    plan = TrainingPlan.model_validate(record["plan"])

    zone = ZoneInfo(context.timezone)
    local_now = context.clock().astimezone(zone)
    all_days = [workout.day for week in plan.weeks for workout in week.workouts]
    first_monday = monday_of(
        week_start or default_week_start(all_days, local_now.date())
    )
    range_start = datetime.combine(first_monday, time.min, zone)
    range_end = range_start + timedelta(days=7 * len(plan.weeks))
    try:
        busy = list(context.calendar.busy_periods(range_start, range_end))
    except CalendarError as error:
        logger.warning("Calendar read failed: %s", error)
        return failure("calendar_unavailable", str(error))

    done = {(f["week_number"], f["workout_index"]) for f in record["workout_feedback"]}
    done |= {
        (event["week_number"], event["workout_index"])
        for event in context.repository.list_calendar_events(
            context.student_id, record["id"]
        )
    }

    results = []
    for week in plan.weeks:
        days_taken = {workout.day.casefold() for workout in week.workouts}
        for index, workout in enumerate(week.workouts):
            if (week.number, index) in done:
                continue
            minutes = workout.estimated_minutes
            planned_date = date_of_workout(first_monday, week.number, workout.day)
            starts = (
                free_start_times(
                    planned_date, minutes, busy, zone, local_now, earliest, latest
                )
                if planned_date
                else []
            )
            moved = False
            if not starts:
                for other_day in profile.available_days:
                    other_date = date_of_workout(first_monday, week.number, other_day)
                    if other_day.casefold() in days_taken or other_date is None:
                        continue
                    starts = free_start_times(
                        other_date, minutes, busy, zone, local_now, earliest, latest
                    )
                    if starts:
                        moved = True
                        days_taken.add(other_day.casefold())
                        break
            options = [
                Slot(
                    start=start,
                    end=start + timedelta(minutes=minutes),
                    day=WEEKDAYS[start.weekday()],
                    label=f"{WEEKDAYS[start.weekday()]} {start:%Y-%m-%d %H:%M}",
                )
                for start in spread_options(starts)
            ]
            if options:
                busy.append(BusyPeriod(start=options[0].start, end=options[0].end))
            results.append(
                WorkoutSlots(
                    plan_id=record["id"],
                    week_number=week.number,
                    workout_index=index,
                    planned_day=workout.day,
                    focus=workout.focus,
                    minutes=minutes,
                    options=options,
                    moved_to_another_day=moved,
                    note=(
                        None if options else "No free time that week fits this workout."
                    ),
                )
            )
    return SlotsResult(
        timezone=context.timezone, week_start=first_monday, workouts=results
    )


@report_storage_errors
def add_workouts_to_calendar(
    context: ToolContext, bookings: list[Booking], plan_id: str | None = None
) -> BookingResult | ToolFailure:
    """Add approved times to the external calendar, rechecking each one first.

    Each booking has its own outcome, so one conflict does not hide the others.
    Booking on a different day than planned does not change the plan: the
    outcome sets ``plan_day_differs``, and ``organize_schedule`` can move it.
    This tool always requires approval.

    Args:
        context: Trusted tool context.
        bookings: Workouts and the start times the student chose.
        plan_id: The plan, or ``None`` for the current plan.

    Returns:
        One outcome per booking, or a failure (``calendar_not_connected``,
        ``plan_not_found``, ``invalid_request``).
    """
    if context.calendar is None:
        return failure("calendar_not_connected", NOT_CONNECTED_MESSAGE)
    record = resolve_plan(context, plan_id)
    if isinstance(record, ToolFailure):
        return record
    if not bookings:
        return failure("invalid_request", "Give at least one booking.")
    plan = TrainingPlan.model_validate(record["plan"])
    existing = {
        (item["week_number"], item["workout_index"]): item
        for item in context.repository.list_calendar_events(
            context.student_id, record["id"]
        )
    }
    zone = ZoneInfo(context.timezone)
    now = context.clock()

    def book(booking: Booking) -> BookingOutcome:
        """Check and create one event, returning its outcome."""
        position = {
            "week_number": booking.week_number,
            "workout_index": booking.workout_index,
        }
        workout = find_workout(plan, booking.week_number, booking.workout_index)
        if workout is None:
            return BookingOutcome(
                **position, status="failed", detail="Workout not found."
            )
        linked = existing.get((booking.week_number, booking.workout_index))
        if linked:
            return BookingOutcome(
                **position,
                status="already_booked",
                event_id=linked["event_id"],
                start=linked["start_at"],
                end=linked["end_at"],
            )
        # A time without a zone is read in the student's zone, not UTC.
        start = (
            booking.start
            if booking.start.tzinfo
            else booking.start.replace(tzinfo=zone)
        )
        end = start + timedelta(minutes=workout.estimated_minutes)
        if start <= now:
            return BookingOutcome(
                **position, status="failed", detail="This time has passed."
            )
        try:
            if context.calendar.busy_periods(start, end):
                return BookingOutcome(
                    **position,
                    status="conflict",
                    start=start,
                    end=end,
                    detail="The calendar is busy at this time.",
                )
            event_id = context.calendar.create_event(
                event_key(
                    context.student_id,
                    record["id"],
                    booking.week_number,
                    booking.workout_index,
                ),
                f"RepWise: {workout.focus}",
                event_description(workout),
                start,
                end,
            )
        except CalendarError as error:
            logger.warning("Calendar write failed: %s", error)
            return BookingOutcome(**position, status="failed", detail=str(error))
        context.repository.save_calendar_event(
            context.student_id,
            record["id"],
            booking.week_number,
            booking.workout_index,
            event_id,
            start.isoformat(),
            end.isoformat(),
        )
        return BookingOutcome(
            **position,
            status="booked",
            event_id=event_id,
            start=start,
            end=end,
            plan_day_differs=start.astimezone(zone).weekday()
            != weekday_number(workout.day),
        )

    parsed = [Booking.model_validate(item) for item in bookings]
    return BookingResult(plan_id=record["id"], outcomes=[book(item) for item in parsed])
