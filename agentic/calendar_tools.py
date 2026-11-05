"""Tools that fit a training plan into the student's calendar.

The steps are small pure functions, so a wrong slot can be traced to one of them:

    plan weekday --> real date (``date_of_workout``)
    real date + busy times --> free start times (``free_start_times``)
    free start times --> a few options (``spread_options``)

The tools talk to any :class:`CalendarProvider`: ``GoogleCalendar`` in
``agentic/google_calendar.py``, or :class:`FakeCalendar` for tests and demos.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, time, timedelta
from typing import Literal, Protocol
from zoneinfo import ZoneInfo

from pydantic import Field

from agentic import tools
from agentic.tools import report_storage_errors
from agentic.training_tools import Failure, find_workout, open_agent_repository
from src.generation.adaptation import WEEKDAY_ORDER
from src.schemas.models import Model, TrainingPlan, Workout

logger = logging.getLogger(__name__)

SLOT_STEP_MINUTES = 30
OPTIONS_PER_WORKOUT = 3
# Not strftime("%A"), which follows the computer's language.
DAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
NOT_CONNECTED = Failure(
    code="calendar_not_connected",
    error="The student has not connected a calendar.",
)


class CalendarError(Exception):
    """The calendar could not answer, so free times are unknown."""


class BusyPeriod(Model):
    """A time when the student is busy. Calendars give no titles or details."""

    start: datetime
    end: datetime


class CalendarProvider(Protocol):
    """What the tools need from a calendar."""

    def busy_periods(self, start: datetime, end: datetime) -> list[BusyPeriod]:
        """Return busy times in the range. Raise ``CalendarError`` if unknown."""

    def create_event(
        self, key: str, title: str, description: str, start: datetime, end: datetime
    ) -> str:
        """Create an event and return its ID. The same ``key`` must not duplicate."""


class FakeCalendar:
    """An in-memory calendar for tests and the demo. It never leaves the process."""

    def __init__(self, busy: list[BusyPeriod] | None = None):
        self.busy = list(busy or [])
        self.events: dict[str, dict] = {}

    def busy_periods(self, start: datetime, end: datetime) -> list[BusyPeriod]:
        return [
            period for period in self.busy if period.start < end and period.end > start
        ]

    def create_event(
        self, key: str, title: str, description: str, start: datetime, end: datetime
    ) -> str:
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
    """A free time that fits one workout."""

    start: datetime
    end: datetime
    day: str


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


def weekday_number(day: str) -> int | None:
    """Return 0 for Monday through 6 for Sunday, or None for an unknown name."""
    return WEEKDAY_ORDER.get(day.strip().casefold())


def monday_of(day: date) -> date:
    return day - timedelta(days=day.weekday())


def default_week_start(plan_days: list[str], today: date) -> date:
    """Use this week unless a workout's day has already passed, then use next week."""
    numbers = [weekday_number(day) for day in plan_days]
    known = [number for number in numbers if number is not None]
    this_monday = monday_of(today)
    if known and min(known) < today.weekday():
        return this_monday + timedelta(days=7)
    return this_monday


def date_of_workout(week_start: date, week_number: int, day: str) -> date | None:
    """Turn "week 1, Monday" into a real date, or None for an unknown day name."""
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
    """List start times on one day when the whole workout is free and in the future."""
    start = datetime.combine(day, earliest, zone)
    last_end = datetime.combine(day, latest, zone)
    length, step = timedelta(minutes=minutes), timedelta(minutes=SLOT_STEP_MINUTES)
    starts = []
    while start + length <= last_end:
        overlaps = any(
            start < period.end and start + length > period.start for period in busy
        )
        if start > now and not overlaps:
            starts.append(start)
        start += step
    return starts


def spread_options(starts: list[datetime]) -> list[datetime]:
    """Choose the first, middle, and last times, so the options really differ."""
    if len(starts) <= OPTIONS_PER_WORKOUT:
        return starts
    return [starts[0], starts[len(starts) // 2], starts[-1]]


def _slots(starts: list[datetime], minutes: int) -> list[Slot]:
    return [
        Slot(
            start=start,
            end=start + timedelta(minutes=minutes),
            day=DAY_NAMES[start.weekday()],
        )
        for start in spread_options(starts)
    ]


def event_key(
    student_id: str, plan_id: str, week_number: int, workout_index: int
) -> str:
    """Build a stable key, so that retrying a booking cannot create a duplicate."""
    text = f"{student_id}:{plan_id}:{week_number}:{workout_index}"
    return hashlib.sha256(text.encode()).hexdigest()


def event_description(workout: Workout) -> str:
    """Describe a workout for the calendar. This text is sent to the provider."""
    lines = [
        f"{item.name}: {item.sets} x {item.repetitions}" for item in workout.exercises
    ]
    return "\n".join([workout.focus, *lines, "", "Added by RepWise."])


def get_calendar_busy(
    calendar: CalendarProvider | None,
    timezone: str,
    now: datetime,
    start_date: date | None = None,
    days: int = 7,
) -> BusyResult | Failure:
    """Read the student's busy times. Nothing is written."""
    if calendar is None:
        return NOT_CONNECTED
    zone = ZoneInfo(timezone)
    first_day = start_date or now.astimezone(zone).date()
    start = datetime.combine(first_day, time.min, zone)
    end = start + timedelta(days=days)
    try:
        busy = calendar.busy_periods(start, end)
    except CalendarError as error:
        logger.warning("Calendar read failed: %s", error)
        return Failure(code="calendar_unavailable", error=str(error))
    return BusyResult(timezone=timezone, start=start, end=end, busy=busy)


@report_storage_errors
def propose_training_slots(
    student_id: str,
    calendar: CalendarProvider | None,
    timezone: str,
    now: datetime,
    week_start: date | None = None,
    earliest: time = time(7, 0),
    latest: time = time(21, 0),
) -> SlotsResult | Failure | tools.ToolFailure:
    """Find free times for the latest plan's unbooked, unlogged workouts.

    A workout stays on its planned day when possible. Otherwise the options are on
    another available day that no other workout uses that week. Nothing is booked.
    """
    if calendar is None:
        return NOT_CONNECTED
    repository = open_agent_repository()
    profile = repository.get_profile(student_id)
    if profile is None:
        return tools.STUDENT_NOT_FOUND
    records = repository.list_plans(student_id, limit=1)
    if not records:
        return tools.PLAN_NOT_FOUND
    record = records[0]
    plan = TrainingPlan.model_validate(record["plan"])

    zone = ZoneInfo(timezone)
    local_now = now.astimezone(zone)
    all_days = [workout.day for week in plan.weeks for workout in week.workouts]
    first_monday = monday_of(
        week_start or default_week_start(all_days, local_now.date())
    )
    range_start = datetime.combine(first_monday, time.min, zone)
    range_end = range_start + timedelta(days=7 * len(plan.weeks))
    try:
        busy = list(calendar.busy_periods(range_start, range_end))
    except CalendarError as error:
        logger.warning("Calendar read failed: %s", error)
        return Failure(code="calendar_unavailable", error=str(error))

    done = {
        (item["week_number"], item["workout_index"])
        for item in record["workout_feedback"]
    }
    done |= {
        (item["week_number"], item["workout_index"])
        for item in repository.list_calendar_events(student_id, record["id"])
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
            options = _slots(starts, minutes)
            if options:
                # Hold the first option, so two workouts are not offered the same time.
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
    return SlotsResult(timezone=timezone, week_start=first_monday, workouts=results)


@report_storage_errors
def add_workouts_to_calendar(
    student_id: str,
    calendar: CalendarProvider | None,
    timezone: str,
    now: datetime,
    plan_id: str,
    bookings: list[Booking],
) -> BookingResult | Failure | tools.ToolFailure:
    """Add approved times to the calendar, rechecking each one first.

    Each booking has its own outcome, so one conflict does not hide the others.
    Booking on a different day than planned does not change the plan: the outcome
    sets ``plan_day_differs``, and ``organize_schedule`` can then move the workout.
    """
    if calendar is None:
        return NOT_CONNECTED
    repository = open_agent_repository()
    saved = repository.get_plan(student_id, plan_id)
    if saved is None:
        return tools.PLAN_NOT_FOUND
    if not bookings:
        return Failure(code="invalid_request", error="Give at least one booking.")
    plan = TrainingPlan.model_validate(saved["plan"])
    existing = {
        (item["week_number"], item["workout_index"]): item
        for item in repository.list_calendar_events(student_id, plan_id)
    }

    def book(booking: Booking) -> BookingOutcome:
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

        zone = ZoneInfo(timezone)
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
            if calendar.busy_periods(start, end):
                return BookingOutcome(
                    **position,
                    status="conflict",
                    start=start,
                    end=end,
                    detail="The calendar is busy at this time.",
                )
            event_id = calendar.create_event(
                event_key(
                    student_id, plan_id, booking.week_number, booking.workout_index
                ),
                f"RepWise: {workout.focus}",
                event_description(workout),
                start,
                end,
            )
        except CalendarError as error:
            logger.warning("Calendar write failed: %s", error)
            return BookingOutcome(**position, status="failed", detail=str(error))

        repository.save_calendar_event(
            student_id,
            plan_id,
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

    return BookingResult(plan_id=plan_id, outcomes=[book(item) for item in bookings])
