"""Slot finding, booking and the Google Calendar adapter. No network."""

import json
from datetime import UTC, date, datetime, time, timedelta

import httpx
import pytest

from src.integrations.google_calendar import GoogleCalendar
from src.tools import calendar as cal
from src.tools import training
from tests.conftest import NEXT_MONDAY, NOW, PARIS, make_context


def busy_on(day, start_hour, end_hour):
    return cal.BusyPeriod(
        start=datetime.combine(day, time(start_hour), PARIS),
        end=datetime.combine(day, time(end_hour), PARIS),
    )


def with_calendar(demo, calendar):
    return make_context(demo.repository, demo.student_id, calendar=calendar)


def booking(index=0, hour=18, day=NEXT_MONDAY):
    return cal.Booking(
        week_number=1,
        workout_index=index,
        start=datetime.combine(day, time(hour), PARIS),
    )


# --- date logic ----------------------------------------------------------------


def test_weekday_becomes_a_date_in_the_right_week():
    assert cal.date_of_workout(NEXT_MONDAY, 1, "Wednesday") == date(2030, 1, 16)
    assert cal.date_of_workout(NEXT_MONDAY, 2, "monday") == date(2030, 1, 21)
    assert cal.date_of_workout(NEXT_MONDAY, 1, "Someday") is None


def test_week_start_waits_until_every_workout_is_still_ahead():
    wednesday = date(2030, 1, 9)
    assert cal.default_week_start(["Monday", "Friday"], wednesday) == NEXT_MONDAY
    assert cal.default_week_start(["Wednesday", "Friday"], wednesday) == date(
        2030, 1, 7
    )


def test_free_times_skip_busy_periods_and_the_past():
    day = date(2030, 1, 9)

    starts = cal.free_start_times(
        day, 60, [busy_on(day, 12, 14)], PARIS, NOW, time(10), time(16)
    )

    assert [item.strftime("%H:%M") for item in starts] == [
        "10:30",
        "11:00",
        "14:00",
        "14:30",
        "15:00",
    ]


def test_free_times_handle_a_daylight_saving_day():
    # Clocks go forward on this Sunday in Paris.
    starts = cal.free_start_times(
        date(2030, 3, 31),
        45,
        [],
        PARIS,
        datetime(2000, 1, 1, tzinfo=UTC),
        time(7),
        time(8),
    )

    assert [item.utcoffset() for item in starts] == [timedelta(hours=2)]


def test_options_are_spread_across_the_day():
    starts = cal.free_start_times(NEXT_MONDAY, 30, [], PARIS, NOW, time(7), time(21))

    assert [item.hour for item in cal.spread_options(starts)] == [7, 14, 20]


# --- propose_training_slots ------------------------------------------------------


def test_slots_stay_on_planned_days_and_avoid_busy_times(demo):
    context = with_calendar(demo, cal.FakeCalendar([busy_on(NEXT_MONDAY, 7, 18)]))

    result = cal.propose_training_slots(context)

    assert result.week_start == NEXT_MONDAY
    monday = result.workouts[0]
    assert (monday.planned_day, monday.moved_to_another_day) == ("Monday", False)
    assert monday.options[0].start.hour == 18
    assert {slot.day for slot in monday.options} == {"Monday"}


def test_each_slot_label_is_the_real_local_weekday_and_date(demo):
    result = cal.propose_training_slots(with_calendar(demo, cal.FakeCalendar()))

    first = result.workouts[0].options[0]
    assert first.label == f"Monday {NEXT_MONDAY.isoformat()} {first.start:%H:%M}"
    for workout in result.workouts:
        for slot in workout.options:
            local = slot.start.astimezone(PARIS)
            assert slot.label == f"{slot.day} {local:%Y-%m-%d %H:%M}"


def test_two_workouts_are_never_offered_the_same_time(demo):
    result = cal.propose_training_slots(with_calendar(demo, cal.FakeCalendar()))

    firsts = [workout.options[0].start for workout in result.workouts]
    assert len(firsts) == len(set(firsts))


def test_a_full_day_moves_to_a_spare_available_day(demo):
    profile = demo.repository.get_profile(demo.student_id)
    demo.repository.save_profile(
        demo.student_id,
        profile.model_copy(
            update={"available_days": [*profile.available_days, "Saturday"]}
        ),
    )
    context = with_calendar(demo, cal.FakeCalendar([busy_on(NEXT_MONDAY, 0, 23)]))

    workout = cal.propose_training_slots(context).workouts[0]

    assert workout.moved_to_another_day
    assert {slot.day for slot in workout.options} == {"Saturday"}


def test_no_free_time_is_said_plainly(demo):
    context = with_calendar(demo, cal.FakeCalendar([busy_on(NEXT_MONDAY, 0, 23)]))

    workout = cal.propose_training_slots(context).workouts[0]

    assert workout.options == [] and "No free time" in workout.note


def test_logged_and_booked_workouts_are_not_offered_again(demo):
    training.save_workout_feedback(demo.context, 1, 0, "completed", "just_right")
    context = with_calendar(demo, cal.FakeCalendar())
    cal.add_workouts_to_calendar(
        context, [booking(1, day=NEXT_MONDAY + timedelta(days=2))]
    )

    result = cal.propose_training_slots(context)

    assert [workout.workout_index for workout in result.workouts] == [2]


# --- add_workouts_to_calendar --------------------------------------------------


def test_booking_creates_one_event_and_a_retry_does_not_duplicate(demo):
    calendar = cal.FakeCalendar()
    context = with_calendar(demo, calendar)

    first = cal.add_workouts_to_calendar(context, [booking()])
    second = cal.add_workouts_to_calendar(context, [booking()])

    assert (first.outcomes[0].status, second.outcomes[0].status) == (
        "booked",
        "already_booked",
    )
    assert len(calendar.events) == 1
    assert "Goblet squat" in next(iter(calendar.events.values()))["description"]


def test_a_conflict_blocks_only_that_booking(demo):
    calendar = cal.FakeCalendar([busy_on(NEXT_MONDAY, 17, 19)])
    wednesday = booking(1, 18, NEXT_MONDAY + timedelta(days=2))

    result = cal.add_workouts_to_calendar(
        with_calendar(demo, calendar), [booking(), wednesday]
    )

    assert [item.status for item in result.outcomes] == ["conflict", "booked"]
    assert len(calendar.events) == 1


def test_a_past_time_is_refused(demo):
    past = booking().model_copy(update={"start": NOW - timedelta(days=1)})

    result = cal.add_workouts_to_calendar(
        with_calendar(demo, cal.FakeCalendar()), [past]
    )

    assert result.outcomes[0].status == "failed"


def test_a_time_without_a_zone_uses_the_students(demo):
    naive = booking().model_copy(
        update={"start": datetime.combine(NEXT_MONDAY, time(18))}
    )

    result = cal.add_workouts_to_calendar(
        with_calendar(demo, cal.FakeCalendar()), [naive]
    )

    assert result.outcomes[0].start.utcoffset() == timedelta(hours=1)


def test_booking_on_another_day_is_flagged(demo):
    tuesday = booking(day=NEXT_MONDAY + timedelta(days=1))

    result = cal.add_workouts_to_calendar(
        with_calendar(demo, cal.FakeCalendar()), [tuesday]
    )

    assert result.outcomes[0].plan_day_differs


def test_moving_a_booked_workout_reports_its_stale_event(demo):
    context = with_calendar(demo, cal.FakeCalendar())
    booked = cal.add_workouts_to_calendar(context, [booking()])

    moves = [
        training.WorkoutMove(week_number=1, workout_index=0, new_day="Friday"),
        training.WorkoutMove(week_number=1, workout_index=2, new_day="Monday"),
    ]
    result = training.organize_schedule(context, moves)

    assert result.calendar_events_now_out_of_date == [booked.outcomes[0].event_id]


@pytest.mark.parametrize(
    "call",
    [
        lambda context: cal.get_calendar_busy(context),
        lambda context: cal.propose_training_slots(context),
        lambda context: cal.add_workouts_to_calendar(context, [booking()]),
    ],
)
def test_no_calendar_is_reported_not_treated_as_free(demo, call):
    assert call(demo.context).code == "calendar_not_connected"


class DownCalendar(cal.FakeCalendar):
    def busy_periods(self, start, end):
        raise cal.CalendarError("Calendar is down")


def test_an_unreadable_calendar_is_never_treated_as_free(demo):
    assert (
        cal.propose_training_slots(with_calendar(demo, DownCalendar())).code
        == "calendar_unavailable"
    )


def test_a_failed_recheck_books_nothing(demo):
    calendar = DownCalendar()

    result = cal.add_workouts_to_calendar(with_calendar(demo, calendar), [booking()])

    assert result.outcomes[0].status == "failed"
    assert calendar.events == {}
    assert demo.repository.list_calendar_events(demo.student_id, demo.plan_id) == []


# --- Google Calendar adapter -----------------------------------------------------

START = datetime(2030, 1, 14, 18, tzinfo=PARIS)
END = START + timedelta(hours=1)


def google(handler):
    return GoogleCalendar(
        lambda: "secret-token", httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_google_busy_periods_are_parsed_and_authorized():
    seen = {}

    def handler(request):
        seen["auth"], seen["url"] = request.headers["Authorization"], str(request.url)
        busy = [{"start": "2030-01-14T17:00:00Z", "end": "2030-01-14T18:00:00Z"}]
        return httpx.Response(200, json={"calendars": {"primary": {"busy": busy}}})

    busy = google(handler).busy_periods(START, END)

    assert seen["auth"] == "Bearer secret-token" and seen["url"].endswith("/freeBusy")
    assert busy[0].start == datetime(2030, 1, 14, 17, tzinfo=UTC)


def test_google_per_calendar_error_is_not_read_as_free():
    def handler(_request):
        return httpx.Response(
            200,
            json={"calendars": {"primary": {"errors": [{"reason": "x"}], "busy": []}}},
        )

    with pytest.raises(cal.CalendarError):
        google(handler).busy_periods(START, END)


@pytest.mark.parametrize("status", [401, 403, 500])
def test_google_http_errors_become_calendar_errors_without_the_token(status):
    with pytest.raises(cal.CalendarError) as error:
        google(lambda _request: httpx.Response(status)).busy_periods(START, END)

    assert "secret-token" not in str(error.value)


def test_google_event_uses_the_stable_key_and_a_duplicate_counts_as_created():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": seen["body"]["id"]})

    assert google(handler).create_event("abc123", "t", "d", START, END) == "abc123"
    assert seen["body"]["start"]["dateTime"] == START.isoformat()
    assert (
        google(lambda _request: httpx.Response(409)).create_event(
            "abc123", "t", "d", START, END
        )
        == "abc123"
    )
