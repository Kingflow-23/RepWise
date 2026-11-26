"""Slot finding, booking and the Google Calendar adapter. No network."""

import json
from datetime import UTC, date, datetime, time, timedelta

import httpx
import pytest

from src.integrations.google_calendar import GoogleCalendar
from src.schemas.models import PlanRequest, PlanResult, Source, TrainingPlan
from src.tools import calendar as cal
from src.tools import training
from src.tools.profile import TimetableResult, get_timetable
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


def test_week_start_keeps_the_week_when_any_workout_day_remains():
    wednesday = date(2030, 1, 9)
    assert cal.default_week_start(["Monday", "Friday"], wednesday) == date(2030, 1, 7)
    assert cal.default_week_start(["Wednesday", "Friday"], wednesday) == date(
        2030, 1, 7
    )
    assert cal.default_week_start(
        ["Monday", "Wednesday", "Friday", "Sunday"], wednesday
    ) == date(2030, 1, 7)


def test_week_start_moves_to_next_week_after_the_last_workout_day():
    saturday = date(2030, 1, 12)
    assert cal.default_week_start(["Monday", "Wednesday", "Friday"], saturday) == date(
        2030, 1, 14
    )


def test_free_times_skip_busy_periods_and_the_past():
    day = date(2030, 1, 9)

    starts = cal.free_start_times(
        day, 60, [busy_on(day, 12, 14)], PARIS, NOW, time(10), time(16)
    )

    assert [item.strftime("%H:%M") for item in starts] == [
        "10:30",
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
        time(7, 45),
    )

    assert [item.utcoffset() for item in starts] == [timedelta(hours=2)]


def test_free_times_stay_inside_daytime_even_if_the_requested_window_is_overnight():
    starts = cal.free_start_times(
        NEXT_MONDAY, 60, [], PARIS, NOW, time(0), time(23, 59)
    )

    assert starts
    assert all(item.time() >= time(6) for item in starts)
    assert all((item + timedelta(minutes=60)).time() <= time(22) for item in starts)


def test_options_prioritize_practical_daytime_starts():
    starts = cal.free_start_times(NEXT_MONDAY, 30, [], PARIS, NOW, time(8), time(20))

    assert [item.strftime("%H:%M") for item in cal.spread_options(starts)] == [
        "10:00",
        "14:00",
        "17:00",
    ]


def test_options_prefer_normal_waking_hours_over_window_edges():
    starts = [
        datetime.combine(NEXT_MONDAY, time(6), PARIS),
        datetime.combine(NEXT_MONDAY, time(18), PARIS),
    ]

    assert cal.spread_options(starts)[0].hour == 18


def test_free_times_leave_a_buffer_around_busy_events():
    day = NEXT_MONDAY
    busy = [
        busy_on(day, 9, 10),
        busy_on(day, 12, 13),
    ]

    starts = cal.free_start_times(day, 60, busy, PARIS, NOW, time(8), time(16))

    assert datetime.combine(day, time(10), PARIS) not in starts
    assert datetime.combine(day, time(11), PARIS) not in starts
    assert datetime.combine(day, time(13, 30), PARIS) in starts


# --- propose_training_slots ------------------------------------------------------


def test_slots_stay_on_planned_days_and_avoid_busy_times(demo):
    context = with_calendar(demo, cal.FakeCalendar([busy_on(NEXT_MONDAY, 7, 18)]))

    result = cal.propose_training_slots(context)

    assert result.week_start == date(2030, 1, 7)
    assert [workout.planned_day for workout in result.workouts] == [
        "Wednesday",
        "Friday",
    ]
    assert all(not workout.moved_to_another_day for workout in result.workouts)
    assert all(
        slot.day in {"Wednesday", "Friday"}
        for workout in result.workouts
        for slot in workout.options
    )


def test_slot_tool_cannot_be_widened_to_night_and_prefers_daytime(demo):
    result = cal.propose_training_slots(
        with_calendar(demo, cal.FakeCalendar()),
        earliest=time(0),
        latest=time(23, 59),
    )

    assert result.workouts[0].options[0].start.strftime("%H:%M") == "10:30"
    for workout in result.workouts:
        for slot in workout.options:
            local_start = slot.start.astimezone(PARIS)
            local_end = slot.end.astimezone(PARIS)
            assert local_start.date() == local_end.date()
            assert local_start.time() >= time(6)
            assert local_end.time() <= time(22)


def test_each_slot_label_is_the_real_local_weekday_and_date(demo):
    result = cal.propose_training_slots(with_calendar(demo, cal.FakeCalendar()))

    first = result.workouts[0].options[0]
    assert (
        first.label == f"Wednesday {date(2030, 1, 9).isoformat()} {first.start:%H:%M}"
    )
    for workout in result.workouts:
        for slot in workout.options:
            local = slot.start.astimezone(PARIS)
            assert slot.label == f"{slot.day} {local:%Y-%m-%d %H:%M}"


def test_two_workouts_are_never_offered_the_same_time(demo):
    result = cal.propose_training_slots(with_calendar(demo, cal.FakeCalendar()))

    firsts = [workout.options[0].start for workout in result.workouts]
    assert len(firsts) == len(set(firsts))


def test_a_full_day_does_not_move_a_workout_without_approval(demo):
    context = with_calendar(demo, cal.FakeCalendar([busy_on(date(2030, 1, 9), 0, 23)]))

    workout = cal.propose_training_slots(context).workouts[0]

    assert workout.options == []
    assert not workout.moved_to_another_day
    assert "planned day" in workout.note


def test_no_free_time_is_said_plainly(demo):
    context = with_calendar(demo, cal.FakeCalendar([busy_on(date(2030, 1, 9), 0, 23)]))

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


def test_booking_result_has_structured_times_and_links_without_source_excerpts(demo):
    record = demo.repository.get_plan(demo.student_id, demo.plan_id)
    plan = TrainingPlan.model_validate(record["plan"])
    weeks = list(plan.weeks)
    workouts = list(weeks[0].workouts)
    exercises = list(workouts[0].exercises)
    exercises[0] = exercises[0].model_copy(update={"source_ids": ["source-1"]})
    workouts[0] = workouts[0].model_copy(update={"exercises": exercises})
    weeks[0] = weeks[0].model_copy(update={"workouts": workouts})
    plan = plan.model_copy(update={"weeks": weeks})
    plan_id = demo.repository.save_plan(
        demo.student_id,
        PlanRequest.model_validate(record["request"]),
        PlanResult(
            status="completed",
            message="Plan with a source.",
            plan=plan,
            sources=[
                Source(
                    source_id="source-1",
                    title="Squat guide",
                    url="https://example.test/squats",
                    excerpt="A source excerpt that must not be copied to the event.",
                )
            ],
        ),
    )

    class LinkedCalendar(cal.FakeCalendar):
        def create_event(self, key, title, description, _start, _end):
            event = super().create_event(key, title, description, _start, _end)
            return event.model_copy(
                update={"event_url": "https://calendar.test/events/abc"}
            )

    context = with_calendar(demo, LinkedCalendar())

    result = cal.add_workouts_to_calendar(
        context, [booking()], plan_id=plan_id
    ).model_dump(mode="json")
    saved = demo.repository.list_calendar_events(demo.student_id, plan_id)
    timetable = get_timetable(context)

    assert result["timezone"] == "Europe/Paris"
    outcome = result["outcomes"][0]
    assert outcome["event_url"] == "https://calendar.test/events/abc"
    assert outcome["start"].endswith("+01:00")
    assert outcome["end"].endswith("+01:00")
    assert outcome["workout"]["exercises"][0]["sources"] == [
        {
            "source_id": "source-1",
            "title": "Squat guide",
            "url": "https://example.test/squats",
        }
    ]
    assert saved[0]["event_url"] == "https://calendar.test/events/abc"
    assert isinstance(timetable, TimetableResult)
    scheduled = next(
        item for item in timetable.planned_workouts if item.plan_id == plan_id
    )
    assert scheduled.booking is not None
    assert scheduled.booking.event_url == "https://calendar.test/events/abc"


def test_a_conflict_blocks_only_that_booking(demo):
    calendar = cal.FakeCalendar([busy_on(NEXT_MONDAY, 17, 19)])
    wednesday = booking(1, 18, NEXT_MONDAY + timedelta(days=2))

    result = cal.add_workouts_to_calendar(
        with_calendar(demo, calendar), [booking(), wednesday]
    )

    assert [item.status for item in result.outcomes] == ["conflict", "booked"]
    assert len(calendar.events) == 1


def test_booking_requires_a_gap_before_and_after_an_existing_event(demo):
    calendar = cal.FakeCalendar([busy_on(NEXT_MONDAY, 9, 10)])
    too_close_before = booking(hour=10)
    far_enough = booking(hour=11)

    too_close = cal.add_workouts_to_calendar(
        with_calendar(demo, calendar), [too_close_before]
    )
    far_result = cal.add_workouts_to_calendar(
        with_calendar(demo, cal.FakeCalendar([busy_on(NEXT_MONDAY, 9, 10)])),
        [far_enough],
    )

    assert too_close.outcomes[0].status == "conflict"
    assert far_result.outcomes[0].status == "booked"


def test_a_past_time_is_refused(demo):
    past = booking().model_copy(update={"start": NOW - timedelta(days=1)})

    result = cal.add_workouts_to_calendar(
        with_calendar(demo, cal.FakeCalendar()), [past]
    )

    assert result.outcomes[0].status == "failed"


@pytest.mark.parametrize("hour", [0, 5, 22])
def test_booking_outside_daytime_window_is_refused(demo, hour):
    outside_window_booking = booking(hour=hour)
    calendar = cal.FakeCalendar()

    result = cal.add_workouts_to_calendar(
        with_calendar(demo, calendar), [outside_window_booking]
    )

    assert result.outcomes[0].status == "failed"
    assert result.outcomes[0].detail is not None
    assert "06:00 and 22:00" in result.outcomes[0].detail
    assert calendar.events == {}


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
        return httpx.Response(
            200,
            json={
                "id": seen["body"]["id"],
                "htmlLink": "https://calendar.google.com/event/abc",
            },
        )

    created = google(handler).create_event("abc123", "t", "d", START, END)
    assert created.event_id == "abc123"
    assert created.event_url == "https://calendar.google.com/event/abc"
    assert seen["body"]["colorId"] == "11"
    assert seen["body"]["start"]["dateTime"] == START.isoformat()
    assert (
        google(lambda _request: httpx.Response(409))
        .create_event("abc123", "t", "d", START, END)
        .event_id
        == "abc123"
    )
