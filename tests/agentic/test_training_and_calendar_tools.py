"""Tests for the history, adaptation, search, and calendar tools. No network or LLM."""

import json
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest
from langchain_core.messages import AIMessage

from agentic import calendar_tools as cal
from agentic import places, tools
from agentic import training_tools as training
from agentic.agent import ask, build_agent, wants_training_slots
from agentic.agent_tools import AgentContext
from agentic.call_tool import SCHEMA_FILE, agent_tool_schemas
from agentic.places import find_nearby_places
from agentic.google_calendar import GoogleCalendar
from agentic.mcp_demo_client import seed_demo_account
from agentic.planning import Plan, PlanStep
from src.retrieval.query_expansion import ScopeRefusalError
from src.schemas.models import (
    PlanRequest,
    PlanResult,
    RetrievedChunk,
    TrainingPlan,
    Workout,
)

PARIS = ZoneInfo("Europe/Paris")
TIMEZONE = "Europe/Paris"
# A Wednesday. The demo plan's Monday has passed, so slots start the next Monday.
NOW = datetime(2030, 1, 9, 10, 0, tzinfo=PARIS)
NEXT_MONDAY = date(2030, 1, 14)
SQUAT = training.PerformanceRecord(
    exercise_name="goblet squat", sets_completed=3, reps_per_set="10", weight_kg=12
)


@pytest.fixture
def account(tmp_path, monkeypatch):
    path = tmp_path / "agent.sqlite3"
    monkeypatch.setenv("REPWISE_DATABASE_PATH", str(path))
    return seed_demo_account(path)


def give_feedback(account, index=0, comment=None):
    tools.save_workout_feedback(
        account[0], account[1], 1, index, "completed", "slightly_too_hard", comment
    )


def busy_on(day, start_hour, end_hour):
    return cal.BusyPeriod(
        start=datetime.combine(day, time(start_hour), PARIS),
        end=datetime.combine(day, time(end_hour), PARIS),
    )


# ---------------------------------------------------------------- performance


def test_performance_uses_the_plan_exercise_name_and_appears_in_history(account):
    result = training.log_workout_performance(account[0], account[1], 1, 0, [SQUAT])

    assert result.records[0].exercise_name == "Goblet squat"
    entry = training.get_workout_history(account[0]).workouts[0]
    assert entry.performance[0].weight_kg == 12


def test_saving_performance_again_replaces_it(account):
    training.log_workout_performance(account[0], account[1], 1, 0, [SQUAT])
    heavier = SQUAT.model_copy(update={"weight_kg": 14})
    training.log_workout_performance(account[0], account[1], 1, 0, [heavier])

    records = training.get_workout_history(account[0]).workouts[0].performance
    assert [record.weight_kg for record in records] == [14]


def test_one_wrong_exercise_saves_nothing(account):
    wrong = SQUAT.model_copy(update={"exercise_name": "Bench press"})

    result = training.log_workout_performance(account[0], account[1], 1, 0, [SQUAT, wrong])

    assert result.code == "exercise_not_found"
    assert "Goblet squat" in result.error
    assert training.get_workout_history(account[0]).workouts == []


# -------------------------------------------------------------------- history


def test_history_summary_is_counted_by_code(account):
    give_feedback(account, 0, "J'ai une douleur au genou")
    give_feedback(account, 1)

    summary = training.get_workout_history(account[0]).summary

    assert summary.workouts_logged == 2
    assert summary.completed == 2
    assert summary.difficulty_counts == {"slightly_too_hard": 2}
    assert summary.mentions_pain == 1


def test_history_covers_older_plans(account):
    give_feedback(account)
    repository = tools.open_repository()
    record = repository.list_plans(account[0], limit=1)[0]
    newer = repository.save_plan(
        account[0],
        PlanRequest.model_validate(record["request"]),
        PlanResult(
            status="completed",
            message="Newer",
            plan=TrainingPlan.model_validate(record["plan"]),
        ),
    )

    assert newer != account[1]
    history = training.get_workout_history(account[0])
    assert history.plans_checked == 2
    assert [entry.plan_id for entry in history.workouts] == [account[1]]


def test_history_for_unknown_student(account):
    assert training.get_workout_history("nobody").code == "student_not_found"


def test_recommendation_is_kept_without_changing_the_plan(account):
    before = tools.open_repository().get_plan(account[0], account[1])["plan"]

    result = training.save_recommendation(
        account[0], "Add a squat set", "To progress", account[1], ["source-1"]
    )

    assert result.saved
    pending = training.get_workout_history(account[0]).pending_recommendations
    assert pending[0].id == result.recommendation_id
    assert pending[0].source_ids == ["source-1"]
    assert tools.open_repository().get_plan(account[0], account[1])["plan"] == before


def test_recommendation_defaults_to_the_latest_plan(account):
    training.save_recommendation(account[0], "Add a squat set", "To progress")

    pending = training.get_workout_history(account[0]).pending_recommendations
    assert pending[0].plan_id == account[1]


def test_recommendation_rejects_another_students_plan(account):
    result = training.save_recommendation("nobody", "Idea", "Reason", account[1])

    assert result.code == "plan_not_found"


# ----------------------------------------------------------------- adaptation


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


def test_next_workout_is_adapted_and_saved(account):
    give_feedback(account)

    result = training.update_upcoming_workout(
        account[0], account[1], 1, 0, EasierGenerator()
    )

    assert result.changed
    assert result.workout_index == 1
    assert result.after.exercises[0].sets == result.before.exercises[0].sets - 1
    saved = tools.open_repository().get_plan(account[0], account[1])["plan"]
    assert saved["weeks"][0]["workouts"][1]["exercises"][0]["sets"] == (
        result.after.exercises[0].sets
    )


def test_same_feedback_cannot_adapt_twice(account):
    give_feedback(account)
    training.update_upcoming_workout(account[0], account[1], 1, 0, EasierGenerator())

    result = training.update_upcoming_workout(
        account[0], account[1], 1, 0, EasierGenerator()
    )

    assert result.code == "already_adapted"


@pytest.mark.parametrize("comment", ["My knee hurts", "J'ai mal au genou"])
def test_pain_stops_adaptation_without_calling_the_model(account, comment):
    give_feedback(account, comment=comment)
    generator = EasierGenerator()

    result = training.update_upcoming_workout(account[0], account[1], 1, 0, generator)

    assert result.code == "pain_reported"
    assert generator.calls == 0


def test_adaptation_needs_feedback_first(account):
    result = training.update_upcoming_workout(
        account[0], account[1], 1, 0, EasierGenerator()
    )

    assert result.code == "feedback_not_found"


def test_last_workout_has_nothing_to_adapt(account):
    give_feedback(account, 2)

    result = training.update_upcoming_workout(
        account[0], account[1], 1, 2, EasierGenerator()
    )

    assert result.code == "no_upcoming_workout"


def test_a_model_that_renames_exercises_changes_nothing(account):
    class Renaming(EasierGenerator):
        def generate_workout(self, *arguments):
            workout = super().generate_workout(*arguments)
            renamed = workout.exercises[0].model_copy(update={"name": "Something else"})
            return workout.model_copy(
                update={"exercises": [renamed, *workout.exercises[1:]]}
            )

    give_feedback(account)
    before = tools.open_repository().get_plan(account[0], account[1])["plan"]

    result = training.update_upcoming_workout(account[0], account[1], 1, 0, Renaming())

    assert result.code == "adaptation_failed"
    assert tools.open_repository().get_plan(account[0], account[1])["plan"] == before


# ------------------------------------------------------------------- schedule


def days_of(student_id):
    return [item.day for item in tools.get_timetable(student_id).planned_workouts]


def move(index, day):
    return training.WorkoutMove(week_number=1, workout_index=index, new_day=day)


def test_two_workouts_can_swap_days(account):
    result = training.organize_schedule(
        account[0], account[1], [move(0, "friday"), move(2, "Monday")]
    )

    assert result.ok
    assert days_of(account[0]) == ["Friday", "Wednesday", "Monday"]


def test_a_day_clash_changes_nothing(account):
    result = training.organize_schedule(account[0], account[1], [move(0, "Friday")])

    assert result.code == "day_conflict"
    assert days_of(account[0]) == ["Monday", "Wednesday", "Friday"]


def test_an_unavailable_day_is_refused(account):
    result = training.organize_schedule(account[0], account[1], [move(0, "Sunday")])

    assert result.code == "day_not_available"


def test_moving_a_booked_workout_reports_its_stale_event(account):
    calendar = cal.FakeCalendar()
    start = datetime.combine(NEXT_MONDAY, time(18), PARIS)
    booked = cal.add_workouts_to_calendar(
        account[0],
        calendar,
        TIMEZONE,
        NOW,
        account[1],
        [cal.Booking(week_number=1, workout_index=0, start=start)],
    )

    result = training.organize_schedule(
        account[0], account[1], [move(0, "Friday"), move(2, "Monday")]
    )

    assert result.calendar_events_now_out_of_date == [booked.outcomes[0].event_id]


# --------------------------------------------------------------------- search


class FakeRetriever:
    def __init__(self, error=None):
        self.error = error

    def search(self, _query, _top_k):
        if self.error:
            raise self.error
        return [
            RetrievedChunk(
                text="x" * 1000,
                chunk_id="c1",
                video_id="v1",
                passage_ids=["p1"],
                title="Rest times",
                video_url="https://example.com/v1",
            )
        ]


@pytest.mark.parametrize("query", ["My knee hurts", "J'ai mal au dos, que faire ?"])
def test_search_refuses_pain_questions_before_searching(query):
    retriever = FakeRetriever(AssertionError("should not search"))

    assert training.search_documents(query, retriever).code == "health_question"


def test_agent_search_skips_the_plan_scope_check(account):
    class GatedRetriever:
        """Like QueryExpansionRetriever: it refuses, but wraps a working search."""

        retriever = FakeRetriever()

        def search(self, _query, _top_k):
            raise ScopeRefusalError("out_of_scope", "Plan requests only.")

    class Pipeline:
        retriever = GatedRetriever()
        generator = None

    model = ScriptedModel(call("search_documents", query="rest times"), AIMessage("ok"))
    graph = build_agent(context(account, pipeline_factory=Pipeline), model)

    reply = ask(graph, "search", thread_id="t")

    assert '"title":"Rest times"' in reply.tool_results[0][1]


def test_search_returns_short_cited_passages():
    result = training.search_documents("rest times", FakeRetriever())

    document = result.documents[0]
    assert (document.source_id, document.title) == ("c1", "Rest times")
    assert len(document.excerpt) == training.EXCERPT_CHARACTERS


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ScopeRefusalError("out_of_scope", "Not about training."), "out_of_scope"),
        (RuntimeError("Ollama is down"), "search_unavailable"),
    ],
)
def test_search_failures_are_reported(error, code):
    assert training.search_documents("x", FakeRetriever(error)).code == code


# ----------------------------------------------------------------- date logic


def test_weekday_becomes_a_date_in_the_right_week():
    assert cal.date_of_workout(NEXT_MONDAY, 1, "Wednesday") == date(2030, 1, 16)
    assert cal.date_of_workout(NEXT_MONDAY, 2, "monday") == date(2030, 1, 21)
    assert cal.date_of_workout(NEXT_MONDAY, 1, "Someday") is None


def test_week_start_waits_until_every_workout_is_still_ahead():
    wednesday = date(2030, 1, 9)
    this_monday = date(2030, 1, 7)
    assert cal.default_week_start(["Monday", "Friday"], wednesday) == NEXT_MONDAY
    assert cal.default_week_start(["Wednesday", "Friday"], wednesday) == this_monday


def test_free_times_skip_busy_periods_and_the_past():
    day = date(2030, 1, 9)
    busy = [busy_on(day, 12, 14)]

    starts = cal.free_start_times(day, 60, busy, PARIS, NOW, time(10), time(16))

    assert [item.strftime("%H:%M") for item in starts] == [
        "10:30",
        "11:00",
        "14:00",
        "14:30",
        "15:00",
    ]


def test_free_times_handle_a_daylight_saving_day():
    # Clocks go forward on this Sunday in Paris, so 03:00 local is an hour after 01:00.
    day = date(2030, 3, 31)
    long_ago = datetime(2000, 1, 1, tzinfo=UTC)

    starts = cal.free_start_times(day, 45, [], PARIS, long_ago, time(7), time(8))

    assert [item.utcoffset() for item in starts] == [timedelta(hours=2)]


def test_options_are_spread_across_the_day():
    day = date(2030, 1, 14)
    starts = cal.free_start_times(day, 30, [], PARIS, NOW, time(7), time(21))

    options = cal.spread_options(starts)

    assert [item.hour for item in options] == [7, 14, 20]


# ---------------------------------------------------------------- calendar tools


def test_slots_stay_on_planned_days_and_avoid_busy_times(account):
    calendar = cal.FakeCalendar([busy_on(NEXT_MONDAY, 7, 18)])

    result = cal.propose_training_slots(account[0], calendar, TIMEZONE, NOW)

    assert result.week_start == NEXT_MONDAY
    monday = result.workouts[0]
    assert monday.planned_day == "Monday"
    assert not monday.moved_to_another_day
    assert monday.options[0].start.hour == 18
    assert {slot.day for slot in monday.options} == {"Monday"}


def test_a_full_day_moves_to_a_spare_available_day(account):
    repository = tools.open_repository()
    profile = repository.get_profile(account[0])
    repository.save_profile(
        account[0],
        profile.model_copy(
            update={"available_days": [*profile.available_days, "Saturday"]}
        ),
    )
    calendar = cal.FakeCalendar([busy_on(NEXT_MONDAY, 0, 23)])

    workout = cal.propose_training_slots(account[0], calendar, TIMEZONE, NOW).workouts[0]

    assert workout.moved_to_another_day
    assert {slot.day for slot in workout.options} == {"Saturday"}


def test_no_free_time_is_said_plainly(account):
    calendar = cal.FakeCalendar([busy_on(NEXT_MONDAY, 0, 23)])

    workout = cal.propose_training_slots(account[0], calendar, TIMEZONE, NOW).workouts[0]

    assert workout.options == []
    assert "No free time" in workout.note


def test_logged_and_booked_workouts_are_not_offered_again(account):
    give_feedback(account, 0)
    calendar = cal.FakeCalendar()
    start = datetime.combine(NEXT_MONDAY + timedelta(days=2), time(18), PARIS)
    cal.add_workouts_to_calendar(
        account[0],
        calendar,
        TIMEZONE,
        NOW,
        account[1],
        [cal.Booking(week_number=1, workout_index=1, start=start)],
    )

    result = cal.propose_training_slots(account[0], calendar, TIMEZONE, NOW)

    assert [workout.workout_index for workout in result.workouts] == [2]


def booking(index=0, hour=18):
    return cal.Booking(
        week_number=1,
        workout_index=index,
        start=datetime.combine(NEXT_MONDAY, time(hour), PARIS),
    )


def test_booking_creates_one_event_and_a_retry_does_not_duplicate(account):
    calendar = cal.FakeCalendar()

    first = cal.add_workouts_to_calendar(
        account[0], calendar, TIMEZONE, NOW, account[1], [booking()]
    )
    second = cal.add_workouts_to_calendar(
        account[0], calendar, TIMEZONE, NOW, account[1], [booking()]
    )

    assert first.outcomes[0].status == "booked"
    assert second.outcomes[0].status == "already_booked"
    assert len(calendar.events) == 1
    event = next(iter(calendar.events.values()))
    assert "Goblet squat" in event["description"]


def test_a_conflict_blocks_only_that_booking(account):
    calendar = cal.FakeCalendar([busy_on(NEXT_MONDAY, 17, 19)])
    wednesday = booking(1, 18).model_copy(
        update={"start": datetime.combine(NEXT_MONDAY + timedelta(2), time(18), PARIS)}
    )

    result = cal.add_workouts_to_calendar(
        account[0], calendar, TIMEZONE, NOW, account[1], [booking(), wednesday]
    )

    assert [item.status for item in result.outcomes] == ["conflict", "booked"]
    assert len(calendar.events) == 1


def test_a_past_time_is_refused(account):
    past = booking().model_copy(update={"start": NOW - timedelta(days=1)})

    result = cal.add_workouts_to_calendar(
        account[0], cal.FakeCalendar(), TIMEZONE, NOW, account[1], [past]
    )

    assert result.outcomes[0].status == "failed"


def test_a_time_without_a_zone_uses_the_students(account):
    naive = booking().model_copy(
        update={"start": datetime.combine(NEXT_MONDAY, time(18))}
    )

    result = cal.add_workouts_to_calendar(
        account[0], cal.FakeCalendar(), TIMEZONE, NOW, account[1], [naive]
    )

    assert result.outcomes[0].start.utcoffset() == timedelta(hours=1)


def test_booking_on_another_day_is_flagged(account):
    tuesday = booking().model_copy(
        update={"start": datetime.combine(NEXT_MONDAY + timedelta(1), time(18), PARIS)}
    )

    result = cal.add_workouts_to_calendar(
        account[0], cal.FakeCalendar(), TIMEZONE, NOW, account[1], [tuesday]
    )

    assert result.outcomes[0].plan_day_differs


@pytest.mark.parametrize(
    "call",
    [
        lambda sid, plan: cal.get_calendar_busy(None, TIMEZONE, NOW),
        lambda sid, plan: cal.propose_training_slots(sid, None, TIMEZONE, NOW),
        lambda sid, plan: cal.add_workouts_to_calendar(
            sid, None, TIMEZONE, NOW, plan, [booking()]
        ),
    ],
)
def test_no_calendar_is_reported_not_treated_as_free(account, call):
    assert call(*account).code == "calendar_not_connected"


class DownCalendar(cal.FakeCalendar):
    def busy_periods(self, start, end):
        raise cal.CalendarError("Calendar is down")


def test_an_unreadable_calendar_is_never_treated_as_free(account):
    result = cal.propose_training_slots(account[0], DownCalendar(), TIMEZONE, NOW)

    assert result.code == "calendar_unavailable"


def test_a_failed_recheck_books_nothing(account):
    calendar = DownCalendar()

    result = cal.add_workouts_to_calendar(
        account[0], calendar, TIMEZONE, NOW, account[1], [booking()]
    )

    assert result.outcomes[0].status == "failed"
    assert calendar.events == {}
    assert tools.open_repository().list_calendar_events(account[0], account[1]) == []


# --------------------------------------------------------------- Google client


def google(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GoogleCalendar(lambda: "secret-token", client)


START = datetime(2030, 1, 14, 18, tzinfo=PARIS)
END = START + timedelta(hours=1)


def test_google_busy_periods_are_parsed_and_authorized():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["Authorization"]
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "calendars": {
                    "primary": {
                        "busy": [
                            {
                                "start": "2030-01-14T17:00:00Z",
                                "end": "2030-01-14T18:00:00Z",
                            }
                        ]
                    }
                }
            },
        )

    busy = google(handler).busy_periods(START, END)

    assert seen["auth"] == "Bearer secret-token"
    assert seen["url"].endswith("/freeBusy")
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


def test_google_network_failure_becomes_a_calendar_error():
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(cal.CalendarError):
        google(handler).create_event("key", "t", "d", START, END)


def test_google_event_uses_the_stable_key():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": seen["body"]["id"]})

    assert google(handler).create_event("abc123", "t", "d", START, END) == "abc123"
    assert seen["body"]["start"]["dateTime"] == START.isoformat()


def test_google_duplicate_event_counts_as_already_created():
    assert (
        google(lambda _request: httpx.Response(409)).create_event(
            "abc123", "t", "d", START, END
        )
        == "abc123"
    )


def feature(name, latitude, longitude=2.33):
    return {
        "properties": {"name": name},
        "geometry": {"coordinates": [longitude, latitude]},
    }


# ------------------------------------------------------------- through the graph


class ScriptedPlanner:
    """Give each plan in the list in turn, then the default one-step plan."""

    def __init__(self, plans):
        self.plans = plans

    def invoke(self, _messages):
        if self.plans:
            return self.plans.pop(0)
        return Plan(steps=[PlanStep(task="Answer the request.")])


class ScriptedModel:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.plans = []
        self.seen = []

    def bind_tools(self, _bound):
        return self

    def with_structured_output(self, _schema):
        return ScriptedPlanner(self.plans)

    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def call(name, **arguments):
    return AIMessage(
        "", tool_calls=[{"name": name, "args": arguments, "id": f"{name}-1"}]
    )


def context(account, **extra):
    return AgentContext(
        account[0], timezone=TIMEZONE, clock=lambda: NOW, **extra
    )


def test_prompt_tells_the_model_the_date(account):
    model = ScriptedModel(AIMessage("ok"))
    graph = build_agent(context(account), model)

    ask(graph, "hi", thread_id="t")

    assert "2030-01-09" in model.seen[0][0].content


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "organize_schedule",
            {"moves": [{"week_number": 1, "workout_index": 0, "new_day": "Friday"},
                       {"week_number": 1, "workout_index": 2, "new_day": "Monday"}]},
        ),
        (
            "add_workouts_to_calendar",
            {"bookings": [{"week_number": 1, "workout_index": 0,
                           "start": "2030-01-14T18:00:00"}]},
        ),
    ],
)
def test_writes_do_not_run_when_declined(account, tool_name, arguments):
    calendar = cal.FakeCalendar()
    model = ScriptedModel(
        call(tool_name, plan_id=account[1], **arguments), AIMessage("Not changed.")
    )
    graph = build_agent(context(account, calendar=calendar), model)

    reply = ask(graph, "do it", thread_id="t", approve=lambda _request: False)

    assert reply.approvals[0][1] is False
    assert calendar.events == {}
    assert days_of(account[0]) == ["Monday", "Wednesday", "Friday"]


def test_booking_runs_with_json_arguments_when_approved(account):
    calendar = cal.FakeCalendar()
    model = ScriptedModel(
        call(
            "add_workouts_to_calendar",
            plan_id=account[1],
            bookings=[
                {"week_number": 1, "workout_index": 0, "start": "2030-01-14T18:00:00"}
            ],
        ),
        AIMessage("Booked."),
    )
    graph = build_agent(context(account, calendar=calendar), model)

    reply = ask(graph, "book", thread_id="t", approve=lambda _request: True)

    assert '"status":"booked"' in reply.tool_results[0][1]
    assert len(calendar.events) == 1


def test_performance_runs_with_json_arguments(account):
    model = ScriptedModel(
        call(
            "log_workout_performance",
            plan_id=account[1],
            week_number=1,
            workout_index=0,
            exercises=[
                {"exercise_name": "Goblet squat", "sets_completed": 3, "reps_per_set": "10"}
            ],
        ),
        AIMessage("Saved."),
    )
    graph = build_agent(context(account), model)

    reply = ask(graph, "log", thread_id="t")

    assert reply.approvals == []
    assert '"saved":true' in reply.tool_results[0][1]


class StubPipeline:
    def __init__(self):
        self.retriever = FakeRetriever()
        self.generator = EasierGenerator()

    def create_plan(self, request, *, previous_plan=None):
        return PlanResult(
            status="completed",
            message="Done.",
            plan=TrainingPlan.model_validate(previous_plan["plan"]),
        )


def test_search_and_adaptation_use_the_loaded_pipeline(account):
    give_feedback(account)
    model = ScriptedModel(
        call("search_documents", query="rest times"),
        call("update_upcoming_workout", plan_id=account[1], week_number=1, workout_index=0),
        AIMessage("Done."),
    )
    graph = build_agent(
        context(account, pipeline_factory=lambda: StubPipeline()), model
    )

    reply = ask(graph, "do both", thread_id="t", approve=lambda _request: True)

    assert '"title":"Rest times"' in reply.tool_results[0][1]
    assert '"changed":true' in reply.tool_results[1][1]


def test_missing_corpus_is_reported_by_the_search_tool(account):
    model = ScriptedModel(call("search_documents", query="rest"), AIMessage("Sorry."))
    graph = build_agent(context(account), model)

    reply = ask(graph, "search", thread_id="t")

    assert '"code":"services_unavailable"' in reply.tool_results[0][1]


def test_a_new_plan_shows_free_slots_without_another_model_call(account):
    calendar = cal.FakeCalendar()
    model = ScriptedModel(call("generate_workout_plan"))
    graph = build_agent(
        context(account, calendar=calendar, pipeline_factory=lambda: StubPipeline()), model
    )

    reply = ask(graph, "Build my legs", thread_id="t", approve=lambda _request: True)

    assert len(model.seen) == 1
    assert "not booked yet" in reply.answer
    assert "Monday 2030-01-14 07:00" in reply.answer
    assert calendar.events == {}


def test_a_new_plan_survives_an_unreadable_calendar(account):
    model = ScriptedModel(call("generate_workout_plan"))
    graph = build_agent(
        context(account, calendar=DownCalendar(), pipeline_factory=lambda: StubPipeline()),
        model,
    )

    reply = ask(graph, "Build my legs", thread_id="t", approve=lambda _request: True)

    assert "Saved as plan" in reply.answer
    assert "could not read your calendar" in reply.answer


def test_a_booking_on_another_day_always_reaches_the_student(account):
    model = ScriptedModel(
        call(
            "add_workouts_to_calendar",
            plan_id=account[1],
            bookings=[
                {"week_number": 1, "workout_index": 0, "start": "2030-01-15T18:00:00"}
            ],
        ),
        AIMessage("Booked."),
    )
    graph = build_agent(context(account, calendar=cal.FakeCalendar()), model)

    reply = ask(graph, "book", thread_id="t", approve=lambda _request: True)

    assert reply.answer.startswith("Booked.")
    assert "different day" in reply.answer


@pytest.mark.parametrize(
    "request_text",
    [
        "Comment pourrais-je organiser mes séances de la semaine en fonction de mon programme ?",
        "Propose-moi des créneaux dans mon agenda.",
        "Can you find slots for my workouts?",
        "Organize my week",
    ],
)
def test_requests_for_training_times_are_recognized(request_text):
    assert wants_training_slots(request_text)


@pytest.mark.parametrize(
    "request_text",
    [
        "Réserve le premier créneau proposé pour chaque séance.",
        "Book the first slot.",
        "Organise-moi un programme pour les jambes.",
        "Échange mes séances de lundi et de vendredi.",
        "Quelle est la salle la plus proche ?",
    ],
)
def test_other_requests_are_left_to_the_model(request_text):
    assert not wants_training_slots(request_text)


def test_a_slot_request_runs_the_tool_before_the_model(account):
    model = ScriptedModel(AIMessage("Voici vos créneaux."))
    graph = build_agent(context(account, calendar=cal.FakeCalendar()), model)

    reply = ask(graph, "Organise mes séances de la semaine", thread_id="t")

    assert reply.tool_calls == [("propose_training_slots", {})]
    assert reply.answer == "Voici vos créneaux."
    shown = model.seen[0][-1]
    assert shown.name == "propose_training_slots"
    assert account[1] in shown.content


def test_without_a_calendar_a_slot_request_goes_to_the_model(account):
    model = ScriptedModel(AIMessage("Pas de calendrier."))
    graph = build_agent(context(account), model)

    reply = ask(graph, "Organise mes séances de la semaine", thread_id="t")

    assert reply.tool_calls == []


def test_a_moved_workout_always_reaches_the_student(account):
    repository = tools.open_repository()
    profile = repository.get_profile(account[0])
    repository.save_profile(
        account[0],
        profile.model_copy(
            update={"available_days": [*profile.available_days, "Saturday"]}
        ),
    )
    calendar = cal.FakeCalendar([busy_on(NEXT_MONDAY, 0, 23)])
    graph = build_agent(
        context(account, calendar=calendar), ScriptedModel(AIMessage("Voici."))
    )

    reply = ask(graph, "Propose des créneaux", thread_id="t")

    assert "Monday has no free time" in reply.answer


def test_slots_after_a_new_plan_are_ready_for_a_booking(account):
    model = ScriptedModel(call("generate_workout_plan"), AIMessage("Réservé."))
    graph = build_agent(
        context(
            account,
            calendar=cal.FakeCalendar(),
            pipeline_factory=lambda: StubPipeline(),
        ),
        model,
    )

    ask(graph, "Build my legs", thread_id="t", approve=lambda _request: True)
    ask(graph, "Book the first ones", thread_id="t")

    assert any(
        getattr(message, "name", "") == "propose_training_slots"
        for message in model.seen[1]
    )


def test_a_shop_suited_to_strength_is_named_to_try_first(monkeypatch):
    features = [
        feature("Running Conseil", 48.791),
        feature("Fitnessboutique", 48.80),
    ]
    monkeypatch.setattr(
        places.httpx,
        "get",
        lambda url, **_: httpx.Response(
            200, json={"features": features}, request=httpx.Request("GET", url)
        ),
    )
    location = places.Location(label="Test", latitude=48.79, longitude=2.33)

    result = find_nearby_places("sports_store", location)

    assert result.places[0].name == "Running Conseil"
    assert result.try_first == "Fitnessboutique"


def test_a_gym_search_has_no_shop_to_try_first(monkeypatch):
    monkeypatch.setattr(
        places.httpx,
        "get",
        lambda url, **_: httpx.Response(
            200,
            json={"features": [feature("Fitness Park", 48.791)]},
            request=httpx.Request("GET", url),
        ),
    )
    location = places.Location(label="Test", latitude=48.79, longitude=2.33)

    assert find_nearby_places("gym", location).try_first is None


def test_published_schemas_match_the_tools():
    published = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))

    assert published == agent_tool_schemas(), (
        "Run: python -m agentic.call_tool --write-schemas"
    )


def test_every_agent_tool_is_documented():
    readme = (SCHEMA_FILE.parent / "README.md").read_text(encoding="utf-8")

    for item in agent_tool_schemas()["tools"]:
        assert f"`{item['name']}`" in readme


def test_a_crashing_tool_goes_back_to_the_model(account, monkeypatch):
    def crash(*_arguments, **_keywords):
        raise AttributeError("a bug")

    monkeypatch.setattr(training, "get_workout_history", crash)
    model = ScriptedModel(call("get_workout_history"), AIMessage("Sorry."))
    graph = build_agent(context(account), model)

    reply = ask(graph, "history", thread_id="t")

    assert reply.answer == "Sorry."
    assert "failed unexpectedly" in reply.tool_results[0][1]
