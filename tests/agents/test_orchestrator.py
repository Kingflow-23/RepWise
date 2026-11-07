"""Plan-and-Execute across specialists, with a scripted model and mocked services."""

import httpx
import pytest
from langchain_core.messages import AIMessage

from src.agents.orchestrator import (
    HEALTH_NOTICE,
    NO_MODEL_ANSWER,
    Delegation,
    OrchestratorPlan,
    build_orchestrator,
    key_facts,
    places_notice,
)
from src.integrations.photon import PhotonClient
from src.schemas.models import PlanResult, TrainingPlan
from src.tools.calendar import FakeCalendar
from src.tools.context import Location
from src.tools.profile import get_timetable
from tests.conftest import ScriptedModel, make_context, tool_call

HOME = Location(label="Home", latitude=48.79, longitude=2.33)


def gym_service():
    """A Photon stand-in that always finds one gym."""

    def handler(_request):
        feature = {
            "properties": {"name": "Basic-Fit Cachan"},
            "geometry": {"coordinates": [2.331, 48.791]},
        }
        return httpx.Response(200, json={"features": [feature]})

    return PhotonClient(
        "https://photon.test", "tests", transport=httpx.MockTransport(handler)
    )


def demo_pipeline(demo):
    """A plan pipeline that returns the demo plan without any model."""
    record = demo.repository.get_plan(demo.student_id, demo.plan_id)

    class Pipeline:
        retriever = generator = None

        def create_plan(self, request, previous_plan=None):
            return PlanResult(
                status="completed",
                message="ok",
                plan=TrainingPlan.model_validate(record["plan"]),
            )

    return Pipeline()


def agent_context(demo, **extra):
    values = {
        "places": gym_service(),
        "session_location": HOME,
        "pipeline_factory": lambda: demo_pipeline(demo),
    }
    values.update(extra)
    return make_context(demo.repository, demo.student_id, **values)


def plan(*steps, **fields):
    """Build a plan; a plan without steps is a saved-data answer by default."""
    fields.setdefault("request_kind", "needs_specialists" if steps else "saved_data")
    return OrchestratorPlan(
        steps=[Delegation(agent=a, task=t) for a, t in steps], **fields
    )


def test_a_plan_that_needs_specialists_but_has_no_steps_is_asked_again(demo):
    model = ScriptedModel(
        [
            tool_call("search_nearby_places", category="gym"),
            AIMessage("Found."),
            AIMessage("Final."),
        ],
        [
            plan(request_kind="needs_specialists"),
            plan(("location", "Find the nearest gyms.")),
        ],
    )

    response = ask(agent_context(demo), model, "Where is the nearest gym?")

    assert [step.agent for step in response.steps] == ["location"]


def ask(context, model, request, conversation="c1"):
    return build_orchestrator(context, model).ask(request, conversation)


# --- routing of the agent sheet's requests -----------------------------------


@pytest.mark.parametrize(
    ("request_text", "agent", "call"),
    [
        (
            "Pourrais-tu me trouver la salle de sport la plus près de moi ?",
            "location",
            tool_call("search_nearby_places", category="gym"),
        ),
        (
            "Dans quel magasin proche de moi puis-je trouver des haltères ?",
            "location",
            tool_call("search_nearby_places", category="sports_store"),
        ),
        (
            "Dans quel magasin proche de moi puis-je trouver de la créatine ?",
            "location",
            tool_call("search_nearby_places", category="supplement_store"),
        ),
        (
            "Comment pourrais-je organiser mes séances de la semaine en fonction de mon programme ?",
            "schedule",
            tool_call("propose_training_slots"),
        ),
        (
            "Propose-moi un programme pour prendre du muscle au niveau de mes jambes.",
            "planning",
            tool_call("generate_workout_plan", focus="Build leg muscle"),
        ),
    ],
)
def test_each_sheet_request_reaches_its_specialist(demo, request_text, agent, call):
    model = ScriptedModel(
        [call, AIMessage("Specialist report."), AIMessage("Final answer.")],
        [plan((agent, request_text))],
    )

    response = ask(agent_context(demo, calendar=FakeCalendar()), model, request_text)

    assert [step.agent for step in response.steps] == [agent]
    assert response.steps[0].status == "completed"
    assert response.reports[0].tools_used[0]["ok"] is True
    assert response.answer.startswith("Final answer.")


# --- multi-agent requests ------------------------------------------------------


def test_find_a_gym_and_create_a_programme_coordinates_two_specialists(demo):
    model = ScriptedModel(
        [
            tool_call("search_nearby_places", category="gym"),
            AIMessage("Nearest gym: Basic-Fit Cachan."),
            tool_call(
                "generate_workout_plan", focus="Build muscle at Basic-Fit Cachan"
            ),
            AIMessage("Plan saved as a draft."),
            AIMessage("Here is your gym and your new plan."),
        ],
        [
            plan(
                ("location", "Find the nearest gym."),
                ("planning", "Create a muscle programme."),
            )
        ],
    )

    response = ask(
        agent_context(demo),
        model,
        "Find a gym near me and create a training programme.",
    )

    assert [step.agent for step in response.steps] == ["location", "planning"]
    assert all(step.status == "completed" for step in response.steps)
    # The planning specialist saw the location specialist's result.
    planning_prompt = str(model.seen[2][0].content)
    assert (
        "Basic-Fit Cachan" in planning_prompt
        and "Results of earlier steps" in planning_prompt
    )
    # The new plan replaces an active one, so its activation waits for approval.
    assert [action.tool_name for action in response.pending_actions] == [
        "activate_plan"
    ]
    assert "Waiting for your approval" in response.answer
    assert demo.repository.get_current_plan(demo.student_id)["id"] == demo.plan_id


def test_missed_workouts_are_recorded_and_the_week_reorganised_only_with_approval(demo):
    profile = demo.repository.get_profile(demo.student_id)
    demo.repository.save_profile(
        demo.student_id,
        profile.model_copy(
            update={"available_days": [*profile.available_days, "Saturday"]}
        ),
    )
    missed = {
        "completion_status": "not_completed",
        "difficulty": "just_right",
        "week_number": 1,
    }
    model = ScriptedModel(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "save_workout_feedback",
                        "args": {**missed, "workout_index": 0},
                        "id": "f0",
                    },
                    {
                        "name": "save_workout_feedback",
                        "args": {**missed, "workout_index": 1},
                        "id": "f1",
                    },
                ],
            ),
            AIMessage("Recorded two missed workouts."),
            tool_call("organize_schedule", catch_up_week=1),
            AIMessage("Proposed a catch-up schedule."),
            AIMessage("I recorded the missed workouts and proposed a new schedule."),
        ],
        [
            plan(
                ("performance", "Record Monday and Wednesday as missed."),
                ("schedule", "Reorganise the rest of the week."),
            )
        ],
    )

    response = ask(
        agent_context(demo),
        model,
        "I missed two workouts; organise the rest of my week.",
    )

    feedback = demo.repository.list_workout_feedback(demo.student_id, demo.plan_id)
    assert [item.completion_status for item in feedback] == [
        "not_completed",
        "not_completed",
    ]
    assert [action.tool_name for action in response.pending_actions] == [
        "organize_schedule"
    ]
    workouts = demo.repository.get_plan(demo.student_id, demo.plan_id)["plan"]["weeks"][
        0
    ]["workouts"]
    assert workouts[2]["day"] == "Friday"  # unchanged until approved
    # The approval names the exact moves code chose: missed workouts first.
    reason = response.pending_actions[0].reason
    assert "Monday (missed) → Friday" in reason
    assert "Wednesday (missed) → Saturday" in reason


def test_proposed_times_reach_the_student_exactly_as_code_computed_them(demo):
    model = ScriptedModel(
        [
            tool_call("propose_training_slots"),
            AIMessage("Slots found."),
            AIMessage("Lundi 7 novembre à 7h (wrong date)."),
        ],
        [plan(("schedule", "Find free times."))],
    )

    response = ask(
        agent_context(demo, calendar=FakeCalendar()), model, "Organise my week."
    )

    assert "Free times checked in your calendar, nothing booked yet" in response.answer
    assert "Monday 2030-01-14 07:00" in response.answer


def test_key_facts_separate_available_days_from_workout_days(demo):
    facts = key_facts("Saturday", get_timetable(demo.context))

    assert "- Today is Saturday." in facts
    assert (
        "Available days (when the student can train, from the profile): Monday, Wednesday, Friday."
        in facts
    )
    assert "Week 1 of the current plan has 3 workouts." in facts
    assert "Available days without a workout in week 1: none." in facts


def test_places_reach_the_student_in_code_order_whatever_the_answer_says():
    def search(name, km):
        return {"name": name, "distance_km": km, "address": None, "map_url": None}

    report = {
        "all_results": [
            {
                "name": "search_nearby_places",
                "data": {
                    "ok": True,
                    "category": "sports_store",
                    "near": {"label": "Cachan"},
                    "radius_km": 5,
                    "places": [search("Near shop", 0.5), search("Far shop", 2.1)],
                    "data_source": "OpenStreetMap",
                },
            },
            {"name": "search_nearby_places", "data": {"ok": False, "code": "x"}},
        ]
    }

    notice = places_notice(report)

    assert notice.index("Near shop: 0.50 km") < notice.index("Far shop: 2.10 km")
    assert "near Cachan, nearest first" in notice


# --- plan-and-execute control ------------------------------------------------


def test_a_failed_step_is_replanned_once(demo):
    model = ScriptedModel(
        [
            tool_call("get_calendar_busy"),
            AIMessage("No calendar is connected."),
            tool_call(
                "organize_schedule",
                moves=[
                    {"week_number": 1, "workout_index": 0, "new_day": "Friday"},
                    {"week_number": 1, "workout_index": 2, "new_day": "Monday"},
                ],
            ),
            AIMessage("Proposed a swap within the plan."),
            AIMessage("Final."),
        ],
        [
            plan(("schedule", "Check when I am busy.")),
            plan(("schedule", "Move workouts within the plan instead.")),
        ],
    )

    response = ask(agent_context(demo), model, "Organise my week.")

    assert response.replans == 1
    assert [step.status for step in response.steps] == ["failed", "completed"]
    assert "could not finish" in response.answer


def test_replanning_is_bounded(demo):
    model = ScriptedModel(
        [
            tool_call("get_calendar_busy"),
            AIMessage("No calendar."),
            tool_call("get_calendar_busy", call_id="c2"),
            AIMessage("Still no calendar."),
            AIMessage("Final."),
        ],
        [
            plan(("schedule", "Check busy times.")),
            plan(("schedule", "Check busy times again.")),
        ],
    )

    response = ask(agent_context(demo), model, "When am I busy?")

    assert response.replans == 1
    assert [step.status for step in response.steps] == ["failed", "failed"]


def test_the_number_of_delegations_is_capped(demo):
    steps = [("knowledge", f"Question {i}") for i in range(8)]
    model = ScriptedModel([], [plan(*steps)])

    response = ask(agent_context(demo), model, "Many questions.")

    assert len(response.steps) == demo.context.config.agent_limits.max_delegations


def test_a_request_answerable_from_context_needs_no_specialist(demo):
    model = ScriptedModel([AIMessage("Your goal is general strength.")], [plan()])

    response = ask(agent_context(demo), model, "What is my goal?")

    assert response.steps == [] and response.reports == []
    assert "general strength" in response.answer
    # The orchestrator's read tools ran once, in code, to build the shared context.
    assert "Build general strength" in str(model.seen[0][0].content)


def test_without_any_specialist_the_answer_is_told_no_documents_were_searched(demo):
    model = ScriptedModel([AIMessage("Hello!")], [plan()])

    ask(agent_context(demo), model, "Hi")

    assert "no documents were searched" in str(model.seen[0][0].content)


def test_an_unreachable_planner_changes_nothing_and_says_so(demo):
    model = ScriptedModel([], [httpx.ConnectError("down")])

    response = ask(agent_context(demo), model, "Organise my week.")

    assert response.answer.startswith(NO_MODEL_ANSWER)
    assert response.health_concern  # fail closed for memory


def test_a_step_about_one_workout_receives_that_workout(demo):
    model = ScriptedModel(
        [AIMessage("Looked at it."), AIMessage("Final.")],
        [
            OrchestratorPlan(
                steps=[
                    Delegation(
                        agent="performance",
                        task="Review it.",
                        workout={"week_number": 1, "workout_index": 1},
                    )
                ]
            )
        ],
    )

    ask(agent_context(demo), model, "How was Wednesday's workout?")

    assert "The workout this task is about" in str(model.seen[0][0].content)
    assert "Wednesday" in str(model.seen[0][0].content)


# --- memory and safety -----------------------------------------------------------


def test_facts_the_student_asked_to_remember_are_stored_but_health_details_are_not(
    demo,
):
    model = ScriptedModel(
        [AIMessage("Noted.")],
        [plan(remember=["I prefer morning sessions", "My shoulder hurts"])],
    )

    response = ask(
        agent_context(demo),
        model,
        "Remember I prefer mornings, and that my shoulder hurts.",
    )

    assert demo.repository.list_memory_facts(demo.student_id, 10) == [
        "I prefer morning sessions"
    ]
    assert any("was not remembered" in notice for notice in response.notices)


def test_a_health_concern_adds_the_medical_notice_and_is_not_kept_in_memory(demo):
    model = ScriptedModel(
        [
            tool_call("search_documents", query="knee pain"),
            AIMessage("Refused."),
            AIMessage("Final."),
        ],
        [plan(("knowledge", "Answer about knee pain."), health_concern=True)],
    )

    response = ask(
        agent_context(demo), model, "My knee hurts when I squat, what should I do?"
    )

    assert HEALTH_NOTICE in response.answer
    episodes = demo.repository.list_memory_episodes(demo.student_id, 5)
    assert episodes and "knee" not in episodes[0]


def test_an_episode_records_what_code_observed(demo):
    model = ScriptedModel(
        [
            tool_call("search_nearby_places", category="gym"),
            AIMessage("Found one."),
            AIMessage("Final."),
        ],
        [plan(("location", "Find a gym."))],
    )

    ask(agent_context(demo), model, "Find a gym")

    episode = demo.repository.list_memory_episodes(demo.student_id, 1)[0]
    assert "Asked: Find a gym" in episode and "Location Agent" in episode


def test_conversations_are_kept_apart(demo):
    context = agent_context(demo)
    model = ScriptedModel(
        [AIMessage("First answer."), AIMessage("Second answer.")], [plan(), plan()]
    )
    orchestrator = build_orchestrator(context, model)

    orchestrator.ask("Hello from thread A", "thread-a")
    orchestrator.ask("Hello from thread B", "thread-b")

    second_prompt = str(model.seen[1][0].content)
    assert "Hello from thread A" not in second_prompt


def test_a_plan_still_without_steps_falls_back_to_one_constrained_choice(demo):
    from src.agents.orchestrator import SpecialistChoice

    model = ScriptedModel(
        [
            tool_call("search_nearby_places", category="gym"),
            AIMessage("Found."),
            AIMessage("Final."),
        ],
        [
            plan(request_kind="needs_specialists"),
            plan(request_kind="needs_specialists"),
            SpecialistChoice(agent="location"),
        ],
    )

    response = ask(agent_context(demo), model, "Where is the nearest gym?")

    assert [(s.agent, s.task) for s in response.steps] == [
        ("location", "Where is the nearest gym?")
    ]


def test_every_request_reports_phases_model_calls_tokens_and_tool_timings(demo):
    reply = AIMessage(
        "",
        tool_calls=[
            {"name": "search_nearby_places", "args": {"category": "gym"}, "id": "c1"}
        ],
    )
    reply.usage_metadata = {
        "input_tokens": 120,
        "output_tokens": 8,
        "total_tokens": 128,
    }
    model = ScriptedModel(
        [reply, AIMessage("Found."), AIMessage("Final.")],
        [plan(("location", "Find a gym."))],
    )

    metrics = ask(agent_context(demo), model, "Find a gym").metrics

    assert [phase.name for phase in metrics.phases] == [
        "load_context",
        "plan",
        "delegate: location",
        "check",
        "respond",
        "remember",
    ]
    purposes = [call.purpose for call in metrics.model_calls]
    assert purposes == ["plan", "respond", "location: think", "location: think"]
    assert metrics.input_tokens == 120 and metrics.output_tokens == 8
    assert metrics.tool_calls == 1 and metrics.tool_seconds >= 0
    assert metrics.total_seconds >= sum(phase.seconds for phase in metrics.phases) * 0.9
