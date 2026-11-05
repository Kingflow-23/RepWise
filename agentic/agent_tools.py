"""The tools the agent's model can call, with the session's values filled in.

The work happens in ``tools.py``, ``places.py``, ``training_tools.py``, and
``calendar_tools.py``. Here, each tool gets the arguments the model may supply, plus
what only the session may decide: the student, the location, and the calendar.

After changing a tool, run ``python -m agentic.call_tool --write-schemas``, which
updates ``docs/agent/s5_agent_tools.json``. A test fails until you do.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from functools import lru_cache, partial
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, StructuredTool
from pydantic import Field

from agentic import calendar_tools, memory, places, tools, training_tools
from config.config import AppConfig
from src.generation.pipeline import PlanPipeline
from src.llm.ollama import OllamaGenerator
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_expansion import OllamaQueryExpander, QueryExpansionRetriever
from src.retrieval.semantic import SemanticRetriever, load_chunks
from src.retrieval.sparse import BM25Retriever
from src.schemas.models import Model, PlanOptions
from src.tools.generate_workout_plan import (
    GenerateWorkoutPlanResult,
    generate_workout_plan,
)

logger = logging.getLogger(__name__)

PLAN_TOOL = "generate_workout_plan"
FEEDBACK_TOOL = "save_workout_feedback"
SLOTS_TOOL = "propose_training_slots"
# These change the plan or the student's calendar, so they always need approval.
ALWAYS_NEEDS_APPROVAL = {
    "update_upcoming_workout": "This adjusts your next workout using your feedback.",
    "organize_schedule": "This moves workouts to other days in your plan.",
    "add_workouts_to_calendar": "This adds events to your calendar.",
}


@dataclass(frozen=True)
class AgentContext:
    """Trusted values from the signed-in session, never from the model."""

    student_id: str
    location: places.Location | None = None
    pipeline_factory: Callable[[], PlanPipeline] | None = None
    calendar: calendar_tools.CalendarProvider | None = None
    timezone: str = "Europe/Paris"
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)


# The arguments the model may supply to each tool.


class NoArgs(Model):
    """A tool the model calls without arguments."""


class WorkoutArgs(Model):
    """One workout, using the values from ``get_timetable``."""

    plan_id: str = Field(min_length=1, description="A plan_id from get_timetable.")
    week_number: int = Field(ge=1, description="A week_number from get_timetable.")
    workout_index: int = Field(ge=0, description="A workout_index from get_timetable.")


class FeedbackArgs(WorkoutArgs):
    completion_status: tools.CompletionStatus
    difficulty: tools.WorkoutDifficulty
    comment: str | None = Field(default=None, max_length=500)


class PerformanceArgs(WorkoutArgs):
    exercises: list[training_tools.PerformanceRecord] = Field(
        min_length=1, max_length=20, description="What the student did, per exercise."
    )


class UpdateWorkoutArgs(Model):
    plan_id: str = Field(min_length=1, description="A plan_id from get_timetable.")
    week_number: int = Field(ge=1, description="The logged workout's week_number.")
    workout_index: int = Field(
        ge=0, description="The logged workout's index, not the workout to change."
    )


class FindPlacesArgs(Model):
    category: places.PlaceCategory = Field(
        description="gym, sports_store (equipment), or supplement_store."
    )
    radius_km: int = Field(default=5, ge=1, le=20, description="Search radius in km.")


class PlanArgs(PlanOptions):
    """``PlanOptions``, but the graph sets the focus from the student's request."""

    focus: Annotated[str, InjectedToolArg] = Field(min_length=3, max_length=300)


class HistoryArgs(Model):
    max_plans: int = Field(default=5, ge=1, le=20, description="How many recent plans.")


class OrganizeScheduleArgs(Model):
    plan_id: str = Field(min_length=1, description="A plan_id from get_timetable.")
    moves: list[training_tools.WorkoutMove] = Field(min_length=1, max_length=21)


class RecommendationArgs(Model):
    plan_id: str | None = Field(default=None, description="Omit for the latest plan.")
    summary: str = Field(min_length=3, max_length=300, description="The proposed change.")
    reason: str = Field(min_length=3, max_length=500, description="Why it would help.")
    source_ids: list[str] = Field(
        default_factory=list, description="source_id values from search_documents."
    )


class RememberFactArgs(Model):
    fact: str = Field(
        min_length=3, max_length=300, description="The fact, as the student said it."
    )


class SearchArgs(Model):
    query: str = Field(min_length=3, max_length=300, description="What to look up.")
    top_k: int = Field(default=3, ge=1, le=5, description="How many passages.")


class CalendarBusyArgs(Model):
    start_date: date | None = Field(default=None, description="First day, or today.")
    days: int = Field(default=7, ge=1, le=31, description="How many days to read.")


class ProposeSlotsArgs(Model):
    week_start: date | None = Field(
        default=None, description="A date in the first week to schedule, if requested."
    )
    earliest: time = Field(default=time(7, 0), description="Earliest start, as HH:MM.")
    latest: time = Field(default=time(21, 0), description="Latest finish, as HH:MM.")


class AddToCalendarArgs(Model):
    plan_id: str = Field(min_length=1, description="The plan_id of the offered slots.")
    bookings: list[calendar_tools.Booking] = Field(min_length=1, max_length=21)


def build_plan_pipeline(config: AppConfig) -> PlanPipeline:
    """Build the same RAG pipeline as the API, from the local corpus and index."""
    if not config.chunk_file:
        raise FileNotFoundError("Set REPWISE_CHUNK_FILE to the prepared corpus.")
    chunks = load_chunks(Path(config.chunk_file).parent)
    if config.retrieval_strategy == "bm25":
        retriever = BM25Retriever(chunks)
    else:
        semantic = SemanticRetriever(
            chunks, config.embedding_index, config.embedding_model
        )
        retriever = (
            semantic
            if config.retrieval_strategy == "semantic"
            else HybridRetriever(BM25Retriever(chunks), semantic)
        )
    return PlanPipeline(
        QueryExpansionRetriever(
            retriever,
            OllamaQueryExpander(config.query_expansion_model, config.ollama_url),
            rewrite=config.query_expansion,
        ),
        OllamaGenerator(config.ollama_model, config.ollama_url),
        config.top_k,
    )


def _missing_pipeline() -> PlanPipeline:
    raise RuntimeError("Plan generation is not configured for this agent.")


def _tool(name: str, description: str, schema: type[Model], function) -> StructuredTool:
    """Make a tool the model can call. Its result reaches the model as JSON."""
    return StructuredTool.from_function(
        lambda **arguments: function(**arguments).model_dump_json(),
        name=name,
        description=description,
        args_schema=schema,
    )


def build_tools(context: AgentContext) -> list[StructuredTool]:
    """Create the model's tools, with the session's values already filled in."""
    student = context.student_id
    calendar = context.calendar
    timezone = context.timezone
    load_pipeline = lru_cache(maxsize=1)(context.pipeline_factory or _missing_pipeline)

    def with_pipeline(use: Callable[[PlanPipeline], Model]) -> Model:
        """Run a tool that needs the corpus or model, which may not be set up."""
        try:
            return use(load_pipeline())
        except (OSError, RuntimeError, ValueError) as error:
            logger.exception("Agent could not load the plan pipeline")
            return training_tools.Failure(code="services_unavailable", error=str(error))

    def create_plan(**arguments: Any) -> GenerateWorkoutPlanResult:
        try:
            return generate_workout_plan(
                PlanOptions.model_validate(arguments),
                student_id=student,
                repository=tools.open_repository(),
                pipeline=load_pipeline(),
            )
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
            logger.exception("Agent plan generation failed")
            return GenerateWorkoutPlanResult(
                status="unavailable",
                message=f"The plan could not be generated or saved: {error}",
            )

    def update_upcoming_workout(**arguments: Any) -> Model:
        return with_pipeline(
            lambda pipeline: training_tools.update_upcoming_workout(
                student, generator=pipeline.generator, **arguments
            )
        )

    def search_documents(query: str, top_k: int = 3) -> Model:
        # The pipeline's outer retriever refuses anything but plan requests, so use
        # the corpus search inside it.
        return with_pipeline(
            lambda pipeline: training_tools.search_documents(
                query,
                getattr(pipeline.retriever, "retriever", pipeline.retriever),
                top_k,
            )
        )

    def get_calendar_busy(**arguments: Any) -> Model:
        return calendar_tools.get_calendar_busy(
            calendar, timezone, context.clock(), **arguments
        )

    def propose_training_slots(**arguments: Any) -> Model:
        return calendar_tools.propose_training_slots(
            student, calendar, timezone, context.clock(), **arguments
        )

    def add_workouts_to_calendar(**arguments: Any) -> Model:
        return calendar_tools.add_workouts_to_calendar(
            student, calendar, timezone, context.clock(), **arguments
        )

    return [
        _tool(
            "get_user_profile",
            "Read the student's experience, goal, available days, session length, "
            "equipment, and preferences.",
            NoArgs,
            partial(tools.get_user_profile, student),
        ),
        _tool(
            "get_timetable",
            "Read the student's available days and the planned workouts of their "
            "latest plan, each with plan_id, week_number, workout_index, and saved "
            "feedback.",
            NoArgs,
            partial(tools.get_timetable, student),
        ),
        _tool(
            FEEDBACK_TOOL,
            "Save how one workout went. Call only when the student explicitly asks "
            "to log it.",
            FeedbackArgs,
            partial(tools.save_workout_feedback, student),
        ),
        _tool(
            "find_nearby_places",
            "Find gyms, sports equipment stores, or supplement stores near the "
            "student, nearest first.",
            FindPlacesArgs,
            partial(places.find_nearby_places, location=context.location),
        ),
        _tool(
            PLAN_TOOL,
            "Generate and save a one-week, evidence-based workout program from the "
            "student's request and saved profile.",
            PlanArgs,
            create_plan,
        ),
        _tool(
            "get_workout_history",
            "Read logged workouts, weights and reps, summary counts, and pending "
            "recommendations across the student's recent plans.",
            HistoryArgs,
            partial(training_tools.get_workout_history, student),
        ),
        _tool(
            "log_workout_performance",
            "Save the weights, sets, and reps the student reports for one workout's "
            "exercises.",
            PerformanceArgs,
            partial(training_tools.log_workout_performance, student),
        ),
        _tool(
            "update_upcoming_workout",
            "Adjust the next unlogged workout's sets, reps, and rest, using the saved "
            "feedback on a logged workout. Give the logged workout.",
            UpdateWorkoutArgs,
            update_upcoming_workout,
        ),
        _tool(
            "organize_schedule",
            "Move workouts to other available days in the saved plan. All the moves "
            "apply, or none do.",
            OrganizeScheduleArgs,
            partial(training_tools.organize_schedule, student),
        ),
        _tool(
            "save_recommendation",
            "Store a proposed change for later review, without changing the plan.",
            RecommendationArgs,
            partial(training_tools.save_recommendation, student),
        ),
        _tool(
            "remember_fact",
            "Remember a stable preference or useful fact about the student for later "
            "conversations. Call only when the student asks you to remember it, not "
            "to answer a question about what you already remember.",
            RememberFactArgs,
            partial(memory.remember_fact, student),
        ),
        _tool(
            "search_documents",
            "Search the fitness knowledge base for passages, with their sources.",
            SearchArgs,
            search_documents,
        ),
        _tool(
            "get_calendar_busy",
            "Answer whether the student is busy or free. Gives times only, not event "
            "details. To find training times, use propose_training_slots instead.",
            CalendarBusyArgs,
            get_calendar_busy,
        ),
        _tool(
            SLOTS_TOOL,
            "Find free calendar times for the latest plan's workouts that are not yet "
            "booked, ready to offer to the student. Books nothing.",
            ProposeSlotsArgs,
            propose_training_slots,
        ),
        _tool(
            "add_workouts_to_calendar",
            "Add workouts to the student's calendar at the times they chose.",
            AddToCalendarArgs,
            add_workouts_to_calendar,
        ),
    ]


def approval_reason(
    name: str, arguments: dict[str, Any], context: AgentContext
) -> str | None:
    """Say why a call needs the student's approval, or return None if it does not."""
    if name == PLAN_TOOL:
        timetable = tools.get_timetable(context.student_id)
        if timetable.ok and not timetable.planned_workouts:
            return None
        return "This generates a new plan, which replaces your latest plan as the active one."
    if name == FEEDBACK_TOOL:
        comment = arguments.get("comment")
        if training_tools.mentions_pain(comment):
            return f"This feedback mentions pain or injury: {comment!r}."
    return ALWAYS_NEEDS_APPROVAL.get(name)
