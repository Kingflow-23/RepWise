"""Connect each tool name in ``config/tools.json`` to its Python implementation.

``config/tools.json`` is the authoritative catalog: it holds each tool's
description, owner agents, read/write access, approval rule and error codes.
This module holds what only Python can: the arguments model the language model
may fill in, and the function to run. ``tests/tools/test_catalog.py`` fails if
the two disagree (a tool missing on either side, or a different input schema),
and ``python -m scripts.sync_tool_catalog`` rewrites the generated schema parts.

Arguments never include the student's ID, a database path, an API URL or a
location from the model: those come from the trusted :class:`ToolContext`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, time

from pydantic import Field, ValidationError, model_validator

from src.agents import memory
from src.schemas.models import CompletionStatus, Model, PlanOptions, WorkoutDifficulty
from src.tools import calendar, knowledge, places, planning, profile, training
from src.tools.catalog import ToolSpec, generated_parts, get_tool_spec, load_catalog
from src.tools.context import ToolContext
from src.tools.results import ToolFailure, failure

logger = logging.getLogger(__name__)


# --- The arguments the model may supply to each tool. ----------------------


class NoArguments(Model):
    """The tool takes no arguments: everything comes from the session."""


class WorkoutReference(Model):
    """One workout, using the values from ``get_timetable``."""

    week_number: int = Field(ge=1, description="A week_number from get_timetable.")
    workout_index: int = Field(ge=0, description="A workout_index from get_timetable.")
    plan_id: str | None = Field(
        default=None,
        description="A plan_id from get_timetable; omit for the current plan.",
    )


class FeedbackArguments(WorkoutReference):
    """Feedback on one workout."""

    completion_status: CompletionStatus = Field(
        description="completed, partially_completed, or not_completed for a missed workout."
    )
    difficulty: WorkoutDifficulty = Field(description="How the workout felt.")
    comment: str | None = Field(
        default=None, max_length=500, description="The student's own words, if any."
    )


class PerformanceArguments(WorkoutReference):
    """What the student did in one workout."""

    exercises: list[profile.PerformanceRecord] = Field(
        min_length=1, max_length=20, description="One record per exercise done."
    )


class LoggedWorkoutArguments(WorkoutReference):
    """The logged workout whose feedback adapts the next one."""


class HistoryArguments(Model):
    """How much history to read."""

    max_plans: int = Field(default=5, ge=1, le=20, description="How many recent plans.")


class PlanArguments(PlanOptions):
    """Options for a new one-week plan (the profile is read from the database)."""


class ActivatePlanArguments(Model):
    """The draft plan to activate."""

    plan_id: str = Field(min_length=1, description="The draft plan's plan_id.")


class ScheduleArguments(Model):
    """Workouts to move to other days: explicit moves, or a catch-up week."""

    moves: list[training.WorkoutMove] = Field(
        default_factory=list,
        max_length=21,
        description="Explicit moves. Leave empty when catch_up_week is given.",
    )
    catch_up_week: int | None = Field(
        default=None,
        ge=1,
        description=(
            "After missed workouts: the week to reorganise. Code puts the missed "
            "workouts first on the available days left; do not write moves."
        ),
    )
    from_day: str | None = Field(
        default=None,
        max_length=20,
        description="With catch_up_week: the first day that may be used, such as today's weekday.",
    )
    plan_id: str | None = Field(default=None, description="Omit for the current plan.")

    @model_validator(mode="after")
    def moves_or_catch_up(self) -> "ScheduleArguments":
        """Require explicit moves or a catch-up week, not both.

        Returns:
            The validated arguments.

        Raises:
            ValueError: If both or neither are given, or ``from_day`` comes
                without ``catch_up_week``.
        """
        if bool(self.moves) == (self.catch_up_week is not None):
            raise ValueError("Give either moves or catch_up_week.")
        if self.from_day and self.catch_up_week is None:
            raise ValueError("from_day only applies with catch_up_week.")
        return self


def prepare_schedule(
    context: ToolContext, arguments: ScheduleArguments
) -> ScheduleArguments | ToolFailure:
    """Turn a catch-up request into the exact moves, before any approval.

    The student then approves concrete moves, and the stored action (and its
    hash) holds exactly what will run.

    Args:
        context: Trusted tool context.
        arguments: Validated arguments.

    Returns:
        Arguments with explicit moves, or a :func:`training.catch_up_moves`
        failure.
    """
    if arguments.catch_up_week is None:
        return arguments
    moves = training.catch_up_moves(
        context, arguments.catch_up_week, arguments.from_day, arguments.plan_id
    )
    if isinstance(moves, ToolFailure):
        return moves
    return ScheduleArguments(moves=moves, plan_id=arguments.plan_id)


def describe_schedule(context: ToolContext, arguments: ScheduleArguments) -> str:
    """List the moves a schedule change makes, for the approval message.

    Args:
        context: Trusted tool context.
        arguments: Validated arguments with explicit moves.

    Returns:
        One sentence per move.
    """
    return training.describe_moves(context, arguments.moves, arguments.plan_id)


class RecommendationArguments(Model):
    """A proposed change to keep for later review."""

    summary: str = Field(
        min_length=3, max_length=300, description="The proposed change."
    )
    reason: str = Field(min_length=3, max_length=500, description="Why it would help.")
    source_ids: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="source_id values from search_documents.",
    )
    plan_id: str | None = Field(default=None, description="Omit for the current plan.")


class SearchArguments(Model):
    """A knowledge-base query."""

    query: str = Field(min_length=3, max_length=300, description="What to look up.")
    top_k: int = Field(default=3, ge=1, le=8, description="How many passages.")


class BusyArguments(Model):
    """Which days to read from the calendar."""

    start_date: date | None = Field(default=None, description="First day, or today.")
    days: int = Field(default=7, ge=1, le=31, description="How many days to read.")


class SlotArguments(Model):
    """Constraints for proposed training times."""

    week_start: date | None = Field(
        default=None,
        description="A date in the first week to schedule, if the student named one.",
    )
    earliest: time = Field(default=time(7, 0), description="Earliest start, as HH:MM.")
    latest: time = Field(default=time(21, 0), description="Latest finish, as HH:MM.")


class BookingArguments(Model):
    """Approved times to add to the calendar."""

    bookings: list[calendar.Booking] = Field(min_length=1, max_length=21)
    plan_id: str | None = Field(default=None, description="Omit for the current plan.")


class LocationArguments(Model):
    """Where to search, if the student named a place."""

    place: str | None = Field(
        default=None,
        max_length=200,
        description="A place name the student gave. Omit to use their own location.",
    )


class NearbyArguments(LocationArguments):
    """A nearby-place search."""

    category: places.PlaceCategory = Field(
        description="gym, sports_store (fitness equipment), or supplement_store."
    )
    radius_km: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Search radius in km; omit for the default.",
    )


class FactArguments(Model):
    """A preference or fact to remember."""

    fact: str = Field(
        min_length=3, max_length=300, description="The fact, as the student said it."
    )


# --- Implementations. -------------------------------------------------------


def feedback_needs_approval(context: ToolContext, arguments: Model) -> str | None:
    """Ask for approval before saving feedback that may describe pain or injury.

    Args:
        context: Trusted tool context (its health checker decides).
        arguments: Validated :class:`FeedbackArguments`.

    Returns:
        The reason shown to the student, or ``None`` when no approval is needed.
    """
    comment = getattr(arguments, "comment", None)
    if comment and context.health.check(comment).health_concern:
        return (
            "This feedback may describe pain or injury. It will be saved, but no "
            "workout will be adapted automatically. Please confirm it is accurate."
        )
    return None


@dataclass(frozen=True)
class ToolImplementation:
    """The Python side of one catalog tool.

    Attributes:
        arguments: Pydantic model of what the language model may supply.
        run: ``run(context, **arguments)`` returning a result model.
        outputs: Success result models (a :class:`ToolFailure` is always possible).
        approval_rule: For ``conditional`` tools, decides whether this call needs
            approval and why.
        prepare: Optional step that turns shorthand arguments into exact ones
            (or a failure) before the approval check.
        describe: Optional text added to the approval reason, saying exactly
            what this call will change.
    """

    arguments: type[Model]
    run: Callable[..., Model]
    outputs: tuple[type[Model], ...]
    approval_rule: Callable[[ToolContext, Model], str | None] | None = None
    prepare: Callable[[ToolContext, Model], Model | ToolFailure] | None = None
    describe: Callable[[ToolContext, Model], str] | None = None


IMPLEMENTATIONS: dict[str, ToolImplementation] = {
    "get_user_profile": ToolImplementation(
        NoArguments, profile.get_user_profile, (profile.ProfileResult,)
    ),
    "get_current_plan": ToolImplementation(
        NoArguments, profile.get_current_plan, (profile.CurrentPlanResult,)
    ),
    "get_timetable": ToolImplementation(
        NoArguments, profile.get_timetable, (profile.TimetableResult,)
    ),
    "get_workout": ToolImplementation(
        WorkoutReference, profile.get_workout, (profile.WorkoutResult,)
    ),
    "remember_fact": ToolImplementation(
        FactArguments, memory.remember_fact, (memory.FactResult,)
    ),
    "get_workout_history": ToolImplementation(
        HistoryArguments, training.get_workout_history, (training.HistoryResult,)
    ),
    "generate_workout_plan": ToolImplementation(
        PlanArguments,
        planning.generate_plan_for_agent,
        (planning.GenerateWorkoutPlanResult,),
    ),
    "activate_plan": ToolImplementation(
        ActivatePlanArguments, planning.activate_plan, (planning.ActivatePlanResult,)
    ),
    "save_recommendation": ToolImplementation(
        RecommendationArguments,
        training.save_recommendation,
        (training.RecommendationResult,),
    ),
    "save_workout_feedback": ToolImplementation(
        FeedbackArguments,
        training.save_workout_feedback,
        (training.FeedbackResult,),
        approval_rule=feedback_needs_approval,
    ),
    "log_workout_performance": ToolImplementation(
        PerformanceArguments,
        training.log_workout_performance,
        (training.PerformanceResult,),
    ),
    "update_upcoming_workout": ToolImplementation(
        LoggedWorkoutArguments,
        training.update_upcoming_workout,
        (training.UpdatedWorkoutResult,),
    ),
    "get_calendar_busy": ToolImplementation(
        BusyArguments, calendar.get_calendar_busy, (calendar.BusyResult,)
    ),
    "propose_training_slots": ToolImplementation(
        SlotArguments, calendar.propose_training_slots, (calendar.SlotsResult,)
    ),
    "organize_schedule": ToolImplementation(
        ScheduleArguments,
        training.organize_schedule,
        (training.ScheduleResult,),
        prepare=prepare_schedule,
        describe=describe_schedule,
    ),
    "add_workouts_to_calendar": ToolImplementation(
        BookingArguments, calendar.add_workouts_to_calendar, (calendar.BookingResult,)
    ),
    "get_location": ToolImplementation(
        LocationArguments, places.get_location, (places.LocationResult,)
    ),
    "search_nearby_places": ToolImplementation(
        NearbyArguments, places.search_nearby_places, (places.PlacesResult,)
    ),
    "search_documents": ToolImplementation(
        SearchArguments, knowledge.search_documents, (knowledge.SearchResult,)
    ),
}


def generated_catalog_parts(name: str) -> dict:
    """Return the catalog fields that code generates for one tool.

    Args:
        name: Tool name.

    Returns:
        ``input_schema`` (from the arguments model) and ``output_models``
        (success model names, then ``ToolFailure``).
    """
    tool = implementation(name)
    return generated_parts(
        tool.arguments.model_json_schema(),
        [model.__name__ for model in tool.outputs] + [ToolFailure.__name__],
    )


def catalog_problems() -> list[str]:
    """Compare ``config/tools.json`` with the implementations.

    Returns:
        One message per problem: a tool missing on either side, or a generated
        part that is out of date. An empty list means they agree.
    """
    catalog_names = [tool.name for tool in load_catalog().tools]
    problems = [
        f"{name} is in config/tools.json but has no implementation"
        for name in catalog_names
        if name not in IMPLEMENTATIONS
    ]
    problems += [
        f"{name} is implemented but missing from config/tools.json"
        for name in IMPLEMENTATIONS
        if name not in catalog_names
    ]
    for spec in load_catalog().tools:
        if spec.name not in IMPLEMENTATIONS:
            continue
        expected = generated_catalog_parts(spec.name)
        if spec.input_schema != expected["input_schema"]:
            problems.append(f"{spec.name}: input_schema is out of date")
        if spec.output_models != expected["output_models"]:
            problems.append(f"{spec.name}: output_models is out of date")
        if (
            spec.approval == "conditional"
            and IMPLEMENTATIONS[spec.name].approval_rule is None
        ):
            problems.append(f"{spec.name}: conditional approval needs an approval_rule")
    return problems


def implementation(name: str) -> ToolImplementation:
    """Return the implementation of a catalog tool.

    Args:
        name: Tool name from the catalog.

    Returns:
        Its implementation.

    Raises:
        KeyError: If no tool has this name.
    """
    if name not in IMPLEMENTATIONS:
        raise KeyError(f"Unknown tool {name!r}.")
    return IMPLEMENTATIONS[name]


def validate_arguments(name: str, arguments: dict | None) -> Model:
    """Check model-supplied arguments against the tool's arguments model.

    Args:
        name: Tool name.
        arguments: Raw arguments, as sent by the model or an MCP client.

    Returns:
        The validated arguments.

    Raises:
        KeyError: If the tool does not exist.
        pydantic.ValidationError: If the arguments are invalid.
    """
    return implementation(name).arguments.model_validate(arguments or {})


def approval_reason(context: ToolContext, name: str, arguments: Model) -> str | None:
    """Say why a call needs the student's approval, or ``None`` if it does not.

    The rule comes from the catalog: ``never``, ``always`` (with the catalog's
    reason), or ``conditional`` (decided by the tool's ``approval_rule``).

    Args:
        context: Trusted tool context.
        name: Tool name.
        arguments: Validated arguments.

    Returns:
        The reason to show the student, or ``None``.
    """
    spec: ToolSpec = get_tool_spec(name)
    tool = implementation(name)
    if spec.approval == "always":
        reason = spec.approval_reason
    elif spec.approval == "conditional":
        reason = (
            tool.approval_rule(context, arguments)
            if tool.approval_rule
            else spec.approval_reason
        )
    else:
        return None
    detail = tool.describe(context, arguments) if reason and tool.describe else ""
    return f"{reason} {detail}".strip() if reason else None


def prepare_arguments(
    context: ToolContext, name: str, arguments: Model
) -> Model | ToolFailure:
    """Run a tool's optional preparation step on validated arguments.

    Args:
        context: Trusted tool context.
        name: Tool name.
        arguments: Validated arguments.

    Returns:
        The exact arguments to approve or run, or a failure.
    """
    tool = implementation(name)
    return tool.prepare(context, arguments) if tool.prepare else arguments


def execute_tool(
    context: ToolContext, name: str, arguments: Model | dict | None
) -> Model:
    """Run one tool now, without any approval check.

    Only two callers may use this directly: the approval service after the
    student approved, and the approval gate for calls that need no approval.

    Args:
        context: Trusted tool context.
        name: Tool name.
        arguments: Validated arguments, or raw ones to validate here.

    Returns:
        The tool's result model, or a :class:`ToolFailure` for invalid
        arguments or an unexpected crash (which is logged in full).
    """
    try:
        validated = (
            arguments
            if isinstance(arguments, Model)
            else validate_arguments(name, arguments)
        )
    except ValidationError as error:
        return failure("invalid_arguments", str(error))
    except KeyError as error:
        return failure("unknown_tool", str(error))
    try:
        return implementation(name).run(context, **dict(validated))
    except (
        Exception
    ) as error:  # noqa: BLE001 - a tool bug must not end the conversation
        logger.exception("Tool crashed tool=%s", name)
        return ToolFailure(
            code="tool_crashed", error=f"The tool failed unexpectedly: {error}"
        )
