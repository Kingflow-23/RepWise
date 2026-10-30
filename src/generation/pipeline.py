"""The complete RAG pipeline, kept deliberately short and explicit."""

from __future__ import annotations

import re

from ..llm.ollama import PlanGenerator
from ..rag.context import make_query, select_evidence
from ..rag.prompt import load_system_prompt, make_user_prompt
from ..retrieval.base import Retriever
from ..schemas.models import PlanRequest, PlanResult, TrainingPlan


class PlanPipeline:
    """Connect retrieval, evidence selection, Ollama and local validation."""

    def __init__(self, retriever: Retriever, generator: PlanGenerator, top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def create_plan(self, request: PlanRequest) -> PlanResult:
        """Create one grounded plan, or return a clear status when this fails."""
        try:
            chunks = self.retriever.search(make_query(request), self.top_k)
        except Exception as error:
            return PlanResult(
                status="retrieval_error",
                message=(
                    "Retrieval failed. Check that the knowledge base corpus and configured index are "
                    f"available. Details: {error}"
                ),
            )

        evidence, sources = select_evidence(chunks)
        if not evidence:
            return PlanResult(
                status="insufficient_evidence",
                message="No useful training evidence was found.",
            )

        system_prompt = load_system_prompt()
        user_prompt = make_user_prompt(request, evidence)
        plan = None
        last_error: Exception | None = None
        for attempt in range(
            3
        ):  # Allow two correction attempts after the initial generation
            try:
                if attempt:
                    user_prompt += (
                        "\n\nCorrection request: the previous response failed validation: "
                        f"{last_error}. Return corrected JSON only."
                    )
                plan = self.generator.generate(system_prompt, user_prompt)
                validate_plan(plan, request, {source.source_id for source in sources})
                break
            except (OSError, ValueError, RuntimeError) as error:
                last_error = error
        else:
            assert last_error is not None
            return PlanResult(
                status="generation_error",
                message=f"The plan could not be validated after two correction attempts: {last_error}",
                sources=sources,
            )

        return PlanResult(
            status="completed",
            message="Plan created from retrieved sources.",
            plan=plan,
            sources=sources,
        )


def validate_plan(
    plan: TrainingPlan,
    request: PlanRequest,
    source_ids: set[str],
    *,
    require_citations: bool = True,
) -> None:
    """Check the plan against the student request and supplied evidence."""
    if len(plan.weeks) != request.weeks:
        raise ValueError("the number of weeks does not match the request")
    if {week.number for week in plan.weeks} != set(range(1, request.weeks + 1)):
        raise ValueError("weeks must be numbered from 1 to the requested duration")

    available_days = set(request.profile.available_days)
    equipment_aliases = {
        "barbell": {"barbell", "barbells", "barre"},
        "dumbbell": {
            "dumbbell",
            "dumbbells",
            "halt\u00e8re",
            "halt\u00e8res",
            "haltere",
            "halteres",
        },
        "kettlebell": {"kettlebell", "kettlebells"},
        "machine": {"machine", "machines"},
        "cable": {"cable", "cables", "poulie", "poulies"},
        "bench": {"bench", "benches", "banc"},
        "pull-up bar": {"pull-up bar", "barre de traction"},
        "resistance band": {
            "resistance band",
            "resistance bands",
            "\u00e9lastique",
            "\u00e9lastiques",
            "elastique",
            "elastiques",
        },
    }

    def canonical_equipment(items: list[str]) -> set[str]:
        result = set()
        for item in items:
            folded = item.casefold()
            result.add(
                next(
                    (
                        canonical
                        for canonical, aliases in equipment_aliases.items()
                        if folded in aliases
                    ),
                    folded,
                )
            )
        return result

    available_equipment = canonical_equipment(request.profile.equipment)
    for week in plan.weeks:
        if len(week.workouts) > len(available_days):
            raise ValueError("a week has more workouts than available days")
        workout_days = [workout.day for workout in week.workouts]
        if len(workout_days) != len(set(workout_days)):
            raise ValueError("a week cannot schedule more than one workout per day")
        for workout in week.workouts:
            if workout.day not in available_days:
                raise ValueError(f"{workout.day} is not an available day")
            if workout.estimated_minutes > request.profile.session_minutes:
                raise ValueError(
                    "a workout is longer than the requested session duration"
                )
            for exercise in workout.exercises:
                required_equipment = canonical_equipment(exercise.equipment)
                name = exercise.name.casefold()
                for equipment, terms in equipment_aliases.items():
                    if any(re.search(rf"\b{re.escape(term)}\b", name) for term in terms):
                        required_equipment.add(equipment)
                unavailable = required_equipment - available_equipment - {"bodyweight"}
                if unavailable and "full gym" not in available_equipment:
                    raise ValueError(
                        "an exercise requires equipment that is not in the student profile"
                    )
                if require_citations and not exercise.source_ids:
                    raise ValueError(
                        "an exercise must cite at least one supplied source"
                    )
                if require_citations and set(exercise.source_ids) - source_ids:
                    raise ValueError(
                        "an exercise cites a source that was not retrieved"
                    )
