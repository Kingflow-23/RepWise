"""The complete RAG pipeline, kept deliberately short and explicit."""

from __future__ import annotations

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
        except Exception:
            return PlanResult(
                status="retrieval_error",
                message="Retrieval failed. Check that the knowledge base is available.",
            )

        evidence, sources = select_evidence(chunks)
        if not evidence:
            return PlanResult(
                status="insufficient_evidence",
                message="No useful training evidence was found.",
            )

        try:
            plan = self.generator.generate(
                load_system_prompt(), make_user_prompt(request, evidence)
            )
            validate_plan(plan, request, {source.source_id for source in sources})
        except (OSError, ValueError, RuntimeError) as error:
            return PlanResult(
                status="generation_error",
                message=f"The plan could not be validated: {error}",
                sources=sources,
            )

        return PlanResult(
            status="completed",
            message="Plan created from retrieved sources.",
            plan=plan,
            sources=sources,
        )


def validate_plan(
    plan: TrainingPlan, request: PlanRequest, source_ids: set[str]
) -> None:
    """Check the plan against the student request and supplied evidence."""
    if len(plan.weeks) != request.weeks:
        raise ValueError("the number of weeks does not match the request")
    if {week.number for week in plan.weeks} != set(range(1, request.weeks + 1)):
        raise ValueError("weeks must be numbered from 1 to the requested duration")

    available_days = set(request.profile.available_days)
    for week in plan.weeks:
        if len(week.workouts) > len(available_days):
            raise ValueError("a week has more workouts than available days")
        for workout in week.workouts:
            if workout.day not in available_days:
                raise ValueError(f"{workout.day} is not an available day")
            if workout.estimated_minutes > request.profile.session_minutes:
                raise ValueError(
                    "a workout is longer than the requested session duration"
                )
            for exercise in workout.exercises:
                if set(exercise.source_ids) - source_ids:
                    raise ValueError(
                        "an exercise cites a source that was not retrieved"
                    )
