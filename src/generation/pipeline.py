"""The complete RAG pipeline, kept deliberately short and explicit."""

from __future__ import annotations

import logging
import json
import re
from time import perf_counter

from ..llm.ollama import PlanGenerator
from ..rag.context import make_highlight_query, make_query, select_evidence
from ..rag.prompt import (
    estimate_prompt_tokens,
    load_system_prompt,
    make_user_prompt,
    plan_response_schema,
)
from ..retrieval.base import QueryAwareRetriever, Retriever
from ..retrieval.query_expansion import ScopeRefusalError
from ..schemas.models import PlanRequest, PlanResult, TrainingPlan

logger = logging.getLogger(__name__)
MINIMUM_RETRIEVED_CHUNKS = 5


class PlanPipeline:
    """Connect retrieval, evidence selection, generation, and local validation.

    Retrieval is configured to request at least five chunks, and generation is
    stopped when fewer than five are returned so the model has a minimum evidence
    base.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: PlanGenerator,
        top_k: int = 5,
        max_generation_attempts: int = 3,
    ):
        """Store the services used to create plans.

        Args:
            retriever: Service that searches the prepared training corpus.
            generator: Service that returns structured training plans.
            top_k: Preferred retrieval count; values below five are raised to five.
            max_generation_attempts: Maximum number of generation attempts,
                including the initial call, between one and three.

        Raises:
            ValueError: If ``max_generation_attempts`` is not an integer from
                one to three.
        """
        if (
            isinstance(max_generation_attempts, bool)
            or not isinstance(max_generation_attempts, int)
            or not 1 <= max_generation_attempts <= 3
        ):
            raise ValueError("max_generation_attempts must be an integer from 1 to 3")
        self.retriever = retriever
        self.generator = generator
        self.top_k = max(MINIMUM_RETRIEVED_CHUNKS, top_k)
        self.max_generation_attempts = max_generation_attempts

    def create_plan(
        self,
        request: PlanRequest,
        *,
        previous_plan: dict[str, object] | None = None,
    ) -> PlanResult:
        """Retrieve evidence, generate a plan, and validate it against the request.

        Args:
            self: Pipeline holding the configured retrieval and generation
                services and the minimum retrieval count.
            request: Validated profile and plan requirements for this generation.
            previous_plan: Latest saved request and plan for this student, if any.

        Returns:
            A result with status ``completed`` and the plan when validation
            succeeds; otherwise a result with an explanatory status such as
            ``out_of_scope``, ``retrieval_error``, ``insufficient_evidence``, or
            ``generation_error``. Retrieved sources are included when generation
            was attempted.

        Notes:
            Generation is attempted up to ``max_generation_attempts`` times,
            with correction prompts after validation or provider errors.
        """
        logger.info(
            "Plan generation started weeks=%s top_k=%s",
            request.weeks,
            self.top_k,
        )
        retrieval_started = perf_counter()
        query = make_query(request)
        retrieval_query = query
        try:
            if isinstance(self.retriever, QueryAwareRetriever):
                chunks, retrieval_query = self.retriever.search_with_query(
                    query, self.top_k
                )
            else:
                chunks = self.retriever.search(query, self.top_k)
        except ScopeRefusalError as error:
            logger.info(
                "Plan request declined by scope gate scope=%s message=%s",
                error.scope,
                error.user_message,
            )
            return PlanResult(
                status="out_of_scope",
                message=error.user_message,
            )
        except Exception as error:
            logger.exception(
                "Plan retrieval failed elapsed_ms=%.1f",
                (perf_counter() - retrieval_started) * 1000,
            )
            return PlanResult(
                status="retrieval_error",
                message=(
                    "Retrieval failed. Check that the knowledge base corpus and configured index are "
                    f"available. Details: {error}"
                ),
            )

        logger.info(
            "Plan evidence retrieved chunks=%s elapsed_ms=%.1f",
            len(chunks),
            (perf_counter() - retrieval_started) * 1000,
        )
        if len(chunks) < MINIMUM_RETRIEVED_CHUNKS:
            logger.warning(
                "Plan generation stopped reason=insufficient_retrieval_count "
                "retrieved=%s minimum=%s",
                len(chunks),
                MINIMUM_RETRIEVED_CHUNKS,
            )
            return PlanResult(
                status="insufficient_evidence",
                message=(
                    f"Retrieval returned {len(chunks)} chunks; at least "
                    f"{MINIMUM_RETRIEVED_CHUNKS} are required."
                ),
            )

        evidence, sources = select_evidence(
            chunks,
            query=make_highlight_query(request, retrieval_query=retrieval_query),
        )
        if not evidence:
            logger.warning("Plan generation stopped reason=no_usable_evidence")
            return PlanResult(
                status="insufficient_evidence",
                message="No useful training evidence was found.",
            )

        system_prompt = load_system_prompt()
        user_prompt = make_user_prompt(request, evidence, previous_plan=previous_plan)
        response_schema = plan_response_schema(
            request, [source.source_id for source in sources]
        )
        plan = None
        last_candidate_plan = None
        last_candidate_attempt = None
        last_raw_output = None
        last_error: Exception | None = None
        last_error_stage = None
        for attempt in range(self.max_generation_attempts):
            generation_started = perf_counter()
            error_stage = "generation"
            try:
                if attempt:
                    logger.warning(
                        "Retrying plan generation attempt=%s reason=%s",
                        attempt + 1,
                        type(last_error).__name__ if last_error else "unknown",
                    )
                    user_prompt += (
                        "\n\nCorrection request: the previous response failed validation: "
                        f"{last_error}. Return corrected JSON only. Keep the plan "
                        "in clear English, interpret any source language as needed, "
                        "and do not invent source IDs."
                    )
                logger.info(
                    "Model prompt measured attempt=%s system_chars=%s user_chars=%s "
                    "schema_chars=%s estimated_input_tokens=%s previous_plan=%s",
                    attempt + 1,
                    len(system_prompt),
                    len(user_prompt),
                    len(
                        json.dumps(
                            response_schema,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                    estimate_prompt_tokens(system_prompt, user_prompt, response_schema),
                    previous_plan is not None,
                )
                plan = self.generator.generate(
                    system_prompt, user_prompt, response_schema
                )
                last_candidate_plan = plan
                last_candidate_attempt = attempt + 1
                last_raw_output = getattr(self.generator, "_last_raw_output", None)
                error_stage = "validation"
                logger.info(
                    "Model returned plan attempt=%s elapsed_ms=%.1f",
                    attempt + 1,
                    (perf_counter() - generation_started) * 1000,
                )
                validate_plan(
                    plan,
                    request,
                    {source.source_id for source in sources},
                    source_evidence={
                        source.source_id: source.evidence_text or ""
                        for source in sources
                    },
                )
                break
            except (OSError, ValueError, RuntimeError) as error:
                last_error = error
                last_error_stage = error_stage
                last_raw_output = getattr(self.generator, "_last_raw_output", None)
                logger.warning(
                    "Generated plan rejected attempt=%s error_type=%s",
                    attempt + 1,
                    type(error).__name__,
                )
        else:
            assert last_error is not None
            logger.error("Plan generation failed after retries")
            result = PlanResult(
                status="generation_error",
                message=(
                    "The plan could not be generated or validated after "
                    f"{self.max_generation_attempts} attempt(s); last error during "
                    f"{last_error_stage}: {last_error}"
                ),
                sources=sources,
            )
            result._evaluation_diagnostics = {
                "error_stage": last_error_stage,
                "last_error": f"{type(last_error).__name__}: {last_error}",
                "candidate_attempt": last_candidate_attempt,
                "candidate_plan": (
                    last_candidate_plan.model_dump(mode="json")
                    if last_candidate_plan is not None
                    else None
                ),
                "raw_model_output": last_raw_output,
            }
            return result

        logger.info(
            "Plan generation completed weeks=%s sources=%s",
            len(plan.weeks),
            len(sources),
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
    source_evidence: dict[str, str] | None = None,
    require_citations: bool = True,
) -> None:
    """Validate schedule, equipment, and citations against the request and sources.

    Args:
        plan: Generated plan to check.
        request: Student requirements against which each week and workout is checked.
        source_ids: IDs of sources supplied to the model.
        source_evidence: Full text for each source, retained for compatibility
            with callers. Exact quote matching is not required.
        require_citations: Whether each exercise must cite at least one supplied
            source and use only IDs from the retrieved evidence.

    Raises:
        ValueError: If the plan has the wrong number or numbering of weeks,
            incorrect workout counts, unavailable or duplicate workout days,
            overlong sessions, non-specific or repeated exercise names, unavailable
            equipment, or missing/unknown source IDs. Supporting quotes are not
            checked for exact text matches.
    """
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
        """Map recognized equipment aliases to canonical names.

        Args:
            items: Equipment names supplied by a profile or exercise.

        Returns:
            A set with known English/French aliases normalized and unknown names
            preserved in case-folded form.
        """
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
    non_exercise_labels = {
        "back",
        "chest",
        "core",
        "full body",
        "legs",
        "lower body",
        "push pull legs",
        "upper body",
    }
    for week in plan.weeks:
        if len(week.workouts) != request.target_workouts_per_week:
            raise ValueError(
                "each week must have exactly the requested number of workouts"
            )
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
            exercise_names: set[str] = set()
            for exercise in workout.exercises:
                normalized_name = " ".join(
                    re.findall(r"[a-z0-9]+", exercise.name.casefold())
                )
                if normalized_name in non_exercise_labels:
                    raise ValueError(
                        "exercise names must identify a specific movement, not a body part or workout split"
                    )
                if normalized_name in exercise_names:
                    raise ValueError("a workout cannot repeat the same exercise")
                exercise_names.add(normalized_name)
                required_equipment = canonical_equipment(exercise.equipment)
                name = exercise.name.casefold()
                for equipment, terms in equipment_aliases.items():
                    if any(
                        re.search(rf"\b{re.escape(term)}\b", name) for term in terms
                    ):
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
