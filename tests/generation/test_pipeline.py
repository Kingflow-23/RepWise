"""Pipeline tests for retrieval context, plan validation, and generation."""

import pytest

from src.generation import PlanPipeline, validate_plan
from src.rag import make_highlight_query, make_query, select_evidence
from src.rag.prompt import (
    PREVIOUS_PLAN_CONTEXT_MAX_CHARS,
    compact_previous_plan,
    make_user_prompt,
    plan_response_schema,
)
from src.retrieval import BM25Retriever
from src.retrieval.query_expansion import ScopeRefusalError
from src.schemas import PlanRequest, RetrievedChunk, TrainingPlan, UserProfile


def request() -> PlanRequest:
    """Build the standard one-week request used by plan pipeline tests."""
    return PlanRequest(
        focus="Build abdominal strength",
        profile=UserProfile(
            experience="beginner",
            goal="get stronger",
            available_days=["Monday"],
            session_minutes=45,
            equipment=["bodyweight"],
        ),
        weeks=1,
    )


def chunk() -> RetrievedChunk:
    """Build one timestamped evidence chunk with a traceable source passage."""
    return RetrievedChunk(
        chunk_id="video-1_chunk",
        text="For hypertrophy, use a range from 6 to 20 repetitions.",
        video_id="video-1",
        passage_ids=["video-1_p001"],
        title="Abdominal training",
        video_url="https://example.com",
        start_url="https://example.com?t=10",
    )


def valid_plan() -> TrainingPlan:
    """Return a valid one-week plan citing the standard evidence chunk."""
    return TrainingPlan.model_validate(
        {
            "title": "Simple abdominal plan",
            "summary": "One focused session.",
            "assumptions": [],
            "safety_note": "Stop if an exercise causes pain.",
            "weeks": [
                {
                    "number": 1,
                    "objective": "Learn the movement.",
                    "workouts": [
                        {
                            "day": "Monday",
                            "focus": "Abdominals",
                            "estimated_minutes": 35,
                            "exercises": [
                                {
                                    "name": "Crunch",
                                    "sets": 3,
                                    "repetitions": "8-12",
                                    "rest_seconds": 60,
                                    "estimated_minutes": 10,
                                    "coaching_note": "Move slowly.",
                                    "source_ids": ["video-1_p001"],
                                    "supporting_evidence": [
                                        {
                                            "source_id": "video-1_p001",
                                            "quote": "For hypertrophy, use a range from 6 to 20 repetitions.",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )


def test_context_keeps_traceable_training_chunk():
    """Retain the retrieved passage ID and timestamped source URL in context."""
    evidence, sources = select_evidence([chunk()])
    assert "SOURCE video-1_p001" in evidence
    assert sources[0].url.endswith("?t=10")


def test_context_keeps_full_evidence_and_marks_query_word_matches():
    """Preserve complete evidence text while recording matching query terms."""
    item = chunk()
    item.text = "Abdominal strength comes from controlled repetitions. " + "extra " * 80

    evidence, sources = select_evidence([item], query="Build abdominal strength")

    assert sources[0].evidence_text == " ".join(item.text.split())
    assert sources[0].matched_terms == ["Abdominal", "strength"]
    assert sources[0].excerpt == sources[0].evidence_text[:300].rstrip()
    assert sources[0].evidence_text in evidence


def test_context_passes_all_chunks_even_when_total_evidence_exceeds_old_limit():
    """Include every retrieved chunk even when combined evidence is large."""
    oversized = chunk()
    oversized.text = "x" * 8_100
    second = chunk().model_copy(
        update={
            "video_id": "video-2",
            "chunk_id": "video-2_chunk",
            "passage_ids": ["video-2_p001"],
        }
    )
    evidence, sources = select_evidence([oversized, second])
    assert "SOURCE video-1_p001" in evidence
    assert "SOURCE video-2_p001" in evidence
    assert len(sources) == 2
    assert len(evidence) > 8_000


def test_highlights_accent_and_plural_variants_of_request_intent():
    """Highlight intent terms across accent and inflection variants, not profile values."""
    training_request = request().model_copy(
        update={
            "focus": "Renforcer muscles et mobilité",
            "profile": request().profile.model_copy(
                update={
                    "goal": "improve strength",
                    "experience": "beginner",
                    "equipment": ["bodyweight"],
                }
            ),
            "extra_constraints": ["more repetitions"],
        }
    )
    item = chunk()
    item.text = "Les répétitions contrôlées renforcent les muscles et les tendons."

    _, sources = select_evidence([item], query=make_highlight_query(training_request))

    assert sources[0].matched_terms == ["répétitions", "muscles"]
    assert "beginner" not in sources[0].matched_terms
    assert "bodyweight" not in sources[0].matched_terms


def test_query_contains_current_goal_and_profile_information():
    """Build a retrieval query from the current focus and profile."""
    query = make_query(request())
    assert "Build abdominal strength" in query
    assert "Experience level: beginner" in query
    assert "Goal: get stronger" in query


def test_query_includes_profile_preferences_and_refinement_instructions():
    """Use saved preferences and requested plan changes to guide retrieval."""
    training_request = request().model_copy(
        update={
            "profile": request().profile.model_copy(
                update={"preferences": "prefer low-impact movements"}
            ),
            "refinement_instructions": "replace crunches with a cable exercise",
        }
    )

    query = make_query(training_request)

    assert "Training preferences: prefer low-impact movements" in query
    assert "Requested plan changes: replace crunches with a cable exercise" in query


def test_highlighting_uses_effective_query_and_user_profile_terms():
    """Highlight rewritten French terms alongside profile and refinement intent."""
    training_request = request().model_copy(
        update={
            "profile": request().profile.model_copy(
                update={
                    "preferences": "prefer cable exercises",
                    "equipment": ["cables"],
                }
            ),
            "refinement_instructions": "replace crunches",
        }
    )
    item = chunk()
    item.text = (
        "Des exercices avec câbles remplacent les crunchs pour renforcer la sangle."
    )

    _, sources = select_evidence(
        [item],
        query=make_highlight_query(
            training_request,
            retrieval_query="renforcer la sangle abdominale avec câbles, crunchs",
        ),
    )

    assert "câbles" in sources[0].matched_terms
    assert "crunchs" in sources[0].matched_terms
    assert "renforcer" in sources[0].matched_terms


def test_query_expands_french_body_region_to_canonical_retrieval_tag():
    """Map a French body-region request to its canonical retrieval tag."""
    training_request = request().model_copy(update={"focus": "Renforcer les jambes"})

    query = make_query(training_request)

    assert "quadriceps" not in query
    assert "body parts: legs" in query


def test_validator_rejects_unknown_source():
    """Reject plans that cite a source ID absent from retrieved evidence."""
    plan = valid_plan()
    plan.weeks[0].workouts[0].exercises[0].source_ids = ["unknown"]
    try:
        validate_plan(
            plan,
            request(),
            {"video-1_p001"},
            source_evidence={"video-1_p001": chunk().text},
        )
    except ValueError as error:
        assert "was not retrieved" in str(error)
    else:
        raise AssertionError("Expected validation to fail")


def test_validator_rejects_duplicate_days_and_undeclared_named_equipment():
    """Reject workouts on duplicate days and equipment omitted from an exercise."""
    plan = valid_plan()
    plan.weeks[0].workouts[0].exercises[0].name = "Barbell squat"
    plan.weeks[0].workouts[0].exercises[0].equipment = []
    try:
        validate_plan(
            plan,
            request(),
            {"video-1_p001"},
            source_evidence={"video-1_p001": chunk().text},
        )
    except ValueError as error:
        assert "equipment" in str(error)
    else:
        raise AssertionError("Expected equipment validation to fail")

    plan = valid_plan()
    plan.weeks[0].workouts.append(plan.weeks[0].workouts[0].model_copy(deep=True))
    training_request = request().model_copy(
        update={
            "workouts_per_week": 2,
            "profile": request().profile.model_copy(
                update={"available_days": ["Monday", "Wednesday"]}
            ),
        }
    )
    try:
        validate_plan(
            plan,
            training_request,
            {"video-1_p001"},
            source_evidence={"video-1_p001": chunk().text},
        )
    except ValueError as error:
        assert "one workout per day" in str(error)
    else:
        raise AssertionError("Expected duplicate workout day validation to fail")


def test_validator_rejects_equipment_not_in_profile():
    """Reject exercise equipment unavailable in the user's profile."""
    plan = valid_plan()
    plan.weeks[0].workouts[0].exercises[0].equipment = ["barbell"]
    try:
        validate_plan(
            plan,
            request(),
            {"video-1_p001"},
            source_evidence={"video-1_p001": chunk().text},
        )
    except ValueError as error:
        assert "equipment" in str(error)
    else:
        raise AssertionError("Expected equipment validation to fail")


def test_validator_rejects_generic_and_repeated_exercise_names():
    """Require specific exercise names and prevent duplicate movements."""
    plan = valid_plan()
    plan.weeks[0].workouts[0].exercises[0].name = "Full Body"
    with pytest.raises(ValueError, match="specific movement"):
        validate_plan(
            plan,
            request(),
            {"video-1_p001"},
            source_evidence={"video-1_p001": chunk().text},
        )

    plan = valid_plan()
    plan.weeks[0].workouts[0].exercises.append(
        plan.weeks[0].workouts[0].exercises[0].model_copy(deep=True)
    )
    with pytest.raises(ValueError, match="cannot repeat"):
        validate_plan(
            plan,
            request(),
            {"video-1_p001"},
            source_evidence={"video-1_p001": chunk().text},
        )


def test_validator_does_not_require_exact_quotes_across_source_languages():
    """Accept an unmatched quote while still validating source IDs."""
    plan = valid_plan()
    plan.weeks[0].workouts[0].exercises[0].supporting_evidence[
        0
    ].quote = "This quote is absent from the source."

    validate_plan(
        plan,
        request(),
        {"video-1_p001"},
        source_evidence={"video-1_p001": chunk().text},
    )


class FakeRetriever:
    """Return five variations of the standard evidence chunk for pipeline tests."""

    def search(self, query, top_k):
        """Supply uniquely identified copies of the shared test chunk.

        Args:
            query: Query produced from the test request; the fixed fixture does
                not need to inspect it.
            top_k: Requested retrieval depth; the fake supplies the five chunks
                needed by the pipeline's evidence minimum.
        """
        return [
            chunk().model_copy(
                update={
                    "video_id": f"video-{index}",
                    "chunk_id": f"video-{index}_chunk",
                    "passage_ids": [f"video-{index}_p001"],
                }
            )
            for index in range(1, 6)
        ]


class FakeGenerator:
    """Return a valid fixed plan without invoking a language model."""

    def generate(self, system_prompt, user_prompt, response_schema=None):
        """Provide the standard plan expected by pipeline assertions.

        Args:
            system_prompt: System instruction text accepted by the generator
                protocol but unused by this deterministic fake.
            user_prompt: Request and evidence text accepted but not inspected
                in this general pipeline test.
            response_schema: Structured-output schema accepted by the protocol.
        """
        return valid_plan()


def test_pipeline_returns_valid_plan_with_mocked_dependencies():
    """Complete plan generation with deterministic retrieval and generation doubles."""
    result = PlanPipeline(FakeRetriever(), FakeGenerator()).create_plan(request())
    assert result.status == "completed"
    assert result.plan is not None
    assert result.sources[0].evidence_text == chunk().text


def test_pipeline_returns_scope_message_without_calling_generator():
    """Return user-facing scope guidance rather than a retrieval error."""

    class RefusingRetriever:
        def search(self, query, top_k):
            raise ScopeRefusalError(
                "out_of_scope",
                "RepWise cannot diagnose pain. Please consult a health professional.",
            )

    class UnusedGenerator:
        def generate(self, *_args, **_kwargs):
            pytest.fail("generation must not run after scope refusal")

    result = PlanPipeline(RefusingRetriever(), UnusedGenerator()).create_plan(request())

    assert result.status == "out_of_scope"
    assert result.message == (
        "RepWise cannot diagnose pain. Please consult a health professional."
    )
    assert result.plan is None


def test_pipeline_highlights_the_exact_rewritten_query_used_for_retrieval():
    """Keep translated retrieval terms available to evidence relevance review."""

    class QueryAwareRetriever:
        def search(self, query, top_k):
            raise AssertionError("query-aware retrieval must not search twice")

        def search_with_query(self, query, top_k):
            assert "Requested plan changes:" in query
            chunks = [
                chunk().model_copy(
                    update={
                        "chunk_id": f"video-{index}_chunk",
                        "video_id": f"video-{index}",
                        "passage_ids": [f"video-{index}_p001"],
                        "text": "Repos entre les séries pour renforcer la sangle.",
                    }
                )
                for index in range(1, 6)
            ]
            return chunks, "repos entre les séries, renforcer la sangle abdominale"

    training_request = request().model_copy(
        update={"refinement_instructions": "replace crunches"}
    )
    result = PlanPipeline(QueryAwareRetriever(), FakeGenerator()).create_plan(
        training_request
    )

    assert result.status == "completed"
    matched_terms = {term.casefold() for term in result.sources[0].matched_terms}
    assert {"repos", "séries", "renforcer", "sangle"}.issubset(matched_terms)


def test_pipeline_does_not_generate_when_retrieval_returns_fewer_than_five():
    """Stop with insufficient evidence before calling the generator.

    The nested retriever and generator track whether the minimum-evidence
    gate runs before generation.
    """

    class ThreeChunkRetriever:
        """Return fewer passages than the pipeline's required evidence minimum."""

        def search(self, query, top_k):
            """Return only three passages, below the pipeline's minimum.

            Args:
                query: Generated search query; irrelevant to the fixed fixture.
                top_k: Requested depth; intentionally ignored to simulate a
                    retriever that cannot supply enough evidence.
            """
            return [chunk()] * 3

    class UnusedGenerator:
        """Record whether generation was called despite insufficient evidence."""

        called = False

        def generate(self, system_prompt, user_prompt, response_schema=None):
            """Mark an unexpected generation attempt and return a valid plan.

            Args:
                system_prompt: Generator protocol input, unused because this
                    method should not be reached.
                user_prompt: Generator protocol input, unused for the same
                    reason.
                response_schema: Generator protocol input, unused for the same
                    reason.
            """
            self.called = True
            return valid_plan()

    generator = UnusedGenerator()
    result = PlanPipeline(ThreeChunkRetriever(), generator).create_plan(request())

    assert result.status == "insufficient_evidence"
    assert "at least 5" in result.message
    assert not generator.called


def test_pipeline_sends_all_five_retrieved_chunks_to_generator():
    """Pass all five retrieved passages into the generator prompt.

    Nested doubles expose the requested retrieval depth and captured prompt so
    the test can ensure no evidence was dropped.
    """
    chunks = [
        chunk().model_copy(
            update={
                "video_id": f"video-{index}",
                "chunk_id": f"video-{index}_chunk",
                "passage_ids": [f"video-{index}_p001"],
                "text": (
                    "Crunch uses controlled repetitions and full range of motion."
                    if index == 1
                    else f"Supporting passage number {index} with enough information."
                ),
            }
        )
        for index in range(1, 6)
    ]

    class FiveChunkRetriever:
        """Return the full evidence set and retain the requested top-k value."""

        requested_top_k = None

        def search(self, query, top_k):
            """Record the retrieval depth and return all prepared chunks.

            Args:
                query: Generated request query; captured behavior is independent
                    of its wording.
                top_k: Requested depth, retained to assert the minimum-five
                    configuration.
            """
            self.requested_top_k = top_k
            return chunks

    class PromptCapturingGenerator:
        """Capture the prompt and return a plan quoting the first retrieved chunk."""

        user_prompt = ""

        def generate(self, system_prompt, user_prompt, response_schema=None):
            """Retain the prompt and create a valid evidence-backed plan.

            Args:
                system_prompt: System instructions accepted by the generator
                    protocol.
                user_prompt: Full request and evidence prompt captured by the
                    test.
                response_schema: Structured-output schema accepted by the
                    protocol.
            """
            self.user_prompt = user_prompt
            plan = valid_plan()
            plan.weeks[0].workouts[0].exercises[0].supporting_evidence[0].quote = (
                chunks[0].text
            )
            return plan

    retriever = FiveChunkRetriever()
    generator = PromptCapturingGenerator()
    result = PlanPipeline(retriever, generator, top_k=3).create_plan(request())

    assert result.status == "completed"
    assert retriever.requested_top_k == 5
    assert len(result.sources) == 5
    assert "Crunch uses controlled repetitions" in generator.user_prompt
    assert all(
        f"Supporting passage number {index}" in generator.user_prompt
        for index in range(2, 6)
    )


def test_pipeline_retries_once_after_validation_failure():
    """Retry with clearer instructions when a citation ID is invalid."""

    class CorrectingGenerator:
        """Return an invalid source ID once, then return a valid plan."""

        calls = 0
        prompts = []

        def generate(self, system_prompt, user_prompt, response_schema=None):
            """Count attempts and retain the retry instructions.

            Args:
                system_prompt: System instructions accepted by the generator
                    protocol.
                user_prompt: Prompt captured to verify clear-English retry
                    instructions.
                response_schema: Structured-output schema accepted by the
                    protocol.
            """
            self.calls += 1
            self.prompts.append(user_prompt)
            if self.calls == 1:
                plan = valid_plan()
                plan.weeks[0].workouts[0].exercises[0].source_ids = ["unknown"]
                return plan
            return valid_plan()

    generator = CorrectingGenerator()
    result = PlanPipeline(FakeRetriever(), generator).create_plan(request())
    assert result.status == "completed"
    assert generator.calls == 2
    assert "Keep the plan in clear English" in generator.prompts[1]


def test_pipeline_can_disable_retries_for_bounded_smoke_runs():
    """Make one model call and return its validation failure without retrying."""

    class InvalidGenerator:
        calls = 0

        def generate(self, system_prompt, user_prompt, response_schema=None):
            self.calls += 1
            plan = valid_plan()
            plan.weeks[0].workouts[0].exercises[0].source_ids = ["unknown"]
            return plan

    generator = InvalidGenerator()
    result = PlanPipeline(
        FakeRetriever(), generator, max_generation_attempts=1
    ).create_plan(request())

    assert result.status == "generation_error"
    assert generator.calls == 1
    assert "1 attempt(s)" in result.message
    assert result._evaluation_diagnostics["error_stage"] == "validation"
    assert result._evaluation_diagnostics["last_error"]
    assert result._evaluation_diagnostics["candidate_plan"]["title"]
    assert "candidate_plan" not in result.model_dump()


def test_ollama_generator_retains_raw_response_when_plan_parsing_fails(monkeypatch):
    """Keep the exact provider response available to evaluation diagnostics."""
    from src.llm.ollama import OllamaGenerator

    raw_output = '{"weeks":[]}'
    generator = OllamaGenerator("test-model")
    monkeypatch.setattr(generator, "_generate_content", lambda *_args: (raw_output, 1))

    with pytest.raises(RuntimeError, match="valid plan"):
        generator.generate("system", "user")

    assert generator._last_raw_output == raw_output


def test_prompt_and_structured_schema_require_exact_requested_week_count():
    """Constrain both generated instructions and schema to requested plan length."""
    training_request = request().model_copy(update={"weeks": 4})
    prompt = make_user_prompt(training_request, "")
    schema = plan_response_schema(training_request)

    assert "exactly 4 weeks" in prompt
    assert "write the plan in clear English" in prompt
    assert "Exact quotations and semantic verification are not required" in prompt
    assert '"minItems": 4' in prompt
    assert schema["properties"]["weeks"]["minItems"] == 4
    assert schema["properties"]["weeks"]["maxItems"] == 4
    assert schema["$defs"]["Week"]["properties"]["workouts"]["minItems"] == 1
    exercise_schema = schema["$defs"]["Exercise"]
    assert "estimated_minutes" in exercise_schema["required"]
    assert exercise_schema["properties"]["estimated_minutes"]["minimum"] == 1
    assert "rough estimates, not exact timings" in prompt


def test_prompt_includes_previous_plan_and_explicit_refinement_instructions():
    """Make prior-plan context available without letting it override this request."""
    training_request = request().model_copy(
        update={"refinement_instructions": "Avoid crunches and add more variety."}
    )
    previous_plan = {
        "request": {"focus": "Build abdominal strength"},
        "plan": {
            "title": "Starter core plan",
            "summary": "A short introductory plan.",
            "weeks": [
                {
                    "objective": "Learn core control.",
                    "workouts": [
                        {
                            "day": "Monday",
                            "focus": "Core stability",
                            "exercises": [
                                {
                                    "name": "Crunch",
                                    "sets": 3,
                                    "repetitions": "8-12",
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        "sources": [{"evidence_text": "This source should not enter the prompt."}],
    }

    prompt = make_user_prompt(training_request, "", previous_plan=previous_plan)

    assert "Avoid crunches and add more variety." in prompt
    assert "Starter core plan" in prompt
    assert "Crunch 3 8-12" in prompt
    assert "current request and refinement instructions take priority" in prompt
    assert "This source should not enter the prompt." not in prompt


def test_previous_plan_summary_stays_within_character_budget():
    """Bound historical prompt context even when a saved plan is unusually large."""
    previous_plan = {
        "request": {"focus": "A" * 10_000},
        "plan": {
            "title": "B" * 10_000,
            "summary": "C" * 10_000,
            "weeks": [
                {
                    "objective": "D" * 10_000,
                    "workouts": [
                        {
                            "day": "Monday",
                            "focus": "E" * 10_000,
                            "exercises": [
                                {
                                    "name": f"Exercise {index} " + "F" * 100,
                                    "sets": 3,
                                    "repetitions": "8",
                                }
                                for index in range(30)
                            ],
                        }
                    ],
                }
            ],
        },
    }

    summary = compact_previous_plan(previous_plan)

    assert len(summary) <= PREVIOUS_PLAN_CONTEXT_MAX_CHARS
    assert "Earlier plan details omitted" in summary


def test_system_prompt_selects_rag_or_no_rag_evidence_policy(tmp_path, monkeypatch):
    """Render one explicit evidence policy without leaving template markers.

    Args:
        tmp_path: Provide an isolated prompt template for marker validation.
        monkeypatch: Point the prompt loader at the temporary template.
    """
    from src.rag import prompt

    prompt_file = tmp_path / "system_prompt.txt"
    prompt_file.write_text(
        "Shared safety rules.\n{{EVIDENCE_POLICY}}\nReturn JSON.",
        encoding="utf-8",
    )
    monkeypatch.setattr(prompt, "PROMPT_FILE", prompt_file)

    rag_prompt = prompt.load_system_prompt()
    no_rag_prompt = prompt.load_system_prompt(require_evidence=False)

    assert "{{EVIDENCE_POLICY}}" not in rag_prompt
    assert "Exact quotations and semantic verification are not required" in rag_prompt
    assert "Include at least one supplied source ID" in rag_prompt
    assert "{{EVIDENCE_POLICY}}" not in no_rag_prompt
    assert "Use general fitness knowledge" in no_rag_prompt
    assert "Cite supplied source IDs" not in no_rag_prompt


def test_system_prompt_rejects_missing_or_duplicate_evidence_policy_marker(
    tmp_path, monkeypatch
):
    """Fail visibly rather than loading a prompt with ambiguous policy wiring.

    Args:
        tmp_path: Hold each malformed prompt-template variant.
        monkeypatch: Point the loader at the temporary prompt file.
    """
    from src.rag import prompt

    prompt_file = tmp_path / "system_prompt.txt"
    monkeypatch.setattr(prompt, "PROMPT_FILE", prompt_file)

    for template in ("No policy marker.", "{{EVIDENCE_POLICY}} {{EVIDENCE_POLICY}}"):
        prompt_file.write_text(template, encoding="utf-8")
        with pytest.raises(ValueError, match="exactly one"):
            prompt.load_system_prompt()


def test_pipeline_returns_a_clear_status_when_retrieval_fails():
    """Convert retriever exceptions into a user-facing retrieval error result."""

    class BrokenRetriever:
        """Simulate an unavailable retrieval index."""

        def search(self, query, top_k):
            """Raise an offline-index error when retrieval is attempted.

            Args:
                query: Generated search query, unused because the index is
                    unavailable.
                top_k: Requested depth, unused for the same reason.
            """
            raise ConnectionError("index is offline")

    result = PlanPipeline(BrokenRetriever(), FakeGenerator()).create_plan(request())

    assert result.status == "retrieval_error"
    assert "knowledge base" in result.message


def test_bm25_retriever_finds_matching_chunk():
    """Return a matching BM25 passage with its source ID and rank."""
    results = BM25Retriever([chunk()]).search("abdominal repetitions", top_k=1)

    assert results[0].passage_ids == ["video-1_p001"]
    assert results[0].rank == 1


def test_response_schema_only_allows_retrieved_citations():
    """Stop the model from omitting citations or inventing source IDs."""
    schema = plan_response_schema(request(), ["source-b", "source-a", "source-b"])
    citations = schema["$defs"]["Exercise"]["properties"]["source_ids"]

    assert citations["items"]["enum"] == ["source-a", "source-b"]
    assert citations["minItems"] == 1
    assert "source_ids" in schema["$defs"]["Exercise"]["required"]
    quote_source = schema["$defs"]["EvidenceQuote"]["properties"]["source_id"]
    assert quote_source["enum"] == ["source-a", "source-b"]


def test_response_schema_without_sources_keeps_citations_optional():
    """Leave callers that supply no evidence, such as no-RAG runs, unchanged."""
    schema = plan_response_schema(request())
    citations = schema["$defs"]["Exercise"]["properties"]["source_ids"]

    assert "enum" not in citations["items"]
    assert "minItems" not in citations
