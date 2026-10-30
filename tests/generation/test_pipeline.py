from src.generation import PlanPipeline, validate_plan
from src.rag import make_query, select_evidence
from src.retrieval import BM25Retriever
from src.schemas import PlanRequest, RetrievedChunk, TrainingPlan, UserProfile


def request() -> PlanRequest:
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


def chunk(flags: list[str] | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        text="For hypertrophy, use a range from 6 to 20 repetitions.",
        video_id="video-1",
        passage_ids=["video-1_p001"],
        title="Abdominal training",
        video_url="https://example.com",
        start_url="https://example.com?t=10",
        flags=flags or [],
    )


def valid_plan() -> TrainingPlan:
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
                                    "coaching_note": "Move slowly.",
                                    "source_ids": ["video-1_p001"],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )


def test_context_keeps_traceable_training_chunk_and_removes_sponsor():
    evidence, sources = select_evidence([chunk(["sponsor"]), chunk()])
    assert "SOURCE video-1_p001" in evidence
    assert sources[0].url.endswith("?t=10")


def test_context_skips_an_oversized_chunk_and_keeps_later_evidence():
    oversized = chunk()
    oversized.text = "x" * 8_100
    evidence, sources = select_evidence([oversized, chunk()])
    assert "SOURCE video-1_p001" in evidence
    assert len(sources) == 1


def test_query_contains_current_goal_and_profile_information():
    query = make_query(request())
    assert "Build abdominal strength" in query
    assert "beginner" in query


def test_validator_rejects_unknown_source():
    plan = valid_plan()
    plan.weeks[0].workouts[0].exercises[0].source_ids = ["unknown"]
    try:
        validate_plan(plan, request(), {"video-1_p001"})
    except ValueError as error:
        assert "was not retrieved" in str(error)
    else:
        raise AssertionError("Expected validation to fail")


def test_validator_rejects_duplicate_days_and_undeclared_named_equipment():
    plan = valid_plan()
    plan.weeks[0].workouts[0].exercises[0].name = "Barbell squat"
    plan.weeks[0].workouts[0].exercises[0].equipment = []
    try:
        validate_plan(plan, request(), {"video-1_p001"})
    except ValueError as error:
        assert "equipment" in str(error)
    else:
        raise AssertionError("Expected equipment validation to fail")

    plan = valid_plan()
    plan.weeks[0].workouts.append(
        plan.weeks[0].workouts[0].model_copy(deep=True)
    )
    training_request = request()
    training_request.profile.available_days.append("Wednesday")
    try:
        validate_plan(plan, training_request, {"video-1_p001"})
    except ValueError as error:
        assert "one workout per day" in str(error)
    else:
        raise AssertionError("Expected duplicate workout day validation to fail")


def test_validator_rejects_equipment_not_in_profile():
    plan = valid_plan()
    plan.weeks[0].workouts[0].exercises[0].equipment = ["barbell"]
    try:
        validate_plan(plan, request(), {"video-1_p001"})
    except ValueError as error:
        assert "equipment" in str(error)
    else:
        raise AssertionError("Expected equipment validation to fail")


class FakeRetriever:
    def search(self, query, top_k):
        return [chunk()]


class FakeGenerator:
    def generate(self, system_prompt, user_prompt):
        return valid_plan()


def test_pipeline_returns_valid_plan_with_mocked_dependencies():
    result = PlanPipeline(FakeRetriever(), FakeGenerator()).create_plan(request())
    assert result.status == "completed"
    assert result.plan is not None


def test_pipeline_retries_once_after_validation_failure():
    class CorrectingGenerator:
        calls = 0

        def generate(self, system_prompt, user_prompt):
            self.calls += 1
            if self.calls == 1:
                plan = valid_plan()
                plan.weeks[0].workouts[0].exercises[0].source_ids = ["not-supplied"]
                return plan
            return valid_plan()

    generator = CorrectingGenerator()
    result = PlanPipeline(FakeRetriever(), generator).create_plan(request())
    assert result.status == "completed"
    assert generator.calls == 2


def test_pipeline_returns_a_clear_status_when_retrieval_fails():
    class BrokenRetriever:
        def search(self, query, top_k):
            raise ConnectionError("index is offline")

    result = PlanPipeline(BrokenRetriever(), FakeGenerator()).create_plan(request())

    assert result.status == "retrieval_error"
    assert "knowledge base" in result.message


def test_bm25_retriever_finds_matching_chunk():
    results = BM25Retriever([chunk()]).search("abdominal repetitions", top_k=1)

    assert results[0].passage_ids == ["video-1_p001"]
    assert results[0].rank == 1
