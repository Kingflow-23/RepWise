"""Generation failures: one status per cause, retries only where they help.

No test contacts Ollama: provider errors are simulated with the same exception
types the ``ollama`` client raises.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from ollama import ResponseError

import src.api.main as api_module
from config.config import AppConfig
from src.agents.health import FixedHealthChecker
from src.api.main import create_app
from src.db.repository import StudentRepository
from src.generation.pipeline import PlanPipeline
from src.llm.errors import (
    GenerationTimeout,
    InvalidModelOutput,
    ModelNotFound,
    ProviderUnavailable,
    classify_provider_error,
)
from src.llm.ollama import OllamaGenerator
from src.retrieval.query_expansion import QueryAnalysis
from tests.generation.test_pipeline import FakeRetriever, request, valid_plan

MODEL = "qwen3.5:9b"


def classify(error):
    return classify_provider_error(error, model=MODEL, timeout_seconds=300)


# --- Classification of what the Ollama client raises -------------------------


def test_connection_refused_is_provider_unavailable():
    failure = classify(ConnectionError("Failed to connect to Ollama."))

    assert isinstance(failure, ProviderUnavailable)
    assert failure.status == "provider_unavailable"
    assert failure.retryable is False


def test_missing_model_is_model_not_found_and_names_the_model():
    failure = classify(ResponseError(f'{{"error":"model \\"{MODEL}\\" not found"}}', 404))

    assert isinstance(failure, ModelNotFound)
    assert failure.retryable is False
    assert f"ollama pull {MODEL}" in str(failure)


def test_other_server_error_is_provider_unavailable():
    failure = classify(ResponseError("out of memory", 500))

    assert isinstance(failure, ProviderUnavailable)
    assert "HTTP 500" in str(failure)


@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout("timed out"), TimeoutError("timed out")]
)
def test_timeouts_are_generation_timeout(error):
    failure = classify(error)

    assert isinstance(failure, GenerationTimeout)
    assert failure.status == "generation_timeout"
    assert failure.retryable is False
    assert "300 s" in str(failure)


def test_unreadable_answer_is_retryable_and_keeps_the_detail_private():
    failure = classify(ValueError("weeks: field required at C:\\secret\\path"))

    assert isinstance(failure, InvalidModelOutput)
    assert failure.status == "generation_error"
    assert failure.retryable is True
    assert "secret" not in str(failure)
    assert "field required" in failure.detail


def test_ollama_generator_raises_the_classified_failure(monkeypatch):
    generator = OllamaGenerator(MODEL)

    def missing_model(*_args):
        raise ResponseError("model not found", 404)

    monkeypatch.setattr(generator, "_generate_content", missing_model)

    with pytest.raises(ModelNotFound):
        generator.generate("system", "user")


# --- The pipeline: stop at once, or retry --------------------------------------


class RaisingGenerator:
    """Raise the same failure on every call, counting the calls."""

    def __init__(self, failure):
        self.failure = failure
        self.calls = 0

    def generate(self, system_prompt, user_prompt, response_schema=None):
        self.calls += 1
        raise self.failure


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (ProviderUnavailable("Ollama is not reachable."), "provider_unavailable"),
        (ModelNotFound(f"The configured model {MODEL} is not installed."), "model_not_found"),
        (GenerationTimeout("The model did not finish within 300 s."), "generation_timeout"),
    ],
)
def test_service_failures_stop_without_retry(failure, status):
    generator = RaisingGenerator(failure)

    result = PlanPipeline(FakeRetriever(), generator).create_plan(request())

    assert result.status == status
    assert generator.calls == 1
    assert result.message == str(failure)
    assert result.plan is None


def test_unreadable_answers_are_retried_then_reported():
    generator = RaisingGenerator(
        InvalidModelOutput("The model did not return a readable plan.", detail="bad json")
    )

    result = PlanPipeline(FakeRetriever(), generator).create_plan(request())

    assert result.status == "generation_error"
    assert generator.calls == 3
    assert "3 attempt(s)" in result.message
    assert "bad json" not in result.message


def test_retry_prompt_uses_the_private_detail():
    class FlakyGenerator:
        def __init__(self):
            self.prompts = []

        def generate(self, system_prompt, user_prompt, response_schema=None):
            self.prompts.append(user_prompt)
            if len(self.prompts) == 1:
                raise InvalidModelOutput("unreadable", detail="weeks: field required")
            return valid_plan()

    generator = FlakyGenerator()

    result = PlanPipeline(FakeRetriever(), generator).create_plan(request())

    assert result.status == "completed"
    assert "weeks: field required" in generator.prompts[1]


def test_a_plan_breaking_the_rules_is_plan_not_compliant():
    class LongWorkoutGenerator:
        calls = 0

        def generate(self, system_prompt, user_prompt, response_schema=None):
            self.calls += 1
            plan = valid_plan()
            plan.weeks[0].workouts[0].estimated_minutes = 999
            return plan

    generator = LongWorkoutGenerator()

    result = PlanPipeline(FakeRetriever(), generator).create_plan(request())

    assert result.status == "plan_not_compliant"
    assert generator.calls == 3
    assert "longer than the requested session duration" in result.message


def test_retrieval_errors_do_not_leak_internal_details():
    class BrokenRetriever:
        def search(self, query, top_k):
            raise FileNotFoundError("C:\\Users\\someone\\data\\semantic.npz missing")

    result = PlanPipeline(BrokenRetriever(), RaisingGenerator(None)).create_plan(
        request()
    )

    assert result.status == "retrieval_error"
    assert "semantic.npz" not in result.message
    assert "/api/v1/health" in result.message


# --- The API: one HTTP code per cause -------------------------------------------


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Return a client factory for an app whose generator raises ``failure``."""

    class InScopeAnalyzer:
        def analyze(self, query):
            return QueryAnalysis("in_scope", query, "")

    monkeypatch.setattr(api_module, "OllamaQueryExpander", lambda *_a: InScopeAnalyzer())
    monkeypatch.setattr(api_module, "read_access_token", lambda token, secret: token)

    def build(generator):
        repository = StudentRepository(str(tmp_path / "failures.sqlite3"))
        repository.initialize()
        user_id = repository.create_user("Failure Test", "test-hash")
        app = create_app(
            AppConfig(
                database_path=str(tmp_path / "failures.sqlite3"),
                auth_secret="test-secret",
            ),
            repository=repository,
            retriever=FakeRetriever(),
            generator=generator,
            health_checker=FixedHealthChecker(("pain",)),
            ollama_models=lambda: [],
        )
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {user_id}"}
        profile = request().profile.model_dump(mode="json")
        assert (
            client.put(
                f"/api/v1/students/{user_id}/profile", json=profile, headers=headers
            ).status_code
            == 200
        )
        return client, user_id, headers, repository

    return build


@pytest.mark.parametrize(
    ("failure", "http_status", "status"),
    [
        (ProviderUnavailable("Ollama is not reachable."), 503, "provider_unavailable"),
        (ModelNotFound("The configured model is not installed."), 503, "model_not_found"),
        (GenerationTimeout("Too slow."), 504, "generation_timeout"),
        (InvalidModelOutput("Unreadable."), 502, "generation_error"),
    ],
)
def test_generation_failures_map_to_distinct_http_codes(
    api, failure, http_status, status
):
    client, user_id, headers, repository = api(RaisingGenerator(failure))

    response = client.post(
        f"/api/v1/students/{user_id}/plans",
        json={"focus": "Build a simple routine"},
        headers=headers,
    )

    assert response.status_code == http_status
    assert response.json()["detail"]["status"] == status
    # A failed generation never replaces or adds a saved plan.
    assert repository.list_plans(user_id) == []


def test_invalid_request_is_still_422(api):
    client, user_id, headers, _ = api(RaisingGenerator(None))

    response = client.post(
        f"/api/v1/students/{user_id}/plans",
        json={"focus": "x"},
        headers=headers,
    )

    assert response.status_code == 422
