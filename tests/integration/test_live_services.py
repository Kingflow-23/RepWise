"""Integration tests against the real local services. Opt-in, and slow.

They need Ollama with the configured models, the prepared corpus
(``REPWISE_CHUNK_FILE``) and internet access to Photon. Run them with:

    $env:REPWISE_LIVE_TESTS = "1"; python -m pytest tests/integration -q

Without ``REPWISE_LIVE_TESTS=1`` they are skipped, so the normal suite stays
deterministic and offline.
"""

import os

import pytest

from config.config import AppConfig
from scripts.demo_context import demo_context
from src.agents.health import OllamaHealthChecker
from src.agents.orchestrator import build_chat_model, build_orchestrator
from src.tools import places

pytestmark = pytest.mark.skipif(
    os.getenv("REPWISE_LIVE_TESTS") != "1",
    reason="set REPWISE_LIVE_TESTS=1 to run live tests",
)


@pytest.fixture(scope="module")
def config():
    return AppConfig.from_environment()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("J'ai mal au genou depuis la dernière séance", True),
        ("My knee clicked and swelled after squats", True),
        ("j'ai une gêne à l'épaule", True),
        ("The last set was difficult", False),
        ("Too easy, I could do more reps", False),
    ],
)
def test_the_real_health_check(config, text, expected):
    checker = OllamaHealthChecker(config.health_check_model, config.ollama_url)

    assert checker.check(text).health_concern is expected


def test_the_real_map_service_finds_gyms(config, tmp_path):
    context, _ = demo_context(config, tmp_path / "live.sqlite3")

    result = places.search_nearby_places(context, "gym")

    assert result.ok and result.places, result
    assert result.places == sorted(result.places, key=lambda place: place.distance_km)


def test_a_knowledge_question_end_to_end(config, tmp_path):
    context, _ = demo_context(config, tmp_path / "live.sqlite3")
    orchestrator = build_orchestrator(context, build_chat_model(config))

    response = orchestrator.ask(
        "According to your documents, how long should I rest between sets for hypertrophy?",
        "live",
    )

    assert "knowledge" in [step.agent for step in response.steps]
    search = response.reports[0].results.get("search_documents", {})
    assert search.get("ok"), search
