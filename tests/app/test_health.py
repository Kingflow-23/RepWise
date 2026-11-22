"""``GET /api/v1/health``: configuration versus real availability.

Every test uses a temporary database and corpus. Ollama is replaced by a fake
model listing, except in the bounded-listing tests, which use a local stub
server or a closed port. No test loads an embedding model or runs a generation.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.main as api_module
from config.config import AppConfig
from src.agents.health import FixedHealthChecker
from src.api.diagnostics import (
    OllamaUnavailable,
    list_ollama_models,
    retrieval_setup_error,
)
from src.api.main import create_app
from src.retrieval.base import MissingRetriever
from src.retrieval.query_expansion import QueryAnalysis, QueryExpansionRetriever
from src.retrieval.semantic import corpus_fingerprint, load_chunks

GENERATION_MODEL = "qwen3.5:9b"
SMALL_MODEL = "qwen3.5:4b"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class InScopeAnalyzer:
    """Scope analyzer that never calls Ollama."""

    def analyze(self, query):
        return QueryAnalysis("in_scope", query, "")


class WorkingRetriever:
    """A retriever that initialized correctly."""

    def search(self, query, top_k):
        return []


class NoGenerationAllowed:
    """Fails the test if a health check ever starts a generation."""

    def generate(self, *_args, **_kwargs):
        raise AssertionError("/health must not run a generation")


@pytest.fixture(autouse=True)
def offline_ollama_services(monkeypatch):
    """Replace the Ollama-backed scope analyzer used at app creation."""
    monkeypatch.setattr(
        api_module, "OllamaQueryExpander", lambda *_args: InScopeAnalyzer()
    )


def make_config(tmp_path, **overrides) -> AppConfig:
    """Return settings isolated in ``tmp_path`` (database and logs)."""
    values = {
        "database_path": str(tmp_path / "health.sqlite3"),
        "auth_secret": "test-secret",
        "ollama_model": GENERATION_MODEL,
        "agent_model": GENERATION_MODEL,
        "query_expansion_model": SMALL_MODEL,
        "health_check_model": SMALL_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }
    values.update(overrides)
    return AppConfig(**values)


def get_health(config, *, retriever=None, installed=None, listing=None) -> dict:
    """Create the app and return the JSON of ``GET /api/v1/health``."""
    if listing is None:
        models = list(
            installed if installed is not None else (GENERATION_MODEL, SMALL_MODEL)
        )

        def listing():
            return models

    app = create_app(
        config,
        retriever=retriever,
        generator=NoGenerationAllowed(),
        health_checker=FixedHealthChecker(("pain",)),
        ollama_models=listing,
    )
    response = TestClient(app).get("/api/v1/health")
    assert response.status_code == 200
    return response.json()


def write_corpus(tmp_path):
    """Write a two-chunk corpus and return the chunk file path."""
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    records = [
        {
            "text": f"Training passage {index}.",
            "video_id": f"video-{index}",
            "chunk_id": f"video-{index}_chunk",
            "passage_ids": [f"video-{index}_p001"],
            "title": "Training basics",
            "video_url": "https://example.com",
        }
        for index in (1, 2)
    ]
    chunk_file = chunk_dir / "corpus.jsonl"
    chunk_file.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    return chunk_file


def write_index(path, chunk_file, *, model):
    """Write a small, readable index built for ``model``."""
    chunks = load_chunks(chunk_file.parent)
    np.savez(
        path,
        vectors=np.ones((len(chunks), 4), dtype=np.float32),
        chunk_ids=np.array([chunk.chunk_id for chunk in chunks]),
        model=np.array(model),
        fingerprint=np.array(corpus_fingerprint(chunks)),
    )


def test_everything_available_reports_ok(tmp_path):
    health = get_health(make_config(tmp_path), retriever=WorkingRetriever())

    assert health["status"] == "ok"
    assert health["retrieval"] == "ready"
    assert health["retrieval_error"] is None
    assert health["ollama"] == {"status": "reachable", "error": None}
    assert all(model["installed"] is True for model in health["models"].values())
    assert health["problems"] == []


def test_configuration_is_reported_apart_from_availability(tmp_path):
    health = get_health(
        make_config(tmp_path, retrieval_strategy="bm25"), retriever=WorkingRetriever()
    )

    assert health["configuration"] == {
        "retrieval_strategy": "bm25",
        "embedding_model": EMBEDDING_MODEL,
        "models": {
            "generation": GENERATION_MODEL,
            "agents": GENERATION_MODEL,
            "query_analysis": SMALL_MODEL,
            "health_check": SMALL_MODEL,
        },
    }
    # Fields kept for the existing clients.
    assert health["retriever"] == "bm25"
    assert health["model"] == GENERATION_MODEL


def test_missing_retriever_is_detected_through_the_query_expansion_wrapper(tmp_path):
    health = get_health(
        make_config(tmp_path), retriever=MissingRetriever("Corpus not prepared.")
    )

    assert health["status"] == "degraded"
    assert health["retrieval"] == "not connected"
    assert health["retrieval_error"] == "Corpus not prepared."
    assert any("Retrieval is not available" in p for p in health["problems"])


def test_retrieval_setup_error_looks_through_wrappers():
    missing = MissingRetriever("No index.")
    wrapped = QueryExpansionRetriever(missing, InScopeAnalyzer())

    assert retrieval_setup_error(wrapped) == "No index."
    assert (
        retrieval_setup_error(
            QueryExpansionRetriever(WorkingRetriever(), InScopeAnalyzer())
        )
        is None
    )


def test_missing_index_marks_retrieval_not_connected(tmp_path):
    chunk_file = write_corpus(tmp_path)
    config = make_config(
        tmp_path,
        chunk_file=str(chunk_file),
        retrieval_strategy="hybrid",
        embedding_index=str(tmp_path / "absent" / "semantic.npz"),
    )

    health = get_health(config)

    assert health["status"] == "degraded"
    assert health["retrieval"] == "not connected"
    assert "Embedding index not found" in health["retrieval_error"]
    # The configured strategy is reported, not silently replaced.
    assert health["retriever"] == "hybrid"
    assert health["configuration"]["retrieval_strategy"] == "hybrid"


def test_incompatible_index_marks_retrieval_not_connected(tmp_path):
    chunk_file = write_corpus(tmp_path)
    index_path = tmp_path / "semantic.npz"
    write_index(index_path, chunk_file, model="another/embedding-model")
    config = make_config(
        tmp_path,
        chunk_file=str(chunk_file),
        retrieval_strategy="semantic",
        embedding_index=str(index_path),
    )

    health = get_health(config)

    assert health["status"] == "degraded"
    assert health["retrieval"] == "not connected"
    assert "another/embedding-model" in health["retrieval_error"]
    assert EMBEDDING_MODEL in health["retrieval_error"]


def test_unreachable_ollama_is_reported_without_guessing_models(tmp_path):
    def listing():
        raise OllamaUnavailable("Ollama refused or dropped the connection.")

    health = get_health(
        make_config(tmp_path), retriever=WorkingRetriever(), listing=listing
    )

    assert health["status"] == "degraded"
    assert health["retrieval"] == "ready"
    assert health["ollama"] == {
        "status": "unreachable",
        "error": "Ollama refused or dropped the connection.",
    }
    # Unknown, not "missing": Ollama could not be asked.
    assert all(model["installed"] is None for model in health["models"].values())
    assert any("Ollama is not reachable" in p for p in health["problems"])


def test_missing_model_is_reported_and_not_replaced(tmp_path):
    health = get_health(
        make_config(tmp_path), retriever=WorkingRetriever(), installed=[SMALL_MODEL]
    )

    assert health["status"] == "degraded"
    assert health["ollama"]["status"] == "reachable"
    assert health["models"]["generation"] == {
        "name": GENERATION_MODEL,
        "installed": False,
    }
    assert health["models"]["agents"] == {"name": GENERATION_MODEL, "installed": False}
    assert health["models"]["query_analysis"]["installed"] is True
    # One problem for the model, naming both roles that use it.
    missing = [p for p in health["problems"] if GENERATION_MODEL in p]
    assert len(missing) == 1
    assert "generation, agents" in missing[0]
    # The configuration still names the requested model.
    assert health["model"] == GENERATION_MODEL


def test_untagged_model_name_matches_latest(tmp_path):
    health = get_health(
        make_config(
            tmp_path,
            ollama_model="llama3",
            agent_model="llama3",
            query_expansion_model="llama3",
            health_check_model="llama3",
        ),
        retriever=WorkingRetriever(),
        installed=["llama3:latest"],
    )

    assert health["status"] == "ok"


class _TagsHandler(BaseHTTPRequestHandler):
    """Minimal stand-in for Ollama's ``GET /api/tags``."""

    body = b""
    delay = 0.0
    paths: list[str] = []

    def do_GET(self):  # noqa: N802 - http.server naming
        type(self).paths.append(self.path)
        time.sleep(type(self).delay)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def tags_server():
    """Serve a fake Ollama model list on a free local port."""
    _TagsHandler.paths = []
    _TagsHandler.delay = 0.0
    _TagsHandler.body = json.dumps(
        {"models": [{"name": SMALL_MODEL, "model": SMALL_MODEL}]}
    ).encode()
    server = HTTPServer(("127.0.0.1", 0), _TagsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield _TagsHandler, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_listing_reads_installed_models_only(tags_server):
    handler, url = tags_server

    assert list_ollama_models(url) == [SMALL_MODEL]
    # Only the listing endpoint: no model is loaded and nothing is generated.
    assert handler.paths == ["/api/tags"]


def test_listing_is_bounded_in_time(tags_server):
    handler, url = tags_server
    handler.delay = 2.0

    started = time.perf_counter()
    with pytest.raises(OllamaUnavailable, match="did not answer within 0.3 s"):
        list_ollama_models(url, timeout_seconds=0.3)
    assert time.perf_counter() - started < 1.5


def test_listing_rejects_an_unreadable_answer(tags_server):
    handler, url = tags_server
    handler.body = b"not json"

    with pytest.raises(OllamaUnavailable, match="unreadable"):
        list_ollama_models(url)


def test_listing_fails_fast_when_nothing_listens():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # The port is free again: nothing listens there.

    with pytest.raises(OllamaUnavailable):
        list_ollama_models(f"http://127.0.0.1:{port}", timeout_seconds=1.0)
