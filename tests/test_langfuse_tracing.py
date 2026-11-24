"""Verify optional tracing setup and privacy-safe observation payloads."""

import sys
from contextlib import contextmanager
from time import perf_counter
from types import ModuleType

import pytest

from src.agents.metrics import (
    Recorder,
    collect_request_metrics,
    record_external_call,
)
from src.langfuse_tracing import (
    finish_generation,
    generation_observation,
    request_observation,
)


def test_tracing_is_disabled_without_project_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    with request_observation("local-model") as observation:
        assert observation is None


def test_partial_project_keys_fail_explicitly(monkeypatch):
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    with pytest.raises(RuntimeError, match="both LANGFUSE_SECRET_KEY"):
        with request_observation("local-model"):
            pass


def test_nested_provider_calls_are_included_in_request_metrics():
    recorder = Recorder("local-model")
    started = perf_counter()

    with collect_request_metrics(recorder):
        record_external_call("plan generation", started, 20, 7)

    assert len(recorder.model_calls) == 1
    assert recorder.model_calls[0].purpose == "plan generation"
    assert recorder.model_calls[0].input_tokens == 20
    assert recorder.model_calls[0].output_tokens == 7


def test_observations_only_export_safe_metadata(monkeypatch):
    observations = []

    class FakeObservation:
        def update(self, **values):
            self.updates.append(values)

        def __init__(self):
            self.updates = []
            observations.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeClient:
        @contextmanager
        def start_as_current_observation(self, **values):
            starts.append(values)
            with FakeObservation() as observation:
                yield observation

        def flush(self):
            return None

    starts = []
    fake_module = ModuleType("langfuse")
    fake_module.get_client = lambda: FakeClient()
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public")

    with request_observation("local-model"):
        with generation_observation("planning", "local-model") as generation:
            finish_generation(generation, 1.0, 12, 4)
    with pytest.raises(ValueError, match="private prompt"):
        with generation_observation("failed planning", "local-model"):
            raise ValueError("private prompt")

    assert starts == [
        {"as_type": "span", "name": "repwise.agent.request"},
        {"as_type": "generation", "name": "planning", "model": "local-model"},
        {
            "as_type": "generation",
            "name": "failed planning",
            "model": "local-model",
        },
    ]
    assert all("input" not in values and "output" not in values for values in starts)
    assert observations[1].updates[0]["usage_details"] == {"input": 12, "output": 4}
    assert all(
        "input" not in update and "output" not in update
        for observation in observations
        for update in observation.updates
    )
    assert observations[2].updates[0]["metadata"]["error_type"] == "ValueError"
    assert "private prompt" not in repr(observations[2].updates)
