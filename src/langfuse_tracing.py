"""Privacy-conscious Langfuse observations for agent execution."""

from __future__ import annotations

import os
from contextlib import contextmanager
from time import perf_counter
from types import TracebackType
from typing import Any, Iterator


@contextmanager
def _safe_observation(client: Any, **options: Any) -> Iterator[Any]:
    """Keep exception text out of telemetry while preserving application errors."""
    started = perf_counter()
    caught: Exception | None = None
    traceback: TracebackType | None = None
    with client.start_as_current_observation(**options) as observation:
        try:
            yield observation
        except Exception as error:
            caught = error
            traceback = error.__traceback__
            observation.update(
                level="ERROR",
                metadata={
                    "error_type": type(error).__name__,
                    "duration_seconds": round(perf_counter() - started, 3),
                },
            )
    if caught is not None:
        raise caught.with_traceback(traceback)


def _client() -> Any | None:
    """Return a configured Langfuse client, or disable tracing when unset."""
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    if not secret_key and not public_key:
        return None
    if not secret_key or not public_key:
        raise RuntimeError(
            "Set both LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY to enable tracing."
        )
    from langfuse import get_client

    return get_client()


@contextmanager
def request_observation(model: str) -> Iterator[Any | None]:
    """Group one agent request without exporting its input or output."""
    client = _client()
    if client is None:
        yield None
        return
    with _safe_observation(
        client,
        as_type="span",
        name="repwise.agent.request",
    ) as observation:
        observation.update(metadata={"model": model})
        yield observation


@contextmanager
def generation_observation(name: str, model: str) -> Iterator[Any | None]:
    """Trace one model call without recording prompts or generated text."""
    client = _client()
    if client is None:
        yield None
        return
    with _safe_observation(
        client, as_type="generation", name=name, model=model
    ) as observation:
        yield observation


@contextmanager
def tool_observation(name: str) -> Iterator[Any | None]:
    """Trace one tool execution without recording arguments or results."""
    client = _client()
    if client is None:
        yield None
        return
    with _safe_observation(client, as_type="tool", name=name) as observation:
        yield observation


@contextmanager
def tool_request_observation(name: str) -> Iterator[Any | None]:
    """Trace a model-requested tool call, including calls held for approval."""
    client = _client()
    if client is None:
        yield None
        return
    with _safe_observation(
        client, as_type="span", name=f"agent request: {name}"
    ) as observation:
        yield observation


@contextmanager
def phase_observation(name: str) -> Iterator[Any | None]:
    """Trace one graph phase without capturing its state."""
    client = _client()
    if client is None:
        yield None
        return
    with _safe_observation(
        client, as_type="span", name=f"agent phase: {name}"
    ) as observation:
        yield observation


def finish_generation(
    observation: Any | None,
    started: float,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    error: Exception | None = None,
) -> None:
    """Attach safe usage, timing, and error metadata to a generation."""
    if observation is None:
        return
    usage_details = {}
    if input_tokens is not None:
        usage_details["input"] = input_tokens
    if output_tokens is not None:
        usage_details["output"] = output_tokens
    metadata = {"duration_seconds": round(perf_counter() - started, 3)}
    if input_tokens is not None:
        metadata["input_tokens"] = input_tokens
    if output_tokens is not None:
        metadata["output_tokens"] = output_tokens
    if error is not None:
        metadata["error_type"] = type(error).__name__
    update = {"metadata": metadata}
    if usage_details:
        update["usage_details"] = usage_details
    observation.update(**update)


def flush_traces() -> None:
    """Flush buffered traces for short-lived command-line runs."""
    client = _client()
    if client is not None:
        client.flush()
