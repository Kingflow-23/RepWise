"""Measurements of one agent request, for debugging and evaluation.

Every model call records how long it took and how many tokens it used (Ollama
reports ``prompt_eval_count`` and ``eval_count``; LangChain exposes them as
``usage_metadata``). Every tool call records its latency. The orchestrator
adds the time of each phase and totals, and returns everything in
``AgentResponse.metrics`` so the Streamlit app, the run transcripts and future
evaluations read the same numbers.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from pydantic import Field

from src.schemas.models import Model


class ModelCall(Model):
    """One call to the language model.

    Attributes:
        purpose: Why it was called, such as ``plan`` or ``schedule: think``.
        seconds: Wall-clock duration.
        input_tokens: Prompt tokens, when the model reports them.
        output_tokens: Generated tokens, when the model reports them.
    """

    purpose: str
    seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class Phase(Model):
    """One step of the orchestrator graph and how long it took."""

    name: str
    seconds: float


class RequestMetrics(Model):
    """Everything measured during one request.

    Attributes:
        model: The agent model used.
        total_seconds: Wall-clock time of the whole request.
        phases: Time spent in each orchestrator step, in order.
        model_calls: Every model call, orchestrator and specialists included.
        tool_calls: Number of tool calls.
        tool_seconds: Total time spent inside tools.
        input_tokens: Sum of prompt tokens reported by the model.
        output_tokens: Sum of generated tokens reported by the model.
    """

    model: str
    total_seconds: float = 0.0
    phases: list[Phase] = Field(default_factory=list)
    model_calls: list[ModelCall] = Field(default_factory=list)
    tool_calls: int = 0
    tool_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


def token_usage(message: Any) -> tuple[int | None, int | None]:
    """Read the token counts a model reply carries, if any.

    Args:
        message: A LangChain ``AIMessage`` (or anything else).

    Returns:
        ``(input_tokens, output_tokens)``; ``None`` for a count the model did
        not report.
    """
    usage = getattr(message, "usage_metadata", None) or {}
    if usage:
        return usage.get("input_tokens"), usage.get("output_tokens")
    metadata = getattr(message, "response_metadata", None) or {}
    return metadata.get("prompt_eval_count"), metadata.get("eval_count")


def record_call(purpose: str, started: float, message: Any) -> ModelCall:
    """Build the record of a finished model call.

    Args:
        purpose: Why the model was called.
        started: ``perf_counter()`` value taken just before the call.
        message: The raw reply (used for token counts).

    Returns:
        The model-call record.
    """
    input_tokens, output_tokens = token_usage(message)
    return ModelCall(
        purpose=purpose,
        seconds=round(perf_counter() - started, 3),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class Recorder:
    """Collects the orchestrator's phases and model calls during one request.

    The orchestrator resets it at the start of each request, its graph nodes
    add to it, and :meth:`build` turns it into :class:`RequestMetrics`.
    """

    def __init__(self) -> None:
        """Start empty."""
        self.phases: list[Phase] = []
        self.model_calls: list[ModelCall] = []

    def reset(self) -> None:
        """Forget the previous request's measurements."""
        self.phases, self.model_calls = [], []

    def timed(self, name: Any, node: Any) -> Any:
        """Wrap a graph node so its duration is recorded as a phase.

        Args:
            name: Phase name, or a function ``name(state) -> str``.
            node: The node function ``node(state) -> update``.

        Returns:
            A node function with the same behaviour.
        """

        def run(state: Any) -> Any:
            """Run the node and record how long it took."""
            label = name(state) if callable(name) else name
            started = perf_counter()
            try:
                return node(state)
            finally:
                self.phases.append(
                    Phase(name=label, seconds=round(perf_counter() - started, 3))
                )

        return run

    def structured(self, runnable: Any, messages: list, purpose: str) -> Any:
        """Call a structured-output model and record its time and tokens.

        Args:
            runnable: ``llm.with_structured_output(schema, include_raw=True)``.
            messages: The prompt messages.
            purpose: What the call is for, such as ``plan``.

        Returns:
            The parsed model.

        Raises:
            ValueError: If the model's answer could not be parsed.
        """
        started = perf_counter()
        result = runnable.invoke(messages)
        raw = result.get("raw") if isinstance(result, dict) else None
        self.model_calls.append(record_call(purpose, started, raw))
        if not isinstance(result, dict):
            return result
        if result.get("parsed") is None:
            raise ValueError(
                f"Unusable structured answer: {result.get('parsing_error')}"
            )
        return result["parsed"]

    def build(self, model: str, total_seconds: float, reports: list) -> RequestMetrics:
        """Combine the orchestrator's and the specialists' measurements.

        Args:
            model: Agent model name.
            total_seconds: Wall-clock time of the request.
            reports: The specialists' reports, with their own model calls
                and tool timings.

        Returns:
            The request's metrics with totals.
        """
        calls = [*self.model_calls, *(c for r in reports for c in r.model_calls)]
        tools = [t for r in reports for t in r.tools_used]
        return RequestMetrics(
            model=model,
            total_seconds=round(total_seconds, 3),
            phases=self.phases,
            model_calls=calls,
            tool_calls=len(tools),
            tool_seconds=round(sum(t.get("ms", 0) for t in tools) / 1000, 3),
            input_tokens=sum(c.input_tokens or 0 for c in calls),
            output_tokens=sum(c.output_tokens or 0 for c in calls),
        )
