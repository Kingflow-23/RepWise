"""Typed generation failures, so callers can tell *why* a model call failed.

Only some failures are worth a second attempt. A model that returned unusable
JSON may do better with a correction prompt; a stopped Ollama server, a model
that is not installed, or a call that already used its whole time budget will
not, and retrying them only makes the student wait longer.

Each failure carries the pipeline ``status`` it maps to and whether a retry
can help. Messages are written for the student and the API client: they never
include stack traces, file paths or raw provider output.
"""

from __future__ import annotations

import json


class GenerationFailure(RuntimeError):
    """A model call failed; ``status`` names the cause for the API."""

    status = "generation_error"
    retryable = True

    def __init__(self, message: str, *, detail: str | None = None):
        """Store a client-safe message and an optional internal detail.

        Args:
            message: Text safe to show to the student and API clients.
            detail: Provider or parser detail for logs and correction prompts;
                never returned by the API.
        """
        super().__init__(message)
        self.detail = detail


class ProviderUnavailable(GenerationFailure):
    """Ollama could not be reached or could not serve the request."""

    status = "provider_unavailable"
    retryable = False


class ModelNotFound(GenerationFailure):
    """The configured model is not installed in Ollama."""

    status = "model_not_found"
    retryable = False


class GenerationTimeout(GenerationFailure):
    """The model did not answer within the allowed time."""

    status = "generation_timeout"
    retryable = False


class InvalidModelOutput(GenerationFailure):
    """The model answered, but not with a readable plan. A retry may help."""

    status = "generation_error"
    retryable = True


def _is_timeout(error: BaseException) -> bool:
    """Return whether ``error`` is a client-side timeout."""
    if isinstance(error, TimeoutError):
        return True
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with the ollama client
        return False
    return isinstance(error, httpx.TimeoutException)


def _is_model_not_found(error: BaseException) -> bool:
    """Return whether Ollama reported that the requested model is missing."""
    status_code = getattr(error, "status_code", None)
    text = str(getattr(error, "error", error)).casefold()
    if status_code == 404:
        return True
    return "model" in text and "not found" in text


def classify_provider_error(
    error: BaseException, *, model: str, timeout_seconds: float
) -> GenerationFailure:
    """Turn an exception raised during an Ollama call into a typed failure.

    Args:
        error: What the Ollama client or the response parsing raised.
        model: Configured model name, shown in the model-not-found message.
        timeout_seconds: The client's time budget, shown in the timeout message.

    Returns:
        A :class:`GenerationFailure` subclass with a client-safe message.
    """
    if isinstance(error, GenerationFailure):
        return error
    if _is_timeout(error):
        return GenerationTimeout(
            f"The model did not finish within {timeout_seconds:g} s. "
            "Nothing was saved; try again later or with a simpler request."
        )
    # ollama.ResponseError: the server answered with an HTTP error.
    if type(error).__name__ == "ResponseError":
        if _is_model_not_found(error):
            return ModelNotFound(
                f"The configured model {model} is not installed in Ollama. "
                f"Install it with `ollama pull {model}` or change the configuration."
            )
        return ProviderUnavailable(
            "Ollama could not serve the request "
            f"(HTTP {getattr(error, 'status_code', 'error')}). Check the Ollama server."
        )
    # The ollama client raises the built-in ConnectionError when nothing answers.
    if isinstance(error, ConnectionError):
        return ProviderUnavailable(
            "Ollama is not reachable. Start it with `ollama serve`, then try again."
        )
    if isinstance(error, (ValueError, json.JSONDecodeError)):
        # Includes pydantic.ValidationError, a ValueError subclass.
        return InvalidModelOutput(
            "The model did not return a readable plan.", detail=str(error)
        )
    return InvalidModelOutput(
        "The model call failed unexpectedly.",
        detail=f"{type(error).__name__}: {error}",
    )
