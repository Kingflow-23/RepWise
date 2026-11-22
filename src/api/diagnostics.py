"""Readiness checks behind ``GET /api/v1/health``.

The health route separates two things:

* **configuration**: which retrieval strategy and which Ollama models the
  settings ask for;
* **availability**: whether the retriever really initialized, whether Ollama
  answers, and whether each configured model is installed.

The Ollama check only lists installed models (``GET /api/tags``) with a short
timeout. It never loads a model or starts a generation, so a health call stays
cheap. Nothing here changes the configured model or retriever.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from urllib import error as url_error
from urllib import request as url_request

from config.config import AppConfig

from ..retrieval.base import MissingRetriever

logger = logging.getLogger(__name__)

#: Default time allowed for the Ollama model listing, in seconds.
OLLAMA_CHECK_SECONDS = 2.0

#: Wrappers are followed at most this many levels to find the base retriever.
_MAX_WRAPPER_DEPTH = 5


class OllamaUnavailable(RuntimeError):
    """Ollama could not be reached or returned an unreadable model list."""


def retrieval_setup_error(retriever: object) -> str | None:
    """Return the retriever's setup error, looking through wrappers.

    The API wraps every retriever in ``QueryExpansionRetriever``, so a
    ``MissingRetriever`` created after a failed initialization is not the
    object the routes hold. This follows the ``retriever`` attribute of
    wrappers to find it.

    Args:
        retriever: The retriever used by the API, possibly wrapped.

    Returns:
        The stored setup message when the base retriever is a
        ``MissingRetriever``, otherwise ``None``.
    """
    current = retriever
    for _ in range(_MAX_WRAPPER_DEPTH + 1):
        if isinstance(current, MissingRetriever):
            return current.message
        inner = getattr(current, "retriever", None)
        if inner is None or inner is current:
            return None
        current = inner
    return None


def list_ollama_models(
    base_url: str, timeout_seconds: float = OLLAMA_CHECK_SECONDS
) -> list[str]:
    """List the models installed in Ollama, without loading any of them.

    Args:
        base_url: Ollama service URL, such as ``http://localhost:11434``.
        timeout_seconds: Maximum time for the request.

    Returns:
        Installed model names, as Ollama reports them (``name:tag``).

    Raises:
        OllamaUnavailable: If Ollama does not answer in time or the answer is
            not a model list.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with url_request.urlopen(url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except url_error.HTTPError as error:
        raise OllamaUnavailable(
            f"Ollama answered the model listing with HTTP {error.code}."
        ) from error
    except (url_error.URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        if isinstance(reason, TimeoutError):
            message = f"Ollama did not answer within {timeout_seconds:g} s."
        else:
            message = "Ollama refused or dropped the connection. Is `ollama serve` running?"
        raise OllamaUnavailable(message) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OllamaUnavailable("Ollama returned an unreadable model list.") from error
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise OllamaUnavailable("Ollama returned an unreadable model list.")
    names = []
    for entry in models:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("model")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def normalize_model_name(name: str) -> str:
    """Return a model name with an explicit tag (Ollama assumes ``latest``)."""
    name = name.strip()
    return name if ":" in name else f"{name}:latest"


def configured_models(config: AppConfig) -> dict[str, str]:
    """Return the Ollama model configured for each role the API uses."""
    return {
        "generation": config.ollama_model,
        "agents": config.agent_model,
        "query_analysis": config.query_expansion_model,
        "health_check": config.health_check_model,
    }


def health_report(
    config: AppConfig,
    retriever: object,
    list_models: Callable[[], list[str]],
) -> dict[str, object]:
    """Build the health response: configuration, then real availability.

    Args:
        config: Application settings (what is configured).
        retriever: The retriever the API uses, possibly wrapped.
        list_models: Returns the installed Ollama models, or raises
            :class:`OllamaUnavailable`. It must be bounded in time.

    Returns:
        A JSON-compatible mapping. ``status`` is ``ok`` only when retrieval is
        ready, Ollama answers and every configured model is installed;
        otherwise it is ``degraded`` and ``problems`` says why.
    """
    problems: list[str] = []
    retrieval_error = retrieval_setup_error(retriever)
    if retrieval_error:
        problems.append(f"Retrieval is not available: {retrieval_error}")

    roles = configured_models(config)
    installed: set[str] | None
    ollama_error: str | None = None
    try:
        installed = {normalize_model_name(name) for name in list_models()}
    except OllamaUnavailable as error:
        installed = None
        ollama_error = str(error)
        problems.append(f"Ollama is not reachable: {ollama_error}")

    models: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for role, name in roles.items():
        present = None if installed is None else normalize_model_name(name) in installed
        models[role] = {"name": name, "installed": present}
        if present is False and name not in missing:
            missing.append(name)
    for name in missing:
        used_by = ", ".join(role for role, model in roles.items() if model == name)
        problems.append(
            f"Model {name} is configured ({used_by}) but not installed in Ollama. "
            f"Install it with `ollama pull {name}` or change the configuration."
        )

    return {
        "status": "ok" if not problems else "degraded",
        "database": "ready",
        "retrieval": "ready" if retrieval_error is None else "not connected",
        "retrieval_error": retrieval_error,
        "retriever": config.retrieval_strategy,
        "model": config.ollama_model,
        "configuration": {
            "retrieval_strategy": config.retrieval_strategy,
            "embedding_model": config.embedding_model,
            "models": roles,
        },
        "ollama": {
            "status": "reachable" if installed is not None else "unreachable",
            "error": ollama_error,
        },
        "models": models,
        "problems": problems,
    }
