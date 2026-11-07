"""Environment-backed runtime configuration for the RepWise backend.

Every value that can differ between machines or deployments (models, paths,
service URLs, agent limits, retention) is read here, once, from environment
variables. The rest of the code receives an :class:`AppConfig` instead of
reading the environment itself, so a test can build one with plain arguments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL_CATALOG_PATH = REPOSITORY_ROOT / "config" / "tools.json"
PLACES_CONFIG_PATH = REPOSITORY_ROOT / "config" / "places.json"

TRUE_VALUES = {"true", "1", "yes"}
BOOLEAN_VALUES = TRUE_VALUES | {"false", "0", "no"}


def _read_bool(name: str, default: str) -> bool:
    """Read a true/false environment variable.

    Args:
        name: Environment variable name.
        default: Value used when the variable is absent.

    Returns:
        ``True`` for ``true``, ``1`` or ``yes`` (any case), otherwise ``False``.

    Raises:
        ValueError: If the value is not one of the accepted spellings.
    """
    value = os.getenv(name, default).casefold()
    if value not in BOOLEAN_VALUES:
        raise ValueError(f"{name} must be true, false, 1, 0, yes, or no.")
    return value in TRUE_VALUES


def _read_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a whole-number environment variable with a lower bound.

    Args:
        name: Environment variable name.
        default: Value used when the variable is absent or empty.
        minimum: Smallest accepted value.

    Returns:
        The parsed integer.

    Raises:
        ValueError: If the value is not an integer or is below ``minimum``.
    """
    raw = os.getenv(name) or str(default)
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a whole number, not {raw!r}.") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


@dataclass(frozen=True)
class AgentLimits:
    """Hard limits that keep every agent run bounded.

    Attributes:
        max_delegations: Most specialist steps the orchestrator may plan.
        max_replans: How many times a failed plan may be revised.
        max_tool_calls: Most tool calls one specialist may make per task.
        request_seconds: Wall-clock budget for one whole user request.
        specialist_seconds: Wall-clock budget for one specialist task.
        context_tokens: Context window requested from Ollama for agent calls.
    """

    max_delegations: int = 4
    max_replans: int = 1
    max_tool_calls: int = 5
    request_seconds: int = 420
    specialist_seconds: int = 240
    context_tokens: int = 16384


@dataclass(frozen=True)
class AppConfig:
    """Settings shared by the API, the agents, the MCP server and the scripts.

    Attributes:
        database_path: Local SQLite database location.
        ollama_model: Model that generates workout plans.
        agent_model: Tool-calling model used by the orchestrator and specialists.
        query_expansion_model: Small model for scope analysis and French rewrites.
        health_check_model: Small model that spots pain or injury mentions.
        ollama_comparison_model: Second model used in controlled comparisons.
        ollama_judge_model: Local model used to judge benchmark answers.
        ollama_url: Ollama service base URL.
        top_k: Minimum retrieval count configured for the application.
        chunk_file: Optional single chunk file override.
        retrieval_strategy: Default retrieval implementation name.
        embedding_model: Local embedding model identifier.
        embedding_index: Path to the saved semantic index.
        query_expansion: Whether to translate and expand queries in French.
        auth_secret: Secret used to sign and verify access tokens.
        speech_model: Local faster-whisper model for microphone input.
        timezone: Default IANA time zone for calendar work.
        episode_retention_days: How long episodic memory is kept.
        approval_minutes: How long a pending action can still be approved.
        photon_url: Base URL of the Photon (OpenStreetMap) geocoding service.
        http_user_agent: User-Agent sent to public map services.
        google_directory: Folder holding the Google OAuth client and tokens.
        agent_limits: Step and time limits for agent runs.
    """

    database_path: str = "data/repwise.sqlite3"
    ollama_model: str = "qwen3.5:9b"
    agent_model: str = "qwen3.5:9b"
    query_expansion_model: str = "qwen3.5:4b"
    health_check_model: str = "qwen3.5:4b"
    ollama_comparison_model: str = "gemma4:12b"
    ollama_judge_model: str = "qwen3.5:4b"
    ollama_url: str = "http://localhost:11434"
    top_k: int = 5
    chunk_file: str | None = None
    retrieval_strategy: str = "hybrid"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_index: str = "data/corpus/index/semantic.npz"
    query_expansion: bool = True
    auth_secret: str = ""
    speech_model: str = "base"
    timezone: str = "Europe/Paris"
    episode_retention_days: int = 90
    approval_minutes: int = 30
    photon_url: str = "https://photon.komoot.io"
    http_user_agent: str = "RepWise-course-project"
    google_directory: str = "data/google"
    agent_limits: AgentLimits = field(default_factory=AgentLimits)

    @property
    def resolved_database_path(self) -> Path:
        """Return the database path, resolved against the repository root.

        Returns:
            An absolute path. A relative ``database_path`` is read from the
            repository root, so scripts work from any working directory.
        """
        path = Path(self.database_path)
        return path if path.is_absolute() else REPOSITORY_ROOT / path

    @classmethod
    def from_environment(cls) -> "AppConfig":
        """Load project environment variables into an immutable configuration.

        Calls ``load_dotenv()`` before reading values. ``REPWISE_TOP_K`` is
        clamped to at least five; an absent or empty chunk-file override
        becomes ``None``. ``REPWISE_AGENT_MODEL`` falls back to
        ``OLLAMA_MODEL`` and ``REPWISE_HEALTH_CHECK_MODEL`` falls back to the
        query-expansion model, so a minimal ``.env`` keeps working.

        Returns:
            Configuration populated from environment variables, falling back
            to the class defaults.

        Raises:
            ValueError: If a numeric or boolean variable has an invalid value.
        """
        load_dotenv()
        ollama_model = os.getenv("OLLAMA_MODEL", cls.ollama_model)
        query_expansion_model = os.getenv(
            "OLLAMA_QUERY_EXPANSION_MODEL", cls.query_expansion_model
        )
        defaults = AgentLimits()
        return cls(
            database_path=os.getenv("REPWISE_DATABASE_PATH", cls.database_path),
            ollama_model=ollama_model,
            agent_model=os.getenv("REPWISE_AGENT_MODEL") or ollama_model,
            query_expansion_model=query_expansion_model,
            health_check_model=(
                os.getenv("REPWISE_HEALTH_CHECK_MODEL") or query_expansion_model
            ),
            ollama_comparison_model=os.getenv(
                "OLLAMA_COMPARISON_MODEL", cls.ollama_comparison_model
            ),
            ollama_judge_model=os.getenv("OLLAMA_JUDGE_MODEL", cls.ollama_judge_model),
            ollama_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_url),
            top_k=max(5, _read_int("REPWISE_TOP_K", cls.top_k)),
            chunk_file=os.getenv("REPWISE_CHUNK_FILE") or None,
            retrieval_strategy=os.getenv("REPWISE_RETRIEVAL", cls.retrieval_strategy),
            embedding_model=os.getenv("REPWISE_EMBEDDING_MODEL", cls.embedding_model),
            embedding_index=os.getenv("REPWISE_EMBEDDING_INDEX", cls.embedding_index),
            query_expansion=_read_bool("REPWISE_QUERY_EXPANSION", "true"),
            auth_secret=os.getenv("REPWISE_AUTH_SECRET", ""),
            speech_model=os.getenv("REPWISE_SPEECH_MODEL", cls.speech_model),
            timezone=os.getenv("REPWISE_TIMEZONE", cls.timezone),
            episode_retention_days=_read_int(
                "REPWISE_EPISODE_RETENTION_DAYS", cls.episode_retention_days
            ),
            approval_minutes=_read_int(
                "REPWISE_APPROVAL_MINUTES", cls.approval_minutes
            ),
            photon_url=os.getenv("REPWISE_PHOTON_URL", cls.photon_url).rstrip("/"),
            http_user_agent=os.getenv("REPWISE_HTTP_USER_AGENT", cls.http_user_agent),
            google_directory=os.getenv("REPWISE_GOOGLE_DIR", cls.google_directory),
            agent_limits=AgentLimits(
                max_delegations=_read_int(
                    "REPWISE_AGENT_MAX_DELEGATIONS", defaults.max_delegations
                ),
                max_replans=_read_int(
                    "REPWISE_AGENT_MAX_REPLANS", defaults.max_replans, minimum=0
                ),
                max_tool_calls=_read_int(
                    "REPWISE_AGENT_MAX_TOOL_CALLS", defaults.max_tool_calls
                ),
                request_seconds=_read_int(
                    "REPWISE_AGENT_REQUEST_SECONDS", defaults.request_seconds
                ),
                specialist_seconds=_read_int(
                    "REPWISE_AGENT_SPECIALIST_SECONDS", defaults.specialist_seconds
                ),
                context_tokens=_read_int(
                    "REPWISE_AGENT_CONTEXT_TOKENS", defaults.context_tokens
                ),
            ),
        )
