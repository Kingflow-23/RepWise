"""Environment-backed runtime configuration for the RepWise backend."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """Environment-backed settings shared by the API and command-line runner.

    Attributes:
        database_path: Local SQLite database location.
        ollama_model: Default generation model.
        ollama_comparison_model: Second model used in controlled comparisons.
        ollama_url: Ollama service base URL.
        top_k: Minimum retrieval count configured for the application.
        chunk_file: Optional single chunk file override.
        retrieval_strategy: Default retrieval implementation name.
        embedding_model: Local embedding model identifier.
        embedding_index: Path to the saved semantic index.
        auth_secret: Secret used to configure authentication.
    """

    database_path: str = "data/repwise.sqlite3"
    ollama_model: str = "llama3.2"
    ollama_comparison_model: str = "qwen2.5:7b"
    ollama_url: str = "http://localhost:11434"
    top_k: int = 5
    chunk_file: str | None = None
    retrieval_strategy: str = "hybrid"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_index: str = "data/corpus/index/semantic.npz"
    auth_secret: str = ""

    @classmethod
    def from_environment(cls) -> "AppConfig":
        """Load project environment variables into an immutable configuration.

        Calls ``load_dotenv()`` before reading values. ``REPWISE_TOP_K`` is
        parsed as an integer and clamped to at least five; an absent or empty
        chunk-file override becomes ``None``.

        Returns:
            Configuration populated from environment variables, falling back
            to the class defaults.

        Raises:
            ValueError: If ``REPWISE_TOP_K`` is present but is not an integer.
        """
        load_dotenv()
        return cls(
            database_path=os.getenv("REPWISE_DATABASE_PATH", cls.database_path),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            ollama_comparison_model=os.getenv(
                "OLLAMA_COMPARISON_MODEL", cls.ollama_comparison_model
            ),
            ollama_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_url),
            top_k=max(5, int(os.getenv("REPWISE_TOP_K", cls.top_k))),
            chunk_file=os.getenv("REPWISE_CHUNK_FILE") or None,
            retrieval_strategy=os.getenv("REPWISE_RETRIEVAL", "hybrid"),
            embedding_model=os.getenv(
                "REPWISE_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
            ),
            embedding_index=os.getenv(
                "REPWISE_EMBEDDING_INDEX", "data/corpus/index/semantic.npz"
            ),
            auth_secret=os.getenv("REPWISE_AUTH_SECRET", ""),
        )
