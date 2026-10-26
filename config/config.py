"""Environment-backed runtime configuration for the RepWise backend."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """Backend settings shared by the API and command-line runner."""

    database_path: str = "data/repwise.sqlite3"
    ollama_model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    top_k: int = 5
    chunk_file: str | None = None

    @classmethod
    def from_environment(cls) -> "AppConfig":
        """Load project environment variables and return their current values."""
        load_dotenv()
        return cls(
            database_path=os.getenv("REPWISE_DATABASE_PATH", cls.database_path),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            ollama_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_url),
            top_k=int(os.getenv("REPWISE_TOP_K", cls.top_k)),
            chunk_file=os.getenv("REPWISE_CHUNK_FILE") or None,
        )
