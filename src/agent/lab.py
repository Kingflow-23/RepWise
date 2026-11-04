"""Temporary authenticated backend for the local MCP notebook lab."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time

import httpx
from mcp import StdioServerParameters

from config.config import AppConfig

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LabSession:
    api_url: str
    user_id: str
    access_token: str = field(repr=False)

    def server_parameters(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=sys.executable, args=["-m", "src.mcp_server.server"], cwd=str(ROOT),
            env={"REPWISE_API_URL": self.api_url, "REPWISE_MCP_USER_ID": self.user_id,
                 "REPWISE_MCP_ACCESS_TOKEN": self.access_token, "PYTHONIOENCODING": "utf-8"},
        )


@contextmanager
def local_lab(model: str = "qwen3.5:4b"):
    """Use real corpus and Ollama, with a disposable account and database.

    BM25 intentionally avoids requiring a semantic index for the MCP exercise.
    The regular API, database and .env are not changed. Processes and files are
    cleaned up even when the agent or verification raises an error.
    """
    config = AppConfig.from_environment()
    with httpx.Client(base_url=config.ollama_url, timeout=10) as client:
        tags = client.get("/api/tags")
        tags.raise_for_status()
        installed = [item["name"] for item in tags.json()["models"]]
    if model not in installed:
        raise RuntimeError(f"Install the lab model first: ollama pull {model}")
    if not config.chunk_file:
        raise RuntimeError("Set REPWISE_CHUNK_FILE to a valid corpus chunk before running the lab")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    api_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="repwise-mcp-lab-") as folder:
        env = {**os.environ, "REPWISE_DATABASE_PATH": str(Path(folder) / "lab.sqlite3"),
               "REPWISE_LOG_DIR": str(Path(folder) / "logs"),
               "REPWISE_AUTH_SECRET": secrets.token_hex(32),
               "REPWISE_RETRIEVAL": "bm25", "REPWISE_QUERY_EXPANSION": "false",
               "OLLAMA_MODEL": model, "OLLAMA_QUERY_EXPANSION_MODEL": model,
               "PYTHONIOENCODING": "utf-8"}
        with (Path(folder) / "api.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1",
                 "--port", str(port), "--log-level", "warning"],
                cwd=ROOT, env=env, stdout=log, stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                with httpx.Client(base_url=api_url, timeout=10) as api:
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError("Lab API exited before startup; check project dependencies")
                        try:
                            health = api.get("/api/v1/health")
                            if health.status_code == 200:
                                if health.json()["retrieval"] != "ready":
                                    raise RuntimeError(f"Corpus is not ready: {health.json().get('retrieval_error')}")
                                break
                        except httpx.TransportError:
                            pass
                        time.sleep(0.25)
                    else:
                        raise TimeoutError("Lab API did not start within 60 seconds")
                    account = api.post("/api/v1/auth/register", json={
                        "full_name": "MCP Lab User", "password": secrets.token_urlsafe(24),
                    })
                    account.raise_for_status()
                    identity = account.json()
                    session = LabSession(api_url, identity["user_id"], identity["access_token"])
                    profile = api.put(f"/api/v1/students/{session.user_id}/profile", headers={
                        "Authorization": f"Bearer {session.access_token}",
                    }, json={"experience": "beginner", "goal": "Build leg muscles",
                             "available_days": ["Monday"], "session_minutes": 45,
                             "equipment": ["bodyweight"]})
                    profile.raise_for_status()
                yield session
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
