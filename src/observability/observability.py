"""Shared application logging with per-process rotating log files."""

from __future__ import annotations

import contextvars
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

request_id_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "repwise_request_id", default="-"
)
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class RequestContextFilter(logging.Filter):
    """Attach the active request ID to log records that lack one.

    Existing non-default ``request_id`` values are preserved so explicit
    per-record context is not overwritten.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Populate ``record.request_id`` and allow the record through.

        Args:
            record: Log record being processed by a handler.

        Returns:
            Always ``True``; this filter enriches records but never suppresses
            them.
        """
        if not getattr(record, "request_id", None) or record.request_id == "-":
            record.request_id = request_id_context.get()
        return True


def configure_logging(service: str) -> None:
    """Configure root console and per-service rotating-file handlers.

    Handler names prevent duplicate installation on repeated calls. The level
    and directory are read from ``REPWISE_LOG_LEVEL`` and ``REPWISE_LOG_DIR``,
    defaulting to ``INFO`` and ``logs``.

    Args:
        service: Service label used in the log filename and handler name.

    Returns:
        ``None``.

    Raises:
        ValueError: If ``REPWISE_LOG_LEVEL`` is not a supported logging level.
        OSError: If the log directory cannot be created or opened.
    """
    level_name = os.getenv("REPWISE_LOG_LEVEL", "INFO").upper()
    level = LOG_LEVELS.get(level_name)
    if level is None:
        raise ValueError(
            "REPWISE_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )

    log_dir = Path(os.getenv("REPWISE_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] [request_id=%(request_id)s] %(message)s"
    )

    if not any(handler.get_name() == "repwise:console" for handler in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.addFilter(RequestContextFilter())
        console.set_name("repwise:console")
        root.addHandler(console)

    if not any(handler.get_name() == f"repwise:{service}" for handler in root.handlers):
        file_handler = RotatingFileHandler(
            log_dir / f"repwise-{service}.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestContextFilter())
        file_handler.set_name(f"repwise:{service}")
        root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Logging configured service=%s level=%s directory=%s",
        service,
        level_name,
        log_dir,
    )
