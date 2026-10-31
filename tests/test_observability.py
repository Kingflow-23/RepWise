"""Verify structured logging configuration and request-ID propagation."""

import logging

import pytest

from src.observability import configure_logging, request_id_context


def test_configure_logging_writes_rotating_service_log_with_request_id(
    tmp_path, monkeypatch
):
    """Write a service log containing both the active request ID and event.

    Args:
        tmp_path: Direct the service's rotating log file to a disposable directory.
        monkeypatch: Set the log level and output directory for this test.
    """
    service = "observability-test"
    monkeypatch.setenv("REPWISE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("REPWISE_LOG_DIR", str(tmp_path))

    configure_logging(service)
    token = request_id_context.set("request-test-123")
    try:
        logging.getLogger("test.observability").debug("test diagnostic event")
    finally:
        request_id_context.reset(token)

    handler = next(
        handler
        for handler in logging.getLogger().handlers
        if handler.get_name() == f"repwise:{service}"
    )
    handler.flush()
    content = (tmp_path / f"repwise-{service}.log").read_text(encoding="utf-8")
    assert "request_id=request-test-123" in content
    assert "test diagnostic event" in content

    logging.getLogger().removeHandler(handler)
    handler.close()


def test_configure_logging_rejects_unknown_log_level(monkeypatch):
    """Reject unsupported log levels instead of silently configuring logging.

    Args:
        monkeypatch: Set an invalid level in the logging configuration environment.
    """
    monkeypatch.setenv("REPWISE_LOG_LEVEL", "TRACE")

    with pytest.raises(ValueError, match="REPWISE_LOG_LEVEL"):
        configure_logging("invalid-level-test")
