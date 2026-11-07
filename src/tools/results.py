"""The one failure shape shared by every tool.

Every tool returns either its own success model (with ``ok: true``) or a
:class:`ToolFailure` (with ``ok: false``). ``code`` is a stable, machine-readable
word that agents and tests can branch on; ``error`` explains it in plain words.
The codes each tool can return are listed in ``config/tools.json``.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from functools import wraps
from typing import Literal, TypeVar

from src.schemas.models import Model

logger = logging.getLogger(__name__)


class ToolFailure(Model):
    """Any tool's output when it could not do what was asked.

    Attributes:
        ok: Always ``False``, so callers can tell failures from results.
        code: Stable reason, such as ``plan_not_found`` or ``storage_error``.
        error: Explanation that can be shown to the student.
    """

    ok: Literal[False] = False
    code: str
    error: str


def failure(code: str, error: str) -> ToolFailure:
    """Build a :class:`ToolFailure`.

    Args:
        code: Stable reason code.
        error: Human-readable explanation.

    Returns:
        The failure result.
    """
    return ToolFailure(code=code, error=error)


STUDENT_NOT_FOUND = "student_not_found"
PLAN_NOT_FOUND = "plan_not_found"
WORKOUT_NOT_FOUND = "workout_not_found"

ResultT = TypeVar("ResultT")


def report_storage_errors(tool: Callable[..., ResultT]) -> Callable[..., ResultT]:
    """Turn a database failure inside a tool into a ``storage_error`` result.

    A model cannot do anything useful with a Python traceback, but it can tell
    the student that saving failed. The error is still logged in full.

    Args:
        tool: The tool function to wrap.

    Returns:
        A function with the same signature that never raises a database error.
    """

    @wraps(tool)
    def wrapper(*args, **kwargs):
        """Run the tool and convert database errors into a failure result."""
        try:
            return tool(*args, **kwargs)
        except (OSError, sqlite3.Error):
            logger.exception("Tool storage failure tool=%s", tool.__name__)
            return failure("storage_error", "The database could not be used.")

    return wrapper
