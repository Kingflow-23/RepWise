"""Fixtures shared by the agent, tool, MCP and API tests.

Everything here is deterministic: a temporary SQLite database with the synthetic
demo account, a fixed clock, a fixed health checker and a scripted chat model
standing in for Ollama. Tests that need the real model live in
``tests/integration`` and are opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage

from config.config import AppConfig
from src.agents.health import FixedHealthChecker
from src.db.demo import seed_demo_account
from src.db.repository import StudentRepository
from src.tools.context import ToolContext

PARIS = ZoneInfo("Europe/Paris")
# A Wednesday. The demo plan's Monday has passed, so slots start the next Monday.
NOW = datetime(2030, 1, 9, 10, 0, tzinfo=PARIS)
NEXT_MONDAY = date(2030, 1, 14)
# Stand-in for the health model: these phrases count as a health concern.
HEALTH_PHRASES = ("pain", "hurt", "injur", "douleur", "mal au", "mal à", "blessure")


@dataclass
class Demo:
    """The demo account and a trusted context for it."""

    context: ToolContext
    student_id: str
    plan_id: str
    repository: StudentRepository
    health: FixedHealthChecker


def make_context(
    repository: StudentRepository, student_id: str, **extra
) -> ToolContext:
    """Build a deterministic tool context.

    Args:
        repository: Database access.
        student_id: Account to act for.
        **extra: Overrides, such as ``calendar`` or ``pipeline_factory``.

    Returns:
        The context, with the fixed clock and Paris time zone.
    """
    values = {
        "student_id": student_id,
        "repository": repository,
        "config": AppConfig(
            database_path=repository.database_path, timezone="Europe/Paris"
        ),
        "health": FixedHealthChecker(HEALTH_PHRASES),
        "clock": lambda: NOW,
    }
    values.update(extra)
    return ToolContext(**values)


@pytest.fixture
def demo(tmp_path) -> Demo:
    """A temporary database holding the synthetic account and its active plan."""
    database = tmp_path / "demo.sqlite3"
    student_id, plan_id = seed_demo_account(database)
    repository = StudentRepository(database)
    context = make_context(repository, student_id)
    return Demo(context, student_id, plan_id, repository, context.health)


@pytest.fixture
def other_student(demo) -> str:
    """A second account in the same database, to test isolation."""
    return seed_demo_account(demo.repository.database_path, full_name="Other User")[0]


def tool_call(name: str, call_id: str | None = None, **arguments) -> AIMessage:
    """Build a model reply asking for one tool.

    Args:
        name: Tool name.
        call_id: Optional call ID.
        **arguments: Tool arguments.

    Returns:
        An ``AIMessage`` with one tool call.
    """
    return AIMessage(
        "",
        tool_calls=[{"name": name, "args": arguments, "id": call_id or f"call-{name}"}],
    )


class ScriptedPlanner:
    """Return prepared structured outputs in order, then raise."""

    def __init__(self, plans: list):
        """Keep the queue shared with the model."""
        self.plans = plans

    def invoke(self, _messages):
        """Return the next plan, or raise when an exception was queued."""
        item = self.plans.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class ScriptedModel:
    """A stand-in for ChatOllama: replies come from lists, in order.

    ``replies`` answers ``invoke`` (specialist turns and the final answer);
    ``plans`` answers the planner (``with_structured_output``). Every call is
    recorded so tests can check what the model was shown.
    """

    def __init__(self, replies=(), plans=()):
        """Prepare the queues."""
        self.replies = list(replies)
        self.plans = list(plans)
        self.seen: list[list] = []
        self.bound: list[list[str]] = []

    def bind_tools(self, tools):
        """Record which tools a specialist was offered."""
        self.bound.append([tool["function"]["name"] for tool in tools])
        return self

    def with_structured_output(self, _schema, **_options):
        """Return the planner reading the shared plan queue."""
        return ScriptedPlanner(self.plans)

    def invoke(self, messages):
        """Return the next reply, or raise a queued exception."""
        self.seen.append(list(messages))
        if not self.replies:
            return AIMessage("Done.")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
