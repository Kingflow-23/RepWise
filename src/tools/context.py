"""The trusted values every tool runs with.

A tool never receives the student's identity, database, calendar or location
from the language model. Those come from a :class:`ToolContext`, which the
authenticated caller builds: the API after checking the bearer token, the MCP
server after checking its access token at startup, or a script on a temporary
demo database. The model only supplies the tool's own arguments.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from config.config import AppConfig
from src.agents.health import HealthChecker
from src.db.repository import StudentRepository
from src.schemas.models import Model

if TYPE_CHECKING:
    from src.generation.pipeline import PlanPipeline
    from src.integrations.photon import PhotonClient
    from src.tools.calendar import CalendarProvider

logger = logging.getLogger(__name__)


class Location(Model):
    """A point on the map, with a human-readable label.

    Attributes:
        label: Place name shown to the student, such as "Paris 11e".
        latitude: Latitude in degrees.
        longitude: Longitude in degrees.
    """

    label: str
    latitude: float
    longitude: float


def utc_now() -> datetime:
    """Return the current time in UTC.

    Returns:
        A timezone-aware ``datetime``.
    """
    return datetime.now(UTC)


@dataclass
class ToolContext:
    """Everything a tool may use, decided by trusted code, never by the model.

    Attributes:
        student_id: The authenticated account the tools act for.
        repository: Database access; every query is scoped to ``student_id``.
        config: Application settings (time zone, limits, service URLs).
        health: Decides whether text mentions pain or injury.
        pipeline_factory: Builds the RAG plan pipeline on first use, or
            ``None`` when plan generation and document search are unavailable.
        calendar: The student's connected external calendar, or ``None``.
        places: Client for the map service, or ``None`` when not configured.
        session_location: A location the client shared for this request only.
            It is used for searches and never stored.
        timezone: IANA time zone for dates and times shown to the student.
        clock: Returns the current time; tests replace it.
    """

    student_id: str
    repository: StudentRepository
    config: AppConfig
    health: HealthChecker
    pipeline_factory: Callable[[], "PlanPipeline"] | None = None
    calendar: "CalendarProvider | None" = None
    places: "PhotonClient | None" = None
    session_location: Location | None = None
    timezone: str = ""
    clock: Callable[[], datetime] = utc_now
    _pipeline: "PlanPipeline | None" = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Fill the time zone from the configuration when none was given."""
        if not self.timezone:
            self.timezone = self.config.timezone

    def pipeline(self) -> "PlanPipeline":
        """Return the plan pipeline, building it once on first use.

        Building it loads the corpus and indexes, which takes seconds, so
        requests that never need it do not pay for it.

        Returns:
            The configured :class:`~src.generation.pipeline.PlanPipeline`.

        Raises:
            RuntimeError: If no pipeline was configured for this context.
            OSError: If the corpus or index files cannot be read.
            ValueError: If the retrieval configuration is invalid.
        """
        if self._pipeline is None:
            if self.pipeline_factory is None:
                raise RuntimeError(
                    "Plan generation and document search are not configured. "
                    "Set REPWISE_CHUNK_FILE and start Ollama."
                )
            self._pipeline = self.pipeline_factory()
        return self._pipeline
