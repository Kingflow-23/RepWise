"""A safe, temporary setup shared by the demo scripts.

Each script gets a fresh SQLite file holding the synthetic demo account, a fake
calendar with synthetic busy times, and the real Ollama, Photon and corpus
services configured in ``.env``. Real accounts, calendars and data are never
touched.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from config.config import AppConfig
from src.agents.health import OllamaHealthChecker
from src.db.demo import seed_demo_account
from src.db.repository import StudentRepository
from src.integrations.photon import PhotonClient
from src.retrieval.factory import build_plan_pipeline
from src.tools.calendar import BusyPeriod, FakeCalendar, default_week_start
from src.tools.context import Location, ToolContext, utc_now

DEFAULT_LOCATION = Location(label="Cachan, France", latitude=48.7908, longitude=2.3317)


def demo_calendar(timezone: str, plan_days: list[str]) -> FakeCalendar:
    """Build a calendar where Monday is busy all day and other days partly.

    Args:
        timezone: Time zone the busy times are written in.
        plan_days: Weekday names used by the plan, to pick the week.

    Returns:
        An in-memory calendar.
    """
    zone = ZoneInfo(timezone)
    monday = default_week_start(plan_days, utc_now().astimezone(zone).date())

    def busy(day_offset: int, start_hour: int, end_hour: int) -> BusyPeriod:
        """Make one busy period in the scheduled week."""
        day = monday + timedelta(days=day_offset)
        return BusyPeriod(
            start=datetime.combine(day, time(start_hour), zone),
            end=datetime.combine(day, time(end_hour), zone),
        )

    return FakeCalendar(
        [busy(0, 8, 18), busy(2, 6, 9), busy(4, 12, 14), busy(5, 9, 11)]
    )


def demo_context(
    config: AppConfig,
    database: Path,
    location: Location | None = DEFAULT_LOCATION,
    with_calendar: bool = True,
) -> tuple[ToolContext, str]:
    """Seed the demo account in ``database`` and build its tool context.

    Args:
        config: Settings for Ollama, Photon and the corpus.
        database: A temporary SQLite file.
        location: Location shared for the session, or ``None``.
        with_calendar: Whether to connect the fake calendar.

    Returns:
        The context and the demo plan's ID.
    """
    student_id, plan_id = seed_demo_account(database)
    repository = StudentRepository(database)
    # A spare available day, so a missed workout has somewhere to move.
    profile = repository.get_profile(student_id)
    repository.save_profile(
        student_id,
        profile.model_copy(
            update={"available_days": [*profile.available_days, "Saturday"]}
        ),
    )
    plan_days = [
        workout["day"]
        for week in repository.get_current_plan(student_id)["plan"]["weeks"]
        for workout in week["workouts"]
    ]
    context = ToolContext(
        student_id=student_id,
        repository=repository,
        config=config,
        health=OllamaHealthChecker(config.health_check_model, config.ollama_url),
        pipeline_factory=lambda: build_plan_pipeline(config),
        calendar=demo_calendar(config.timezone, plan_days) if with_calendar else None,
        places=PhotonClient(config.photon_url, config.http_user_agent),
        session_location=location,
    )
    return context, plan_id
