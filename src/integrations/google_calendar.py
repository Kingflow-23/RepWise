"""Connect RepWise to a student's Google Calendar (the external calendar).

Set up once: create a Google Cloud project, enable the Calendar API, create an
OAuth client of type "Desktop app", and save its JSON as
``<REPWISE_GOOGLE_DIR>/client_secret.json`` (``data/google`` by default). Then
connect an account with ``python -m scripts.connect_google_calendar connect "Name"``.

Scopes are minimal: free/busy times (no event titles or details) and creating
events. Tokens are stored per account in ``<REPWISE_GOOGLE_DIR>/tokens/``, which
git ignores. That suits a local deployment; a shared one would need a secret store.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import httpx

from config.config import REPOSITORY_ROOT, AppConfig
from src.tools.calendar import BusyPeriod, CalendarError

logger = logging.getLogger(__name__)

API_URL = "https://www.googleapis.com/calendar/v3"
TIMEOUT_SECONDS = 15
SCOPES = [
    "https://www.googleapis.com/auth/calendar.freebusy",
    "https://www.googleapis.com/auth/calendar.events",
]


def google_directory(config: AppConfig) -> Path:
    """Return the folder holding the OAuth client file and tokens.

    Args:
        config: Application settings (``REPWISE_GOOGLE_DIR``).

    Returns:
        An absolute path; relative settings are read from the repository root.
    """
    path = Path(config.google_directory)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def token_path(config: AppConfig, student_id: str) -> Path:
    """Return where one account's Google token is stored.

    Args:
        config: Application settings.
        student_id: Account ID.

    Returns:
        The token file path (it may not exist yet).
    """
    return google_directory(config) / "tokens" / f"{student_id}.json"


class GoogleCalendar:
    """A :class:`~src.tools.calendar.CalendarProvider` for the primary calendar."""

    def __init__(
        self, access_token: Callable[[], str], client: httpx.Client | None = None
    ):
        """Configure the provider.

        Args:
            access_token: Returns a valid OAuth access token, refreshing it if needed.
            client: Optional HTTP client, used by tests.
        """
        self.access_token = access_token
        self.client = client or httpx.Client(timeout=TIMEOUT_SECONDS)

    def _post(self, url: str, body: dict) -> httpx.Response:
        """Send one authenticated request. Errors never include the token.

        Args:
            url: Path under the Calendar API.
            body: JSON body.

        Returns:
            The HTTP response.

        Raises:
            CalendarError: If Google cannot be reached.
        """
        try:
            return self.client.post(
                f"{API_URL}{url}",
                json=body,
                headers={"Authorization": f"Bearer {self.access_token()}"},
            )
        except httpx.HTTPError as error:
            raise CalendarError(f"Google Calendar did not answer: {error}") from error

    def busy_periods(self, start: datetime, end: datetime) -> list[BusyPeriod]:
        """Read busy times of the primary calendar.

        Args:
            start: Start of the range (timezone-aware).
            end: End of the range (timezone-aware).

        Returns:
            Busy periods.

        Raises:
            CalendarError: If Google refuses or cannot read the calendar. An
                error never reads as "free", because Google then sends an
                empty busy list together with an ``errors`` field.
        """
        response = self._post(
            "/freeBusy",
            {
                "timeMin": start.isoformat(),
                "timeMax": end.isoformat(),
                "items": [{"id": "primary"}],
            },
        )
        if response.status_code != 200:
            raise CalendarError(f"Google Calendar returned {response.status_code}.")
        calendar = response.json().get("calendars", {}).get("primary")
        if calendar is None or calendar.get("errors"):
            raise CalendarError("Google Calendar could not read the busy times.")
        return [BusyPeriod.model_validate(item) for item in calendar.get("busy", [])]

    def create_event(
        self, key: str, title: str, description: str, start: datetime, end: datetime
    ) -> str:
        """Create an event with a fixed ID, so a retry cannot duplicate it.

        Args:
            key: Event ID to use (lowercase hexadecimal).
            title: Event title.
            description: Event description.
            start: Event start.
            end: Event end.

        Returns:
            The event ID.

        Raises:
            CalendarError: If Google refuses the event.
        """
        response = self._post(
            "/calendars/primary/events",
            {
                "id": key,
                "summary": title,
                "description": description,
                "start": {"dateTime": start.isoformat()},
                "end": {"dateTime": end.isoformat()},
            },
        )
        if response.status_code == 409:
            logger.info("Calendar event already exists")
            return key
        if response.status_code not in (200, 201):
            raise CalendarError(f"Google Calendar returned {response.status_code}.")
        return response.json()["id"]


def load_google_calendar(config: AppConfig, student_id: str) -> GoogleCalendar | None:
    """Return the account's Google Calendar, or ``None`` if it is not connected.

    Args:
        config: Application settings.
        student_id: Account whose token is loaded.

    Returns:
        A provider whose token refreshes itself, or ``None``.
    """
    path = token_path(config, student_id)
    if not path.exists():
        return None

    from google.auth.exceptions import GoogleAuthError, TransportError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    credentials = Credentials.from_authorized_user_file(str(path), SCOPES)

    def access_token() -> str:
        """Return a valid token, refreshing and re-saving it when expired."""
        if not credentials.valid:
            try:
                credentials.refresh(Request())
            except (GoogleAuthError, TransportError) as error:
                raise CalendarError(
                    f"Reconnect the calendar, as its sign-in stopped working: {error}"
                ) from error
            path.write_text(credentials.to_json(), encoding="utf-8")
        return credentials.token

    return GoogleCalendar(access_token)


def connect(config: AppConfig, student_id: str) -> Path:
    """Open Google's consent screen in a browser and save the account's token.

    Args:
        config: Application settings.
        student_id: Account to connect.

    Returns:
        Where the token was saved.

    Raises:
        FileNotFoundError: If the OAuth client file is missing.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    client_file = google_directory(config) / "client_secret.json"
    if not client_file.exists():
        raise FileNotFoundError(f"Save your Google OAuth client file as {client_file}")
    flow = InstalledAppFlow.from_client_secrets_file(str(client_file), SCOPES)
    credentials = flow.run_local_server(port=0)
    path = token_path(config, student_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(credentials.to_json(), encoding="utf-8")
    return path
