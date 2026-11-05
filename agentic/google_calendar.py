"""Connect RepWise to a student's Google Calendar.

Set up once: create a Google Cloud project, enable the Calendar API, create an OAuth
client of type "Desktop app", and save its JSON as ``data/google/client_secret.json``.
Then, from the repository root:

    python -m agentic.google_calendar connect "Your account name"
    python -m agentic.google_calendar check "Your account name"

``check`` only reads, and prints how many busy periods it found this week. Tokens are
stored unencrypted in ``data/google/tokens/``, which git ignores: this suits a local
demo, and a shared deployment would need a proper secret store.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from agentic import tools
from agentic.calendar_tools import BusyPeriod, CalendarError

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://www.googleapis.com/calendar/v3"
TIMEOUT_SECONDS = 15
# Busy times only, without titles, plus the student's events. No access to others.
SCOPES = [
    "https://www.googleapis.com/auth/calendar.freebusy",
    "https://www.googleapis.com/auth/calendar.events",
]


def google_directory() -> Path:
    return Path(os.environ.get("REPWISE_GOOGLE_DIR", ROOT / "data" / "google"))


def token_path(student_id: str) -> Path:
    return google_directory() / "tokens" / f"{student_id}.json"


class GoogleCalendar:
    """A :class:`CalendarProvider` for the student's primary Google Calendar."""

    def __init__(
        self, access_token: Callable[[], str], client: httpx.Client | None = None
    ):
        self.access_token = access_token
        self.client = client or httpx.Client(timeout=TIMEOUT_SECONDS)

    def _post(self, url: str, body: dict) -> httpx.Response:
        """Send one request. Errors never include the token."""
        try:
            return self.client.post(
                f"{API_URL}{url}",
                json=body,
                headers={"Authorization": f"Bearer {self.access_token()}"},
            )
        except httpx.HTTPError as error:
            raise CalendarError(f"Google Calendar did not answer: {error}") from error

    def busy_periods(self, start: datetime, end: datetime) -> list[BusyPeriod]:
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
        # An error here comes with an empty busy list, which must not read as free.
        if calendar is None or calendar.get("errors"):
            raise CalendarError("Google Calendar could not read the busy times.")
        return [BusyPeriod.model_validate(item) for item in calendar.get("busy", [])]

    def create_event(
        self, key: str, title: str, description: str, start: datetime, end: datetime
    ) -> str:
        response = self._post(
            "/calendars/primary/events",
            {
                # A fixed ID makes Google reject a retry instead of duplicating it.
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


def load_google_calendar(student_id: str) -> GoogleCalendar | None:
    """Return the student's calendar, or None if they have not connected one."""
    path = token_path(student_id)
    if not path.exists():
        return None

    from google.auth.exceptions import GoogleAuthError, TransportError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    credentials = Credentials.from_authorized_user_file(str(path), SCOPES)

    def access_token() -> str:
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


def connect(student_id: str) -> None:
    """Open Google's consent screen in a browser and save the student's token."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    client_file = google_directory() / "client_secret.json"
    if not client_file.exists():
        raise SystemExit(f"Save your Google OAuth client file as {client_file}")
    flow = InstalledAppFlow.from_client_secrets_file(str(client_file), SCOPES)
    credentials = flow.run_local_server(port=0)
    path = token_path(student_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(credentials.to_json(), encoding="utf-8")
    print(f"Connected. Token saved in {path}")


def check(student_id: str) -> None:
    """Prove the connection works, without writing anything."""
    calendar = load_google_calendar(student_id)
    if calendar is None:
        raise SystemExit("Not connected yet. Run the connect command first.")
    start = datetime.now(UTC)
    busy = calendar.busy_periods(start, start + timedelta(days=7))
    print(f"Connected. Found {len(busy)} busy periods in the next 7 days.")


def main() -> int:
    commands = {"connect": connect, "check": check}
    if len(sys.argv) != 3 or sys.argv[1] not in commands:
        print(__doc__)
        return 2
    account = tools.open_repository().get_account_by_name(sys.argv[2])
    if account is None:
        print(f"No account named {sys.argv[2]!r}. Use your exact sign-in name.")
        return 1
    commands[sys.argv[1]](account[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
