"""Connect a RepWise account to Google Calendar, or check the connection.

Setup and scopes are described in ``src/integrations/google_calendar.py``.
From the repository root:

    python -m scripts.connect_google_calendar connect "Your account name"
    python -m scripts.connect_google_calendar check "Your account name"

``check`` only reads, and prints how many busy periods it found this week.
"""

from __future__ import annotations

import sys
from datetime import timedelta

from config.config import AppConfig
from src.db.repository import StudentRepository
from src.integrations.google_calendar import connect, load_google_calendar
from src.tools.context import utc_now


def main() -> int:
    """Run the command line.

    Returns:
        0 on success, 1 for an unknown account or missing connection, 2 for
        wrong usage.
    """
    if len(sys.argv) != 3 or sys.argv[1] not in {"connect", "check"}:
        print(__doc__)
        return 2
    config = AppConfig.from_environment()
    account = StudentRepository(config.resolved_database_path).get_account_by_name(
        sys.argv[2]
    )
    if account is None:
        print(f"No account named {sys.argv[2]!r}. Use your exact sign-in name.")
        return 1
    student_id = account[0]
    if sys.argv[1] == "connect":
        try:
            path = connect(config, student_id)
        except FileNotFoundError as error:
            print(error)
            return 1
        print(f"Connected. Token saved in {path}")
        return 0
    calendar = load_google_calendar(config, student_id)
    if calendar is None:
        print("Not connected yet. Run the connect command first.")
        return 1
    start = utc_now()
    busy = calendar.busy_periods(start, start + timedelta(days=7))
    print(f"Connected. Found {len(busy)} busy periods in the next 7 days.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
