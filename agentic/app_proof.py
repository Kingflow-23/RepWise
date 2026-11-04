"""Prove the three tools work on your real RepWise account, in one command.

After generating a plan in the app, run from the repository root:

    python -m agentic.app_proof "Your account name"

This saves real feedback for your first planned workout, which then shows as
"Workout logged" in the app. Use it only on your own local account.
"""

import json
import sqlite3
import sys

from agentic import tools


def main() -> int:
    """Call all three tools on the named account and report the outcome."""
    if len(sys.argv) != 2:
        print('Usage: python -m agentic.app_proof "Your account name"')
        return 2
    try:
        account = tools.open_repository().get_account_by_name(sys.argv[1])
    except sqlite3.Error as error:
        print(f"{error}. Start the app and create an account first.")
        return 1
    if account is None:
        print(f"No account named {sys.argv[1]!r}. Use your exact sign-in name.")
        return 1
    student_id = account[0]

    profile = tools.get_user_profile(student_id)
    print(json.dumps(profile.model_dump(), indent=2))
    if not profile.ok:
        print("Save your profile in the app, then run this again.")
        return 1

    timetable = tools.get_timetable(student_id)
    if not timetable.ok or not timetable.planned_workouts:
        print("Generate a plan in the app, then run this again.")
        return 1
    workout = timetable.planned_workouts[0]
    print(f"Found {len(timetable.planned_workouts)} planned workouts.")

    saved = tools.save_workout_feedback(
        student_id,
        workout.plan_id,
        workout.week_number,
        workout.workout_index,
        "completed",
        "slightly_too_hard",
        "Saved by the agent tool",
    )
    print(json.dumps(saved.model_dump(), indent=2))
    if not saved.ok:
        return 1
    print(
        f"Proof complete. Refresh the app and open your latest plan: the {workout.day} "
        "workout shows 'Workout logged' with the comment 'Saved by the agent tool'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
