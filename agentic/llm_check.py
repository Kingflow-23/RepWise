"""Prove your LLM chooses the right tools, using requests in plain language.

The model reads each request and the schemas in
``docs/agent/s2_tool_schemas.json``, then
decides which tools to call and with which values. Our code runs those calls on a
temporary database with one synthetic account, and checks what happened.

Needs Ollama running. From the repository root:

    python -m agentic.llm_check
"""

import json
import os
import sys
import tempfile
from pathlib import Path

from ollama import Client

from agentic import tools
from agentic.mcp_demo_client import seed_demo_account
from config.config import AppConfig

SCHEMAS = Path(__file__).resolve().parents[1] / "docs" / "agent" / "s2_tool_schemas.json"
TOOLS = {
    "get_user_profile": tools.get_user_profile,
    "get_timetable": tools.get_timetable,
    "save_workout_feedback": tools.save_workout_feedback,
}
SAVE = "save_workout_feedback"

# ``tool`` must be called. ``saved`` is the feedback that must exist afterwards,
# as (workout_index, completion_status, allowed difficulties), or None for no write.
CASES = [
    {
        "request": "What is my training goal, and what equipment do I have?",
        "tool": "get_user_profile",
        "saved": None,
    },
    {
        "request": "Which workouts do I have this week?",
        "tool": "get_timetable",
        "saved": None,
    },
    {
        "request": "Please log my Monday workout: I completed it, and it was "
        "slightly too hard. The last set was difficult.",
        "tool": SAVE,
        "saved": (0, "completed", {"slightly_too_hard"}),
    },
    {
        "request": "Log Wednesday's workout: I only finished half of it, and it "
        "felt way too easy.",
        "tool": SAVE,
        "saved": (1, "partially_completed", {"much_too_easy", "slightly_too_easy"}),
    },
    {
        "request": "I'm thinking about Monday's workout. What does it focus on?",
        "tool": "get_timetable",
        "saved": None,
    },
]


def load_tool_definitions() -> list[dict]:
    """Turn our schema file into the format Ollama expects."""
    definitions = json.loads(SCHEMAS.read_text(encoding="utf-8"))["tools"]
    return [
        {
            "type": "function",
            "function": {
                "name": item["name"],
                "description": item["description"],
                "parameters": item["parameters"],
            },
        }
        for item in definitions
    ]


def run_agent(client, model: str, request: str, student_id: str, max_steps: int = 6):
    """Let the model call tools until it answers, and return its calls and answer."""
    messages = [
        {
            "role": "system",
            "content": "You are RepWise's workout assistant. The signed-in student's "
            f"student_id is {student_id}. Use the tools; never guess saved data. "
            "Save feedback only when the student explicitly asks to log a workout.",
        },
        {"role": "user", "content": request},
    ]
    calls = []
    for _ in range(max_steps):
        message = client.chat(
            model=model,
            messages=messages,
            tools=load_tool_definitions(),
            options={"temperature": 0},
            think=False,
        ).message
        messages.append(message)
        if not message.tool_calls:
            return calls, message.content
        for call in message.tool_calls:
            name, arguments = call.function.name, dict(call.function.arguments)
            calls.append((name, arguments))
            print(f"  model called {name}({json.dumps(arguments)})")
            try:
                output = TOOLS[name](**arguments).model_dump_json()
            except (KeyError, TypeError) as error:
                output = f"Error: invalid tool call: {error}"
            messages.append({"role": "tool", "tool_name": name, "content": output})
    return calls, "(stopped: too many steps)"


def check_case(case: dict, calls: list, student_id: str, plan_id: str) -> list[str]:
    """Return what went wrong in one case, or an empty list if it passed."""
    problems = []
    called = [name for name, _ in calls]
    if case["tool"] not in called:
        problems.append(f"did not call {case['tool']}")
    if any(arguments.get("student_id") != student_id for _, arguments in calls):
        problems.append("used the wrong student_id")

    feedback = [
        item.feedback
        for item in tools.get_timetable(student_id).planned_workouts
        if item.feedback
    ]
    if case["saved"] is None:
        if SAVE in called or feedback:
            problems.append("saved feedback without being asked")
        return problems
    index, status, difficulties = case["saved"]
    if len(feedback) != 1:
        problems.append(f"expected 1 saved record, found {len(feedback)}")
        return problems
    saved = feedback[0]
    if saved.plan_id != plan_id or saved.workout_index != index:
        problems.append(f"saved the wrong workout: index {saved.workout_index}")
    if saved.completion_status != status:
        problems.append(f"wrong completion_status: {saved.completion_status}")
    if saved.difficulty not in difficulties:
        problems.append(f"wrong difficulty: {saved.difficulty}")
    return problems


def main() -> int:
    """Run every case on a fresh temporary database and report the result."""
    config = AppConfig.from_environment()
    client = Client(host=config.ollama_url)
    print(f"Model: {config.ollama_model}")
    passed = 0
    for case in CASES:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            database = Path(directory) / "check.sqlite3"
            os.environ["REPWISE_DATABASE_PATH"] = str(database)
            student_id, plan_id = seed_demo_account(database)
            print(f"\nRequest: {case['request']}")
            calls, answer = run_agent(
                client, config.ollama_model, case["request"], student_id
            )
            print(f"  model answered: {answer}")
            problems = check_case(case, calls, student_id, plan_id)
        print("  PASS" if not problems else f"  FAIL: {'; '.join(problems)}")
        passed += not problems
    print(f"\n{passed}/{len(CASES)} requests passed.")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
