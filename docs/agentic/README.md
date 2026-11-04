# Session 2: RepWise agent tools

Three tools that any agent can call, either as Python functions or through a
FastMCP stdio server. They need no LLM or API key.

| Tool | Purpose | Writes? |
| --- | --- | --- |
| `get_user_profile(user_id)` | Read the saved `UserProfile`. | No |
| `get_timetable(user_id)` | Read weekly availability and planned workouts, each with a unique `workout_id` and any saved feedback. | No |
| `save_workout_feedback(user_id, workout_id, difficulty, notes="")` | Save feedback for a completed workout. | Yes |

## Install and run

From the repository root, with the project's virtual environment active:

```powershell
python -m pip install -r requirements.txt -r requirements-agentic.txt
python -m pytest tests/test_agentic_tools.py
python -m agentic.mcp_demo_client
python -m agentic.mcp_server
```

The demo client starts the server itself, discovers the tools, calls all three,
shows the rejections, and reloads the saved feedback from a second server
process. It writes to a temporary directory unless `REPWISE_DATA_DIR` is set.
The last command starts the server alone and waits for an MCP client on stdin,
so it prints nothing until a client connects. `python server.py` starts the same
server.

To connect from a notebook:

```python
import sys
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

transport = StdioTransport(sys.executable, ["-m", "agentic.mcp_server"])
async with Client(transport) as client:
    tools = await client.list_tools()
    profile = await client.call_tool("get_user_profile", {"user_id": "demo_user"})
```

Run this with the repository root as the working directory. `profile.data` holds
the dictionary the tool returned.

## How the tools map to the main API

The tools use the same Pydantic schemas as the API, and feedback is written by
the same `StudentRepository` code. Only these names differ:

| Tool | Main API | Notes |
| --- | --- | --- |
| `user_id` | `AccessToken.user_id`, `student_id` in routes and tables | Same value. |
| `workout_id` | `plan_id` + `week_number` + `workout_index` | The API has no workout ID, so the tools add one. |
| `difficulty` `easy` | `slightly_too_easy` | |
| `difficulty` `moderate` | `just_right` | |
| `difficulty` `hard` | `slightly_too_hard` | |
| `notes` | `comment` | Same 500-character limit. Empty notes are stored as `null`. |
| Save time | `created_at` | UTC ISO-8601. |
| Workout completed | `completion_status: "completed"` | The tool only records completed workouts. |

The API's `much_too_easy` and `much_too_hard` can't be reached from the tool's
three-value `difficulty`.

## Results

Every result has an `ok` boolean. Failures also have a stable `code` and a
readable `error`, and feedback failures include `"saved": false`.

```json
{
  "ok": true,
  "saved": true,
  "feedback": {
    "user_id": "demo_user",
    "workout_id": "leg_day",
    "id": "a generated UUID",
    "plan_id": "demo_plan",
    "week_number": 1,
    "workout_index": 0,
    "completion_status": "completed",
    "difficulty": "slightly_too_hard",
    "comment": "Last set was difficult",
    "created_at": "2026-11-04T19:00:00+00:00",
    "adapted_week_number": null,
    "adapted_workout_index": null
  }
}
```

Failure codes are `unknown_user`, `unknown_workout`, `invalid_difficulty`,
`invalid_notes`, and `storage_error`. Over MCP, the server checks arguments
against the schema first, so an invalid `difficulty` or over-long `notes` raises
a `ToolError` instead of returning one of these results.

Saving again for the same workout updates its record, as the API does.

## Data

- **Profiles and plans:** the synthetic `demo_user` in
  [`agentic/data/demo_data.json`](../../agentic/data/demo_data.json). Its
  `profile` and `plan` are validated against `UserProfile` and `TrainingPlan` on
  each read.
- **Feedback:** `agentic.sqlite3` in `REPWISE_DATA_DIR`, by default
  `data/agentic/`, which git ignores. It has the same tables as the application
  database but is a separate file, so `data/repwise.sqlite3` is never touched.
  The first save creates it and adds `demo_user`'s rows, which the foreign keys
  require. That account has no usable password. Reads never create it.

## Schemas

[`s2_tool_schemas.json`](../../s2_tool_schemas.json) holds the input schemas of
the three tools, and [`../agent/tools.json`](../agent/tools.json) is the wider
project catalog. A test checks that both match what the running server reports.

## Limitations

- The tools only know the synthetic users in the data file. They don't read real
  accounts from `data/repwise.sqlite3`.
- The server has no authentication: it trusts the `user_id` it receives, which
  is safe only for this synthetic data. Before real accounts are connected, the
  identity must come from the authenticated backend, as it does for
  `generate_workout_plan`.
- Saving feedback doesn't adapt the plan. The API route can do that, but it is
  out of scope for this tool.
