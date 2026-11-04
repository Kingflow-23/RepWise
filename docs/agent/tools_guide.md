# Session 2: RepWise agent tools

Three tools an agent can call, as Python functions or through a FastMCP server.
Their parameters are the main API's own names, types, and values.

| Tool | Type | Parameters |
| --- | --- | --- |
| `get_user_profile` | read | `student_id` |
| `get_timetable` | read | `student_id` |
| `save_workout_feedback` | write | `student_id`, `plan_id`, `week_number`, `workout_index`, `completion_status`, `difficulty`, `comment` (optional) |

`get_timetable` returns `plan_id`, `week_number`, and `workout_index` for each
workout of the latest plan, ready to pass to `save_workout_feedback`. The full
schemas are in [`s2_tool_schemas.json`](s2_tool_schemas.json), and a test
checks that they match the running server and the API's feedback route.

## Run it

From the repository root, with dependencies from `requirements.txt` installed:

```powershell
python -m pytest tests/agentic/test_tools.py
python -m agentic.mcp_demo_client
```

The demo starts the server, discovers and calls the tools, and shows rejected
input. It uses a temporary database with one synthetic account, and needs no LLM
or API key.

Start the server alone with `python server.py` or `python -m agentic.mcp_server`.
It then waits for an MCP client, so it looks idle.

## Show that your LLM chooses the tools

With Ollama running:

```powershell
python -m agentic.llm_check
```

Your model reads five plain-language requests and the schemas in
`docs/agent/s2_tool_schemas.json`, then decides which tools to call and with which values. The
code runs those calls on a temporary database and checks the result, including that
a question doesn't save anything.

## Show it working on the real app

1. Start the API and the Streamlit app, sign in, save a profile, and generate a plan.
2. Run this with your exact sign-in name:

   ```powershell
   python -m agentic.app_proof "Your account name"
   ```

3. Refresh the app and open your latest plan. The first workout shows
   **Workout logged** with the comment `Saved by the agent tool`.

## Results

Each tool returns a pydantic model, so a malformed result raises an error instead
of reaching the agent.

| Tool | Success | Failure |
| --- | --- | --- |
| `get_user_profile` | `ProfileResult` | `ToolFailure` |
| `get_timetable` | `TimetableResult` | `ToolFailure` |
| `save_workout_feedback` | `FeedbackResult` | `ToolFailure` |

Every result has an `ok` boolean. Failures also have an `error` message and a
`code`: `invalid_request`, `student_not_found`, `plan_not_found`,
`workout_not_found`, or `storage_error`. Over MCP, an invalid enum value or an
over-long `comment` is rejected earlier, as a `ToolError`.

## Data

The tools use the database at `REPWISE_DATABASE_PATH`, through the same
`StudentRepository` as the API, so feedback is stored in the same format and shows
in the app. Set that variable to another file to test without touching real data.

## Limitations

- **No login check:** the tools trust the `student_id` they receive, so use them
  only on your own machine. Before anyone else can reach them, the identity must
  come from the authenticated backend, as it does for `generate_workout_plan`.
- **No adaptation:** saving feedback doesn't adapt the next workout, unlike the API
  route.
