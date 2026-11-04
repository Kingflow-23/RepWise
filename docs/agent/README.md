# Agent tools

Four tools let an agent use RepWise. Their parameters are the main API's own names,
types, and values.

| Tool | Type | Parameters |
|---|---|---|
| `get_user_profile` | read | `student_id` |
| `get_timetable` | read | `student_id` |
| `save_workout_feedback` | write | `student_id`, `plan_id`, `week_number`, `workout_index`, `completion_status`, `difficulty`, `comment` (optional) |
| `generate_workout_plan` | write | `focus`, `workouts_per_week`, `extra_constraints`, `refinement_instructions` (only `focus` is required) |

In this folder:

- [`s2_tool_schemas.json`](s2_tool_schemas.json): the Session 2 deliverable, with the
  current definitions of the first three tools. A test checks it against the running
  server and the API's feedback route.
- [`tools.json`](tools.json): the wider project catalog, and the source for
  `generate_workout_plan`. Its entries for the other three tools are out of date, so
  use `s2_tool_schemas.json` for those.
- `Agent Sheet.xlsx`: the agent's requests and design decisions.

## The three MCP tools

`get_timetable` returns `plan_id`, `week_number`, and `workout_index` for each workout
of the latest plan, ready to pass to `save_workout_feedback`. Saving again updates the
workout's record. Unlike the API route, saving never adapts the next workout.

Each tool in [`agentic/tools.py`](../../agentic/tools.py) returns a pydantic model, so
a malformed result raises an error instead of reaching the agent.

| Tool | Success | Failure |
|---|---|---|
| `get_user_profile` | `ProfileResult` | `ToolFailure` |
| `get_timetable` | `TimetableResult` | `ToolFailure` |
| `save_workout_feedback` | `FeedbackResult` | `ToolFailure` |

Every result has an `ok` boolean. Failures also have an `error` message and a `code`:
`invalid_request`, `student_not_found`, `plan_not_found`, `workout_not_found`, or
`storage_error`. Over MCP, an invalid enum value or an over-long `comment` is rejected
earlier, as a `ToolError`.

The tools use the database at `REPWISE_DATABASE_PATH`, through the same
`StudentRepository` as the API, so feedback appears in the app. Set that variable to
another file to test without touching real data.

**These tools have no login check.** They trust the `student_id` they receive, so use
them only on your own machine. Before anyone else can reach them, the identity must
come from the authenticated backend, as it does for `generate_workout_plan`.

## See it work

| Prove that... | Run | Needs |
|---|---|---|
| The tools behave as specified | `python -m pytest tests/agentic/test_tools.py` | Nothing else |
| A client can discover and call them over MCP | `python -m agentic.mcp_demo_client` | Nothing else; uses a temporary database |
| **Your LLM picks the right tool and values** | `python -m agentic.llm_check` | Ollama; uses a temporary database |
| They work on your real account | `python -m agentic.app_proof "Your account name"` | A plan saved in the app; writes real feedback |

After `app_proof`, refresh the app and open your latest plan: the first workout shows
**Workout logged** with the comment `Saved by the agent tool`.

Start the server alone with `python server.py` or `python -m agentic.mcp_server`. It
waits for an MCP client, so it looks idle.

## `generate_workout_plan`

[`src/tools/generate_workout_plan`](../../src/tools/generate_workout_plan) exposes the
[RAG pipeline](../rag_pipeline.md) as a function, and the API's plan route already
calls it. The model only supplies `PlanOptions`. The account, repository, and pipeline
come from the authenticated backend and are never chosen by the model.

```python
from src.schemas.models import PlanOptions
from src.tools.generate_workout_plan import generate_workout_plan

options = PlanOptions.model_validate({
    "focus": "Generate a program to build leg muscles",
    "workouts_per_week": 2,
})
result = generate_workout_plan(
    options,
    student_id=authenticated_user_id,
    repository=repository,
    pipeline=pipeline,
)
```

It returns a `GenerateWorkoutPlanResult` with `status`, `message`, `plan`, `sources`,
and `plan_id`. Missing profiles and impossible schedules return `missing_profile` or
`invalid_request`. Refusals and pipeline failures keep the pipeline's status and save
nothing. Database errors are raised, so a failed save can't look like a success.
