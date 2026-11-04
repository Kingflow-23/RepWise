# Generate workout plan tool

`src/tools/generate_workout_plan` exposes the existing RAG generation workflow
as a function for the API and a future agent dispatcher. It generates one week
using the saved profile, includes the latest saved plan for refinements, and
saves only a completed, validated plan. The existing API plan endpoint calls
this function after checking account ownership.

The model supplies `PlanOptions`: `focus`, optional `workouts_per_week`,
`extra_constraints`, and `refinement_instructions`. The JSON Schema is listed
in `docs/agent/tools.json`. Account identity, the repository and the configured
pipeline are supplied by the authenticated backend, never chosen by the model.
The dispatcher must validate tool arguments with `PlanOptions.model_validate`
and authorize the account before calling the tool.

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
result_json = result.model_dump_json()
```

The return value is a `GenerateWorkoutPlanResult` Pydantic model containing
`status`, `message`, `plan`, `sources`, and `plan_id`. A completed result has a
saved ID and a full structured plan. Missing profiles and impossible schedules
return `missing_profile` or `invalid_request`. Scope refusals and retrieval,
evidence or generation failures preserve the pipeline outcome and do not save
a plan. Database errors propagate rather than claiming that a save succeeded.

The tool uses the pipeline's configured scope gate, retrieval, model, plan
validation and correction retries. It adds no separate LLM or retrieval path.
This is the callable capability; intent routing and the agent loop are not yet
implemented.

Run the tool and endpoint checks with Python 3.11 or later:

```powershell
python -m pytest tests/tools/test_generate_workout_plan.py tests/app/test_api.py -q
```

These checks use deterministic model responses and a temporary SQLite database.
A live run still requires the configured corpus, index, models and Ollama.
