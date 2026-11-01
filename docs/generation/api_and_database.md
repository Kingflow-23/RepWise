# API, accounts, and SQLite data

## Start the local application

Install `requirements.txt`, copy `.env.example` to `.env`, and replace the example `REPWISE_AUTH_SECRET` with a private random value. Configure the local corpus and index as described in [hybrid retrieval](../retrieval/hybrid_retrieval.md). Start the backend:

```powershell
uvicorn src.api.main:app --reload
```

The interactive API page is at `http://127.0.0.1:8000/docs`. In another terminal, start the UI:

```powershell
python -m streamlit run .\app\streamlit_app.py
```

The UI is an HTTP client. It stores the bearer token and internal account ID in the Streamlit session and sends them to the backend for user-specific requests. It does not call the model or database directly, and the user does not need to enter an account ID.

## Authentication flow

1. `POST /api/v1/auth/register` accepts a full name and password of at least eight characters. Full names must be unique. The backend generates a UUID account ID and stores only an Argon2 password hash.
2. `POST /api/v1/auth/login` looks up the full name and checks the submitted password against the stored hash.
3. Both successful endpoints return a signed JWT bearer token, the internal account ID, and the full name. Tokens expire after one hour.
4. Profile and plan endpoints require a bearer token. The backend gets the internal account ID from the signed token and requires it to match the `student_id` path segment.
5. The database queries include both the plan ID and owner when updating or deleting. A client cannot gain access by changing a URL ID.

The secret used to sign tokens comes from `REPWISE_AUTH_SECRET`. Without it, registration and login return a setup error. Never commit `.env` or send the secret to the UI.

## Routes

| Method | Route | Login required | Purpose |
|---|---|---:|---|
| `GET` | `/api/v1/health` | No | Reports database, retriever readiness, and model configuration; does not contact Ollama. |
| `POST` | `/api/v1/auth/register` | No | Create an account and return a token. |
| `POST` | `/api/v1/auth/login` | No | Verify a password and return a token. |
| `PUT` | `/api/v1/students/{student_id}/profile` | Yes | Create or update the signed-in account's profile. |
| `GET` | `/api/v1/students/{student_id}/profile` | Yes | Read the signed-in account's profile. |
| `POST` | `/api/v1/students/{student_id}/plans` | Yes | Generate and save a validated plan. |
| `GET` | `/api/v1/students/{student_id}/plans` | Yes | List the account's saved plans. |
| `POST` | `/api/v1/students/{student_id}/plans/{plan_id}/weeks/{week_number}/workouts/{workout_index}/feedback` | Yes | Log feedback for one saved workout; may adapt the next unlogged workout in that plan. |
| `PUT` | `/api/v1/students/{student_id}/plans/{plan_id}` | Yes | Replace the account's saved plan content. |
| `DELETE` | `/api/v1/students/{student_id}/plans/{plan_id}` | Yes | Delete one of the account's saved plans. |

## Workout feedback and targeted adaptation

The feedback route accepts `completion_status` (`completed`,
`partially_completed`, or `not_completed`), `difficulty` (`much_too_easy`,
`slightly_too_easy`, `just_right`, `slightly_too_hard`, or `much_too_hard`),
and an optional comment of up to 500 characters. The week number and
zero-based workout index identify the workout in the saved plan. The route
requires the plan to belong to the authenticated account.

Feedback is upserted, so submitting again updates that workout's feedback.
When the configured workout generator is available, the API can adjust only
the next unlogged workout in calendar order in that same plan. Adaptation keeps
exercise names, order, equipment, and source evidence; it can change sets,
repetitions, rest, time estimates, and coaching notes. The adjusted plan is
validated against the original request before it is saved. The response
includes the saved feedback, current plan, an `adapted` flag, the changed
workout location when applicable, and a user-facing message.

Once a feedback record has been linked to an adapted workout, resubmitting it
updates the feedback but does not trigger another adaptation. If the comment
matches the current English pain/discomfort keyword check, the feedback is
saved but the plan is not changed. Feedback is still saved if no later
unlogged workout exists, adaptation is unavailable, or adaptation fails; the
existing plan remains unchanged in those cases. This is a narrow, single-plan
adjustment flow, not a medical assessment or multi-week progression system.

## Tables and ownership

- `users` stores the unique full name, generated account ID, password hash, and creation time.
- `student_profiles` stores the profile JSON and update time; its account ID is a foreign key to `users`.
- `training_plans` stores the complete generation request (including the profile snapshot), validated plan, selected source records, and creation time. Its student key references `student_profiles`, which ties each plan to an existing account. Account deletion cascades through profiles to plans.
- `workout_feedback` stores the latest completion status, difficulty, optional comment, and creation time for each plan/week/workout position. It also records the week/workout location adapted from that feedback; duplicate submissions update the existing row.
- Each saved source keeps its source, chunk, passage, and video IDs; title; canonical and timestamp URLs; timestamps; a short excerpt; and the full normalized evidence text used in the prompt. Saved plans remain understandable if the corpus is later changed or removed.
- Database connections commit on success, roll back on failure, and close after every repository operation. The repository enforces a `list_plans` limit from 1 through 100 independently of API validation.

Only completed plans are inserted. A failed generation result is returned as an API error and is not saved. SQLite is local and initialized when the API starts.

The application creates the SQLite database automatically, but does not migrate older schemas. If you have a database from before account IDs were introduced, startup reports that its schema is unsupported. Keep a backup if you need its data, then set `REPWISE_DATABASE_PATH` to a new database file to start fresh.

## Local-demo threat model

Authentication demonstrates password hashing, signed identity, and owner-scoped database operations for a local course project. It has not had a security review. It does not include email verification, password reset, rate limiting, CSRF protection for browser-cookie sessions, account recovery, or a deployment-grade secret manager. Keep it on a trusted local machine and do not present it as production security.
