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

The UI is an HTTP client. It stores the bearer token in the Streamlit session and sends it to the backend for user-specific requests. It does not call the model or database directly.

## Authentication flow

1. `POST /api/v1/auth/register` accepts a username and password of at least eight characters. The backend hashes the password with Argon2 and saves only that hash.
2. `POST /api/v1/auth/login` checks the submitted password against the stored hash.
3. Both successful endpoints return a signed JWT bearer token. Tokens expire after one hour.
4. Profile and plan endpoints require `Authorization: Bearer <token>`. The backend gets the account name from the signed token and requires it to match the `student_id` path segment.
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
| `PUT` | `/api/v1/students/{student_id}/plans/{plan_id}` | Yes | Replace the account's saved plan content. |
| `DELETE` | `/api/v1/students/{student_id}/plans/{plan_id}` | Yes | Delete one of the account's saved plans. |

## Tables and ownership

- `users` stores a username, password hash, and creation time.
- `student_profiles` stores the profile JSON and update time; its username key is a foreign key to `users`.
- `training_plans` stores the complete generation request (including the profile snapshot), validated plan, selected source records, and creation time. Its student key references `student_profiles`, which ties each plan to an existing account. Account deletion cascades through profiles to plans.
- Each saved source keeps its source, chunk, passage, and video IDs; title; canonical and timestamp URLs; timestamps; a short excerpt; and the full normalized evidence text used in the prompt. Saved plans remain understandable if the corpus is later changed or removed.
- Database connections commit on success, roll back on failure, and close after every repository operation. The repository enforces a `list_plans` limit from 1 through 100 independently of API validation.

Only completed plans are inserted. A failed generation result is returned as an API error and is not saved. SQLite is local and initialized when the API starts.

Existing schemas are migrated to add the profile-to-user foreign key without dropping valid rows. If legacy profile rows have no matching user, startup stops with an explicit reconciliation error rather than discarding or assigning those rows. Plans created before full request snapshots were introduced retain their original options and plan/source JSON, but their historical profile cannot be recovered if it was not stored at creation time; new plans persist the exact profile and options used.

## Local-demo threat model

Authentication demonstrates password hashing, signed identity, and owner-scoped database operations for a local course project. It has not had a security review. It does not include email verification, password reset, rate limiting, CSRF protection for browser-cookie sessions, account recovery, or a deployment-grade secret manager. Keep it on a trusted local machine and do not present it as production security.
