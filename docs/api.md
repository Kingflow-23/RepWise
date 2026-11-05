# API and database

The API is the only way into RepWise's data. Streamlit keeps the token and account
ID in its session and sends them with each request, so users never type an ID. The
interactive reference is at `http://127.0.0.1:8000/docs` when the API is running.

## Accounts

1. `POST /api/v1/auth/register` takes a unique full name and a password of at least
   eight characters. It creates a UUID account ID and stores only an Argon2 hash.
2. `POST /api/v1/auth/login` checks the password against that hash.
3. Both return a signed JWT, the account ID, and the full name. Tokens last one hour.
4. Other routes get the account ID from the token and require it to match the
   `student_id` in the URL, so changing a URL grants no access. Updates and deletes
   also filter on the owner.

Without `REPWISE_AUTH_SECRET`, registration and login return a setup error. Raw
passwords and tokens are never stored.

## Routes

All paths start with `/api/v1`, and `{student}` stands for `students/{student_id}`.

| Method | Route | Login | Purpose |
|---|---|---:|---|
| `GET` | `/health` | No | Database, retriever readiness, and model settings. Doesn't contact Ollama. |
| `POST` | `/auth/register` | No | Create an account and return a token. |
| `POST` | `/auth/login` | No | Verify a password and return a token. |
| `PUT`, `GET` | `/{student}/profile` | Yes | Save or read the profile. |
| `POST` | `/{student}/plans` | Yes | Generate and save a validated plan. |
| `GET` | `/{student}/plans` | Yes | List saved plans. |
| `PUT` | `/{student}/plans/{plan_id}` | Yes | Replace a saved plan's content. |
| `DELETE` | `/{student}/plans/{plan_id}` | Yes | Delete a saved plan. |
| `POST` | `/{student}/plans/{plan_id}/weeks/{week_number}/workouts/{workout_index}/feedback` | Yes | Log feedback; may adapt the next workout. |
| `POST` | `/speech/transcribe` | Yes | Turn a recording into text. Never creates a plan. |

### Logging feedback

The route takes the plan location from its path and this body:

| Field | Values |
|---|---|
| `completion_status` | `completed`, `partially_completed`, `not_completed` |
| `difficulty` | `much_too_easy`, `slightly_too_easy`, `just_right`, `slightly_too_hard`, `much_too_hard` |
| `comment` | Optional, up to 500 characters |

`workout_index` starts at 0. The response has the saved feedback, the current plan, an
`adapted` flag, the changed workout's location if any, and a message.
[The pipeline guide](rag_pipeline.md#after-the-workout-feedback) explains when a
workout is adapted.

### Transcribing speech

Send the WAV bytes as the body with `Content-Type: audio/wav`; the response is
`{"text": "..."}`. Recordings must be complete PCM16 WAV (one or two channels,
8–48 kHz), at most two minutes and 5 MiB. Errors are 422 for invalid or empty audio,
413 for oversized, 415 for the wrong content type, and 503 if transcription is
unavailable. Audio is decoded locally and never saved.

## Data model

```mermaid
erDiagram
    users ||--o| student_profiles : has
    student_profiles ||--o{ training_plans : owns
    training_plans ||--o{ workout_feedback : receives
    training_plans ||--o{ workout_performance : records
    training_plans ||--o{ recommendations : has
    training_plans ||--o{ calendar_events : booked_as
    users ||--o{ memory_facts : remembers
    users ||--o{ memory_episodes : remembers
```

| Table | Stores |
|---|---|
| `users` | Unique full name, generated account ID, password hash, creation time. |
| `student_profiles` | Profile JSON and update time. |
| `training_plans` | The full request (including a profile snapshot), the validated plan, its sources, and creation time. |
| `workout_feedback` | One row per plan, week, and workout: completion, difficulty, comment, time, and which workout it adapted. |
| `workout_performance` | One row per plan, week, workout, and exercise: sets, reps, and weight. Written by the [agent](agent/README.md). |
| `recommendations` | Ideas kept for later review, with their reason and sources. Written by the agent. |
| `calendar_events` | One row per plan, week, and workout: the calendar event's ID and times. Written by the agent. |
| `memory_facts` | Preferences and facts the student asked the agent to remember. Written by the agent. |
| `memory_episodes` | A short record of each request that used a tool: what was asked, and what completed, was declined, or didn't finish. Written by the agent. |

- Deleting an account cascades through profiles, plans, and everything linked to a
  plan. Deleting a plan doesn't remove its events from the calendar itself.
- A successful adaptation updates the plan and its feedback link in one transaction.
- Connections commit on success, roll back on failure, and always close.
- Failed generations are returned as errors and never stored. No medical details or
  conversation transcripts are collected.

The database is created automatically, but older schemas aren't migrated. If startup
reports an unsupported schema, back up the file and point `REPWISE_DATABASE_PATH` at a
new one.

## Security limits

Authentication demonstrates password hashing, signed identity, and owner-scoped
queries for a local course project, and it hasn't had a security review. There's no
email verification, password reset, rate limiting, CSRF protection for browser-cookie
sessions, account recovery, or managed secret storage. Keep it on a trusted machine,
and don't present it as production security.
