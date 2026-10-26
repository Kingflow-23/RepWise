# API and database

## Run the API

```powershell
uvicorn src.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the routes in the browser. Copy `.env.example` to `.env`, then configure the local database with `REPWISE_DATABASE_PATH`. Set `REPWISE_CHUNK_FILE` to use the simple JSONL retriever, or connect the final retriever when the index is ready.

## Streamlit demo

Install dependencies with `python -m pip install -r requirements.txt`, then run `python -m streamlit run .\\app\\streamlit_app.py` from the repository root in another terminal. Configure `REPWISE_API_URL` if the API is not running at `http://127.0.0.1:8000`. The app saves and loads profiles, submits plan options, displays returned plan/source data, and lists saved plans. All generation and persistence calls go through the API. The entry point resolves the repository root before importing local packages, so `config` and `src` imports do not depend on Streamlit's script import path.

## API routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Shows database readiness, whether a retriever is configured, and the configured model name. It does not probe the Ollama server. |
| `PUT` | `/api/v1/students/{student_id}/profile` | Saves the small profile used to personalise plans. |
| `GET` | `/api/v1/students/{student_id}/profile` | Returns the saved profile. |
| `POST` | `/api/v1/students/{student_id}/plans` | Generates and stores one plan. |
| `GET` | `/api/v1/students/{student_id}/plans` | Lists saved plans for that student. |

This is a course-demo API, not a public service. `student_id` is a simple identifier in the URL. Add real authentication before exposing the app outside the project.

## SQLite tables

`student_profiles` has one row per student. It stores the profile as JSON and an update timestamp.

`training_plans` stores a completed structured plan, its original plan options, its selected sources and a creation timestamp. It links to `student_profiles` with a foreign key. A plan is saved only after generation and validation succeed.

This is intentionally small. It is enough to show persistence and personalisation in the demo, and it can be replaced by a fuller application database later.
