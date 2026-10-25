# RepWise

RepWise creates a structured workout plan from a student's profile, a current training request, and relevant passages from a training knowledge base. The project includes a Streamlit demo, a FastAPI backend, a local Ollama generation pipeline, and SQLite persistence.

It is a course project and a first working version. The aim is to show a clear RAG flow that can be demonstrated, explained, and improved over time.

## Project goal

The system helps a student turn a simple request such as “build a beginner abdominal routine” into a plan that fits their available days, session length, equipment, and stated goal. Each completed plan includes links to the transcript passages used as evidence.

RepWise provides training suggestions only. It does not diagnose injuries, replace a coach, or provide medical advice.

## Current features

- A small student profile: experience, goal, available days, session duration, equipment, and preferences.
- Retrieval query construction from the profile and current request.
- Evidence selection that removes sponsor and call-to-action passages.
- Structured plan generation with a local Ollama model.
- Checks for duration, available days, requested weeks, and valid source references.
- SQLite storage for profiles and completed plans.
- A FastAPI interface and Streamlit application for the demo.

## Project structure

```text
src/
  api/          FastAPI routes
  db/           SQLite repository
  generation/   RAG orchestration and plan checks
  llm/          Ollama client
  rag/          Query, evidence, and prompt preparation
  retrieval/    Retrieval interface and JSONL baseline
  schemas/      Shared request and response models
prompts/        Versioned system prompt
app/            Streamlit demo client
scripts/        CLI entry point for the generation pipeline
data/           Local database; corpus chunks are supplied separately
docs/           Architecture and design notes
examples/       Example request
tests/          Unit and API tests
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2
```

Edit `.env` if you use another local model, database location, or chunk file.

## Run with a JSONL chunk file

```powershell
python -m scripts.run_plan --request examples/request.json --chunks data/corpus/chunks/your_video.jsonl
```

The command uses the simple `JsonlRetriever` baseline. The final retrieval adapter can replace it without changing the generation pipeline.

## Run the Streamlit demo

The backend and UI run as separate processes.

Terminal 1:

```powershell
uvicorn src.api.main:app --reload
```

Terminal 2:

```powershell
python -m streamlit run .\app\streamlit_app.py
```

Run this command from the repository root. The app also resolves the project root from its own file location so local `config` and `src` imports work even when Streamlit starts with a different working directory.

Open the Streamlit URL printed by the command (usually `http://localhost:8501`). The app lets you save or update a profile, generate a plan, inspect evidence links, and revisit saved plans. It calls the FastAPI endpoints; retrieval, generation, validation, and persistence remain in the backend. Set `REPWISE_API_URL` if the API is not at `http://127.0.0.1:8000`.

The interactive API page is also available at `http://127.0.0.1:8000/docs`.

## Generation

The implemented generation flow is:

1. Load the saved profile and combine it with the current plan options.
2. Build a query from focus, goal, experience, equipment, and constraints.
3. Retrieve up to `REPWISE_TOP_K` chunks through `Retriever.search`. The current runnable adapter is a transparent JSONL keyword-overlap baseline; it is not a vector database.
4. Remove sponsor/call-to-action and duplicate evidence, cap context at 8,000 characters, and preserve passage IDs and source links.
5. Send the versioned system prompt, request, and evidence to the configured Ollama model for schema-shaped JSON output.
6. Parse with Pydantic and validate week count/numbering, available training days, session duration, and source IDs.
7. Persist only completed plans and their returned sources to SQLite.

The API reports explicit retrieval and generation errors. It does not retry or repair invalid model output. Validation checks structure and request consistency; it does not establish medical safety or scientific quality. 

## Knowledge base and ingestion

This repository consumes prepared transcript chunks; it does not implement transcript acquisition, licensing review, cleaning, chunking, embedding, or vector indexing. No corpus or source licensing manifest is currently checked in. The ingestion/indexing work is external to this codebase, and a production retriever can be connected by implementing the `Retriever` interface. See [`data/README.md`](data/README.md) for the expected local data setup and provenance requirements.

## Tests

```powershell
python -m pytest -q --basetemp .pytest-tmp
```

The tests use fake retrieval and a fake model. They do not require Ollama or a real knowledge base.

## Evaluation

The `evals/` directory currently has no evaluation harness or benchmark. The automated tests cover the pipeline with mocked retrieval/model services; they do not measure retrieval relevance, plan quality, safety, or real-model performance. Evaluation remains future work and should use an explicitly sourced/licensed dataset plus human review for training quality.
