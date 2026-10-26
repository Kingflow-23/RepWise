# RepWise

RepWise is a local-first workout-plan demo. It retrieves transcript passages, sends those passages and a student's request to a local Ollama model, checks the returned plan, and saves completed plans in SQLite.

The project is written as a course example: each layer has one clear job, and the data passed between layers uses shared Pydantic models. The system provides training suggestions; it does not diagnose injuries or replace a qualified coach.

## What currently works

- **Retrieval:** BM25 lexical search, local semantic search, and Reciprocal Rank Fusion (RRF) hybrid search. Sponsor and call-to-action chunks are filtered by each retriever.
- **Evidence:** stable chunk IDs, source passage IDs, titles, URLs, and timestamps are retained for citations.
- **Generation:** `PlanPipeline` retrieves evidence, builds the prompt, calls Ollama, validates the plan, and returns a readable status on failure.
- **Application:** FastAPI and Streamlit clients, SQLite persistence, Argon2 password hashes, and signed bearer tokens. Saved profile and plan endpoints require a token and only accept the signed-in account's username.
- **Evaluation:** one runner for labelled retrieval queries and unified generation/safety cases, with JSON and Markdown reports. Mocked planner cases are simulations; only `--llm ollama` exercises local generation.

The transcript passages are not distributed with this repository. You need authorized, prepared passage JSONL files to build an index and run retrieval against the corpus. The current generation checks validate structure and request consistency; they do not prove medical safety or semantic correctness.

## Setup (Windows PowerShell)

Use Python 3.11 or 3.12. Create an environment and install the app, API, ingestion, and authentication dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set a private random value for `REPWISE_AUTH_SECRET` in `.env` before starting the API. Keep `.env` local; it is ignored by Git. The SQLite database is created automatically.

### Prepare local corpus and embeddings

Place cleaned passage JSONL files in `data/corpus/passages/` and ensure `manifest.csv` and `coverage_matrix.csv` describe those videos. Then run:

```powershell
python -m src.ingestion.chunker
python -m pip install -r requirements-embeddings.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='intfloat/multilingual-e5-small')"
python -m src.retrieval.index --build --chunks data/corpus/chunks --index data/corpus/index/semantic.npz
```

The selected model is `intfloat/multilingual-e5-small`, an MIT-licensed multilingual model. Its model weights are about 471 MB; the full model repository is larger. Plan for roughly 1–2 GB of free RAM while encoding and additional disk space for the Python/PyTorch dependencies and index. The first model download needs internet; embedding and search run locally afterwards. The dependency package is separate so BM25-only use stays lightweight. See [hybrid retrieval](docs/retrieval/hybrid_retrieval.md) for the index format, RRF, and resource notes.

### Run the application

Configure the corpus and model in `.env` (the path may point to any JSONL file inside the chunks folder):

```dotenv
REPWISE_CHUNK_FILE=data/corpus/chunks/VIDEO_ID.jsonl
REPWISE_RETRIEVAL=hybrid
REPWISE_EMBEDDING_MODEL=intfloat/multilingual-e5-small
REPWISE_EMBEDDING_INDEX=data/corpus/index/semantic.npz
OLLAMA_MODEL=llama3.2
```

Install [Ollama](https://ollama.com/) and fetch the model once with `ollama pull llama3.2`. Start the backend and UI in separate terminals:

```powershell
uvicorn src.api.main:app --reload
python -m streamlit run .\app\streamlit_app.py
```

Create an account in the sidebar, then save a profile and generate a plan. API documentation is at `http://127.0.0.1:8000/docs`. The local demo uses username as the account key; it is not reviewed for deployment on an untrusted network.

## Evaluation and tests

The fast test suite uses fake services and does not need the corpus, Ollama, or embedding model:

```powershell
python -m pytest -q
```

Compare retrieval strategies on the same labelled queries (semantic and hybrid need prepared chunks and the downloaded model):

```powershell
python -m src.evaluation.runner --mode retrieval --retriever bm25
python -m src.evaluation.runner --mode retrieval --retriever semantic
python -m src.evaluation.runner --mode retrieval --retriever hybrid
```

Run the unified generation cases and run both retrieval and generation with:

```powershell
python -m src.evaluation.runner --mode generation --llm mock
python -m src.evaluation.runner --mode generation --llm ollama
python -m src.evaluation.runner --mode all --retriever hybrid --llm ollama
```

The `mock` planner is for deterministic test cases and simulated failure examples. It is not a substitute for the real pipeline and must not be described as real-model performance. Evaluation outputs go under `outputs/`; generated reports identify the retrieval strategy and generation mode. Metrics and interpretation are documented in [the evaluation protocol](docs/evaluation_protocol.md).

## How the request flows

1. A signed-in student saves a profile and sends plan options to FastAPI.
2. FastAPI loads that profile and constructs one `PlanRequest`.
3. The retrieval layer ranks corpus chunks; hybrid search adds the BM25 and semantic reciprocal-rank scores.
4. Evidence selection removes sponsor/CTA material and duplicate passage IDs, then formats bounded context.
5. The prompt builder supplies the request, evidence, and JSON schema to Ollama.
6. Pydantic parses the response; `validate_plan` checks week numbering, available days, session duration, and source ID membership.
7. Only completed plans are saved, with their original options and source links.

See [generation](docs/generation/generation.md), [API and database](docs/generation/api_and_database.md), and [hybrid retrieval](docs/retrieval/hybrid_retrieval.md) for the exact responsibilities and current limitations of each step.
