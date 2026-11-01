# Configuration reference

RepWise reads backend settings from environment variables, loading a local
`.env` file when the API or runner starts. Copy `.env.example` to `.env` as
described in the [setup guide](setup.md), then change only the values needed
for your machine. Never commit `.env` or share `REPWISE_AUTH_SECRET`.

## Backend and model settings

| Variable | Default | Purpose |
|---|---|---|
| `REPWISE_DATABASE_PATH` | `data/repwise.sqlite3` | Local SQLite database for accounts, profiles, plans, and workout feedback. |
| `REPWISE_AUTH_SECRET` | Empty | Secret used to sign login tokens. Set a private random value before using registration or login. |
| `REPWISE_CHUNK_FILE` | No override | Path to a JSONL chunk file. The API loads all JSONL files in that file's parent folder. |
| `REPWISE_RETRIEVAL` | `hybrid` | Search strategy: `bm25`, `semantic`, or `hybrid`. Semantic and hybrid require a working local embedding index. |
| `REPWISE_TOP_K` | `5` | Number of chunks requested as evidence. Values below five are raised to five. |
| `REPWISE_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Local Sentence Transformers model used for semantic retrieval. |
| `REPWISE_EMBEDDING_INDEX` | `data/corpus/index/semantic.npz` | Semantic index file; rebuild it when the chunk corpus or embedding model changes. |
| `REPWISE_QUERY_EXPANSION` | `true` | Enables French query translation/expansion. Setting it to `false` disables rewriting, but does not disable request-scope screening. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for the local Ollama service. |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Ollama model used to generate workout plans. |
| `OLLAMA_QUERY_EXPANSION_MODEL` | `qwen3.5:4b` | Ollama model used for request-scope screening and, when enabled, query rewriting. |

`REPWISE_TOP_K` must be an integer. `REPWISE_QUERY_EXPANSION` accepts
`true`, `false`, `1`, `0`, `yes`, or `no` (case-insensitive). An unsupported
retrieval strategy or invalid setting is reported as a configuration error; it
does not silently select a different retriever.

## Evaluation settings

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_COMPARISON_MODEL` | `gemma4:12b` | Second generation model used by controlled evaluation comparisons. |
| `OLLAMA_JUDGE_MODEL` | `qwen3.5:4b` | Model used to score benchmark answers and compare candidates. |

The comparison and judge models are used by evaluation workflows, not normal
plan generation. Pull the selected tags into Ollama before running a benchmark.
See the [evaluation protocol](evaluation_protocol.md) for workflow commands and
limitations.

## Streamlit and logging

| Variable | Default | Purpose |
|---|---|---|
| `REPWISE_API_URL` | `http://127.0.0.1:8000` | FastAPI base URL used by the Streamlit client. Trailing slashes are removed. |
| `REPWISE_LOG_LEVEL` | `INFO` | Console and file logging threshold: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `REPWISE_LOG_DIR` | `logs` | Directory for rotating per-service log files. |

The API and Streamlit run as separate processes. Configure their URLs
independently when they do not use the local defaults. Logs include request IDs
where available; never add passwords, bearer tokens, or the signing secret to
logs or diagnostic reports.
