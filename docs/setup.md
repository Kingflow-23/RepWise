# Setup and configuration

RepWise's backend is a FastAPI server; a UI (the web app in `app/`, or Streamlit in
`genai_app/`) talks to it. An optional MCP server exposes the same tools to MCP
hosts. Plan generation and the agents need Ollama and a prepared transcript
corpus, which isn't distributed with this repository.

## What you need

- Windows PowerShell and Python 3.11 or 3.12
- An authorized prepared corpus
- Ollama 0.30.9 or newer

## 1. Install the project

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Generate a signing secret, then paste it into `.env` as `REPWISE_AUTH_SECRET`.
Never commit `.env` or share this secret.

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

## 2. Prepare the corpus

Get the authorized corpus files and follow the
[corpus guide](../data/corpus/README.md) to clean the transcripts and prepare
`manifest.csv`, `coverage_matrix.csv`, and the passage files. Then build chunks:

```powershell
python -m src.ingestion.chunker
```

Set `REPWISE_CHUNK_FILE` in `.env` to any JSONL file in `data/corpus/chunks`. The
API loads every chunk file in that folder.

## 3. Choose retrieval

Hybrid is the default and needs the embedding model and a local index:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='intfloat/multilingual-e5-small')"
python -m src.retrieval.index --build --chunks data/corpus/chunks --index data/corpus/index/semantic.npz
```

The weights are about 471 MB; allow at least 1–2 GB of free RAM to encode the
corpus. Rebuild the index whenever the chunks or the model change.

To start without embeddings, set `REPWISE_RETRIEVAL=bm25` instead. RepWise never
switches retriever by itself: if the configured one can't load, the API reports
the setup error.

## 4. Start the app

Pull the models. If you change a model in `.env`, pull that one instead.

```powershell
ollama pull qwen3.5:9b   # generates plans; tool-calling model of the agents
ollama pull qwen3.5:4b   # screens requests, rewrites queries, health check, judges
ollama pull gemma4:12b   # benchmark comparisons only
```

Start the API, then a UI, in separate terminals:

```powershell
python -m uvicorn src.api.main:app --reload
```

```powershell
python -m streamlit run .\genai_app\streamlit_app.py
```

Open the Streamlit URL, register an account, save a profile, and request a plan.
The Streamlit app is also a test console for the whole backend: **Assistant**
(chat with the agents, with each request's plan, the work of each specialist, and
Approve/Decline buttons), **Approvals** (pending and past actions), **Metrics** (time, tokens, model and
tool calls per request, with CSV and JSON export), **Location &
calendar** (opt-in saved location, Google Calendar connection), and **System**
(health, the tool catalog, and an MCP host configuration for your account).
The database is created (or extended) automatically, and the API's interactive
docs, including the agent and approval routes, are at
`http://127.0.0.1:8000/docs`.

## 5. Use the agents and the MCP server

Ask the multi-agent assistant through the API
(`POST /api/v1/students/{id}/agent/requests`), or run the agent sheet's requests on
a temporary synthetic account:

```powershell
python -m scripts.run_agent            # the 5 requests of the agent sheet
python -m scripts.run_agent --multi    # multi-agent scenarios
```

The MCP server is a separate, optional process for MCP hosts. It needs a token
from `/api/v1/auth/login`; see [The MCP server](mcp.md):

```powershell
$env:REPWISE_MCP_ACCESS_TOKEN = "<access_token>"; python -m src.mcp_server.server
python -m scripts.mcp_smoke            # self-check on a temporary account
```

To use a real calendar, connect Google Calendar once per account (the OAuth client
setup is described in
[`google_calendar.py`](../src/integrations/google_calendar.py)):

```powershell
python -m scripts.connect_google_calendar connect "Your account name"
```

## Configuration

Settings are environment variables, loaded from `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `REPWISE_AUTH_SECRET` | Empty | Signs login tokens. Registration and login fail without it. |
| `REPWISE_DATABASE_PATH` | `data/repwise.sqlite3` | SQLite database for accounts, profiles, plans, and feedback. |
| `REPWISE_CHUNK_FILE` | None | Any JSONL file in the chunk folder. |
| `REPWISE_RETRIEVAL` | `hybrid` | `bm25`, `semantic`, or `hybrid`. |
| `REPWISE_TOP_K` | `5` | Evidence chunks requested. Values below 5 are raised to 5. |
| `REPWISE_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Local embedding model. |
| `REPWISE_EMBEDDING_INDEX` | `data/corpus/index/semantic.npz` | Semantic index file. |
| `REPWISE_QUERY_EXPANSION` | `true` | French query rewriting (`true`, `false`, `1`, `0`, `yes`, `no`). Turning it off speeds up retrieval but keeps request screening. |
| `REPWISE_SPEECH_MODEL` | `base` | Local faster-whisper model, downloaded on first use. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama address. |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Generates plans. |
| `OLLAMA_QUERY_EXPANSION_MODEL` | `qwen3.5:4b` | Screens requests and rewrites queries. |
| `OLLAMA_COMPARISON_MODEL` | `gemma4:12b` | Second model in benchmarks. |
| `OLLAMA_JUDGE_MODEL` | `qwen3.5:4b` | Scores benchmark answers. |
| `REPWISE_API_URL` | `http://127.0.0.1:8000` | API address used by Streamlit. |
| `REPWISE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `REPWISE_LOG_DIR` | `logs` | Folder for rotating log files. |
| `REPWISE_GOOGLE_DIR` | `data/google` | Google OAuth client file and per-account calendar tokens. |
| `REPWISE_AGENT_MODEL` | `OLLAMA_MODEL` | Tool-calling model of the orchestrator and specialists. |
| `REPWISE_HEALTH_CHECK_MODEL` | `OLLAMA_QUERY_EXPANSION_MODEL` | Decides whether text mentions pain, injury or a medical issue. Fails closed. |
| `REPWISE_TIMEZONE` | `Europe/Paris` | Default time zone for dates and calendar slots. |
| `REPWISE_EPISODE_RETENTION_DAYS` | `90` | How long episodic memory is kept. |
| `REPWISE_APPROVAL_MINUTES` | `30` | How long a pending action can be approved. |
| `REPWISE_AGENT_MAX_DELEGATIONS` | `4` | Specialist steps per request. |
| `REPWISE_AGENT_MAX_REPLANS` | `1` | Plan revisions per request. |
| `REPWISE_AGENT_MAX_TOOL_CALLS` | `5` | Tool calls per specialist task. |
| `REPWISE_AGENT_REQUEST_SECONDS` | `420` | Time budget of one request. |
| `REPWISE_AGENT_SPECIALIST_SECONDS` | `240` | Time budget of one specialist task (also the model call timeout). |
| `REPWISE_AGENT_CONTEXT_TOKENS` | `16384` | Ollama context window for agent calls. |
| `REPWISE_PHOTON_URL` | `https://photon.komoot.io` | Geocoding and nearby-place service (OpenStreetMap data, no key). |
| `REPWISE_HTTP_USER_AGENT` | `RepWise-course-project` | User-Agent sent to public map services. |
| `REPWISE_MCP_ACCESS_TOKEN` | None | Token the MCP server binds at startup. Set only in the MCP host's private configuration. |

An invalid value is reported as an error rather than replaced. The defaults follow
[our measurements](evaluation.md#what-we-found) where those exist.

## Test and troubleshoot

```powershell
python -m pytest -q
```

The tests replace Ollama, Photon and Google with fakes, but some corpus-linkage
tests need the local corpus. Tests against the real services are opt-in:

```powershell
$env:REPWISE_LIVE_TESTS = "1"; python -m pytest tests\integration -q
```

 If a plan fails, check the API terminal and `logs/`; each response carries
a request ID in its `X-Request-ID` header. The health route reports whether
retrieval is ready, but it doesn't check whether Ollama is responding.

RepWise is a local course prototype. Read the
[security limits](api.md#security-limits) before exposing it beyond your machine.
