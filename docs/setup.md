# Run RepWise locally

RepWise runs as two local processes: a FastAPI backend and a Streamlit UI. Plan
generation uses a local Ollama model and a prepared transcript corpus.

## What you need

- Windows PowerShell and Python 3.11 or 3.12
- The project dependencies from `requirements.txt`
- An authorized prepared corpus (not distributed with this repository)
- Ollama and a chat model, such as `llama3.2`

Semantic and hybrid retrieval also need the configured embedding model and
local index. If you want to get started without those, choose BM25 as described
below.

## 1. Install the project

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace `REPWISE_AUTH_SECRET` with a private random value. For
example:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Never commit the resulting `.env` or share its secret.

## 2. Prepare the transcript corpus

Corpus content is kept outside Git. Obtain the project’s authorized corpus
files, then follow the [corpus guide](../data/corpus/README.md) for cleaning
transcripts and preparing `manifest.csv`, `coverage_matrix.csv`, and passage
JSONL files.

Build chunks from the prepared corpus:

```powershell
python -m src.ingestion.chunker
```

Set `REPWISE_CHUNK_FILE` in `.env` to the path of any JSONL file in
`data/corpus/chunks`. The API loads every chunk file in that same folder, so the
selected filename is just the folder entry point.

## 3. Choose retrieval

**BM25 only** needs no embedding model or semantic index. Set:

```dotenv
REPWISE_RETRIEVAL=bm25
```

For **semantic** or **hybrid** retrieval, keep the embedding settings in
`.env`, download the model once, then build the index:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='intfloat/multilingual-e5-small')"
python -m src.retrieval.index --build --chunks data/corpus/chunks --index data/corpus/index/semantic.npz
```

The default strategy is hybrid. If its model or index is unavailable, the
application reports the setup error; it does not silently switch to BM25. See
[retrieval documentation](retrieval/hybrid_retrieval.md) for index details.

## 4. Start Ollama and the app

Install and start [Ollama](https://ollama.com/), then download the configured
chat model:

```powershell
ollama pull llama3.2
```

If you change `OLLAMA_MODEL` in `.env`, pull that model instead. Start the API
and UI in separate PowerShell terminals from the repository root:

```powershell
uvicorn src.api.main:app --reload
```

```powershell
python -m streamlit run .\app\streamlit_app.py
```

Open the Streamlit URL shown in the terminal. The API documentation is at
`http://127.0.0.1:8000/docs`. Register an account, save a profile, and request a
plan. The app creates its SQLite database automatically.

Set `REPWISE_API_URL` in `.env` if the UI should use an API address other than
`http://127.0.0.1:8000`.

## Test and troubleshoot

```powershell
python -m pytest -q
```

The suite uses fakes for Ollama; corpus-linkage tests require the local prepared
corpus. The API health route reports retrieval readiness but does not check
whether Ollama is responding. For plan failures, check the API terminal and
`logs/`; request IDs are returned in the `X-Request-ID` response header.

RepWise is a local course prototype, not reviewed for untrusted-network
deployment. Read the [architecture guide](architecture.md) and
[API security limitations](generation/api_and_database.md#local-demo-threat-model)
before exposing it beyond your machine.
