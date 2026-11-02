# RepWise

**From training transcripts to a workout plan you can inspect.**

RepWise is a local-first learning project: it searches prepared fitness
transcripts, asks a local Ollama model to draft a plan, checks the plan against
your schedule and equipment, and shows the sources behind it.

It is a prototype for general fitness planning, not a medical or coaching
service. It does not diagnose injuries or guarantee that a plan is safe.

Saved workouts can also be logged with completion and difficulty feedback.
RepWise stores that feedback with the specific workout and can use it to adjust
the next unlogged workout in the same plan, without regenerating the rest of the
plan. Feedback mentioning pain or discomfort is saved without automatically
changing the plan.

## How it fits together

```mermaid
flowchart LR
    Student --> UI[Streamlit app]
    UI -->|HTTP + bearer token| API[FastAPI]
    API --> Auth[Authentication]
    API --> DB[(SQLite profiles and plans)]
    API --> Plan[Plan pipeline]
    Plan --> Query[Build search query]
    Query --> Search{Configured retriever}
    Search --> BM25[BM25]
    Search --> Semantic[Semantic search]
    Search --> Hybrid[Hybrid: BM25 + semantic RRF]
    Corpus[(Prepared transcript chunks)] --> Search
    BM25 --> Evidence[Select sources]
    Semantic --> Evidence
    Hybrid --> Evidence
    Evidence --> Prompt[Prompt + JSON schema]
    Prompt --> Ollama[Local Ollama model]
    Ollama --> Validate[Validate plan and citations]
    Validate --> DB
    Validate --> UI

    Raw[Authorized source transcripts] --> Ingest[Ingestion and filtering]
    Ingest --> Corpus
    Corpus --> Index[(Local semantic index)]
    Index --> Semantic

    Cases[Evaluation cases] --> Eval[Evaluation runner]
    Eval --> Search
    Eval --> Reports[Evaluation reports]
```

## Try it locally

You’ll need Python 3.11 or 3.12, an authorized prepared transcript corpus, and
[Ollama](https://ollama.com/). Start with the [setup guide](docs/setup.md);
corpus files and model weights are local and are not distributed with this
repository.

## Microphone input

On the plan-generation tab, sign in and record your request using **Record your request**. Stop recording, click **Use recording**, then review or edit the text in the usual focus field. Click **Generate plan** only when the text is ready; recording and transcription never submit a plan automatically. A failed transcription preserves the current draft.

Microphone access requires browser permission and localhost or an HTTPS connection. Record up to two minutes. The backend transcribes locally with faster-whisper, preserving the spoken language. The first transcription downloads the multilingual `base` model if it is not cached; later requests reuse it. Set `REPWISE_SPEECH_MODEL` to change the model. Install the updated requirements and restart the API/UI before using it. Recordings are handled in memory and are not written to the database or audio files.

## Run the tests

```powershell
python -m pytest -q
```

Some corpus-linkage tests require the prepared local corpus. For evaluation
commands and what the results do and do not show, see the
[evaluation guide](docs/evaluation_protocol.md).

## Explore the docs

Start at the [documentation hub](docs/README.md) for setup, architecture,
generation, retrieval, corpus preparation, API/database details, and
evaluation. Developers can use the [code reference and development guide](docs/development.md)
to find implementation modules, relevant tests, and change workflows.

## Important limits

RepWise checks plan structure, schedule constraints, equipment declarations,
and that cited source IDs were retrieved. It does not require exact transcript
quotes, which lets the model use non-English evidence while producing a
coherent English plan. These checks cannot prove that a source supports an
exercise, that advice is medically appropriate, or that the model followed
every instruction.
