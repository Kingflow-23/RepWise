# Developer guide and code reference

This guide is the entry point for changing RepWise. It maps the implementation,
connects common changes to their tests, and points to the detailed behavior
guides. The [architecture guide](architecture.md) describes the runtime flow;
this page is a source and development reference.

## Repository layout

| Path | Contents |
|---|---|
| [`app/`](../app) | Streamlit user interface; communicates with the API over HTTP. |
| [`config/`](../config) | Runtime and evaluation settings, plus body-part, topic, and transcript-correction vocabularies. |
| [`data/corpus/`](../data/corpus/README.md) | Corpus preparation guide and tracked metadata/placeholders. Authorized source transcripts, prepared passages, and generated chunks are local data. |
| [`docs/`](README.md) | Setup, architecture, configuration, generation, retrieval, ingestion, and evaluation guides. |
| [`evals/`](../evals) | Fixed benchmark, retrieval, expected-evidence, red-team, and smoke-test inputs. |
| [`outputs/`](../outputs) | Local evaluation reports; generated results should not be committed. |
| [`prompts/`](../prompts) | Versioned model prompts. |
| [`src/`](../src) | Application, ingestion, retrieval, generation, and evaluation implementation. |
| [`src/demo/`](../src/demo) | Supporting demo material, including an exploratory notebook and image. |
| [`tests/`](../tests) | Unit, integration, and corpus-linkage tests organized by component. |

The local SQLite database, transcript corpus, embedding index, Ollama model
weights, logs, and evaluation run outputs are runtime artifacts. They are not
reproducible from source alone; see [local setup](setup.md) and
[corpus preparation](../data/corpus/README.md).

## Python source map

The docstring on each class and function is the detailed reference for its
arguments, return values, and failure behavior. The tables below explain where
to find the implementation; they intentionally do not duplicate every symbol's
contract.

### Application, API, and persistence

| Module | Responsibility |
|---|---|
| [`app/streamlit_app.py`](../app/streamlit_app.py) | Streamlit views, authenticated API client, plan/source rendering, and workout feedback forms. |
| [`config/app_config.py`](../config/app_config.py) | UI-side API URL configuration. |
| [`src/api/main.py`](../src/api/main.py) | FastAPI app construction, authentication dependencies, health check, and account-scoped routes. |
| [`src/api/auth.py`](../src/api/auth.py) | Password hashing and signed access-token helpers. |
| [`src/db/repository.py`](../src/db/repository.py) | SQLite schema, ownership-scoped account/profile/plan operations, feedback persistence, and plan adaptation updates. |
| [`agentic/tools.py`](../agentic/tools.py) | Session 2 profile/timetable reads and isolated synthetic feedback saved through `StudentRepository`. |
| [`agentic/mcp_server.py`](../agentic/mcp_server.py) | FastMCP stdio server exposing the Session 2 tools. |
| [`src/schemas/models.py`](../src/schemas/models.py) | Pydantic contracts shared by the API, generation pipeline, retrieval, and persistence. |
| [`src/observability.py`](../src/observability.py) | Shared logging setup, rotating per-process logs, and request identifiers. |

Package initializer modules expose package-level APIs and descriptions:
[`src/__init__.py`](../src/__init__.py),
[`src/api/__init__.py`](../src/api/__init__.py),
[`src/db/__init__.py`](../src/db/__init__.py),
[`src/evaluation/__init__.py`](../src/evaluation/__init__.py),
[`src/evaluation/metrics/__init__.py`](../src/evaluation/metrics/__init__.py),
[`src/generation/__init__.py`](../src/generation/__init__.py),
[`src/llm/__init__.py`](../src/llm/__init__.py),
[`src/rag/__init__.py`](../src/rag/__init__.py),
[`src/retrieval/__init__.py`](../src/retrieval/__init__.py), and
[`src/schemas/__init__.py`](../src/schemas/__init__.py).

### Generation, prompts, and model integration

| Module | Responsibility |
|---|---|
| [`src/generation/pipeline.py`](../src/generation/pipeline.py) | Retrieval-augmented plan generation, evidence selection, retries, and validation orchestration. |
| [`src/generation/adaptation.py`](../src/generation/adaptation.py) | Targeted adjustment of one later unlogged workout while preserving its exercise identities and evidence. |
| [`src/rag/context.py`](../src/rag/context.py) | Search-query construction and retrieved-evidence preparation. |
| [`src/rag/prompt.py`](../src/rag/prompt.py) | Prompt assembly, previous-plan summary bounds, and structured-output schema handling. |
| [`src/llm/ollama.py`](../src/llm/ollama.py) | Local Ollama generation boundary. |

See [generation and validation](generation/generation.md) for request behavior,
validation limits, and adaptation details.

### Retrieval and corpus ingestion

| Module | Responsibility |
|---|---|
| [`src/retrieval/base.py`](../src/retrieval/base.py) | Retriever protocol, shared defaults, and explicit unavailable-retriever behavior. |
| [`src/retrieval/sparse.py`](../src/retrieval/sparse.py) | Local BM25 search over chunk text and searchable metadata. |
| [`src/retrieval/semantic.py`](../src/retrieval/semantic.py) | Local semantic search over an embedding matrix. |
| [`src/retrieval/hybrid.py`](../src/retrieval/hybrid.py) | Reciprocal Rank Fusion of lexical and semantic rankings. |
| [`src/retrieval/query_expansion.py`](../src/retrieval/query_expansion.py) | Request-scope screening, French query rewriting, and one-pass retrieval delegation. |
| [`src/retrieval/index.py`](../src/retrieval/index.py) | CLI for building and validating the local semantic index. |
| [`src/body_parts.py`](../src/body_parts.py) | Shared body-region matching for chunk tags and retrieval-query terms. |
| [`src/ingestion/build_corpus.py`](../src/ingestion/build_corpus.py) | Cleans transcript fragments and creates timestamped passages. |
| [`src/ingestion/chunker.py`](../src/ingestion/chunker.py) | Filters and groups passages into retrieval-ready chunks. |
| [`src/ingestion/coverage_report.py`](../src/ingestion/coverage_report.py) | Reports topic coverage across the prepared corpus. |

See [retrieval](retrieval/hybrid_retrieval.md) and
[knowledge-base scope](ingestion/knowledge_base_scope.md) before changing
retrieval inputs or corpus policy.

### Evaluation

| Module | Responsibility |
|---|---|
| [`config/config.py`](../config/config.py) | Environment-backed runtime settings for the API and generation pipeline. |
| [`config/evaluation_config.py`](../config/evaluation_config.py) | Evaluation-runner settings and paths. |
| [`src/evaluation/datasets.py`](../src/evaluation/datasets.py) | Loads and validates benchmark cases and evidence links. |
| [`src/evaluation/adapters.py`](../src/evaluation/adapters.py) | Adapts production retrievers and planners to evaluation contracts. |
| [`src/evaluation/judge.py`](../src/evaluation/judge.py) | Validates and scores raw benchmark answers with a local LLM judge. |
| [`src/evaluation/runner.py`](../src/evaluation/runner.py) | CLI orchestration for retrieval, generation, comparison, smoke, and red-team modes. |
| [`src/evaluation/report.py`](../src/evaluation/report.py) | Markdown and console evaluation reports. |
| [`src/evaluation/metrics/common.py`](../src/evaluation/metrics/common.py) | Shared ratio and aggregation types. |
| [`src/evaluation/metrics/retrieval.py`](../src/evaluation/metrics/retrieval.py) | Retrieval ranking metrics. |
| [`src/evaluation/metrics/grounding.py`](../src/evaluation/metrics/grounding.py) | Citation, evidence-support, and abstention metrics. |

The [`evals/`](../evals) directory contains the benchmark inputs rather than
evaluation implementation. Read the [evaluation protocol](evaluation_protocol.md)
before interpreting results.

## Tests and change workflow

Run the full test suite from the repository root:

```powershell
python -m pytest -q
```

For a focused change, run the matching component tests:

| Change | Suggested tests |
|---|---|
| Streamlit client or API behavior | `python -m pytest tests\app -q` |
| SQLite schema or repository behavior | `python -m pytest tests\db -q` |
| Plan generation or validation | `python -m pytest tests\generation -q` |
| Ingestion and chunk creation | `python -m pytest tests\ingestion -q` |
| Retriever or semantic-index behavior | `python -m pytest tests\retrieval -q` |
| Evaluation datasets, metrics, or runner | `python -m pytest tests\evaluation -q` |
| Logging and request IDs | `python -m pytest tests\test_observability.py -q` |

Some corpus-linkage tests need the prepared local corpus. The evaluation runner
also uses local Ollama models for real-model modes; deterministic mock tests do
not replace model-performance evaluation. Refer to the relevant guide before
running expensive or corpus-dependent checks.

When changing behavior, update the matching guide as well as the implementation.
Keep docstrings with their classes and functions: describe intent and observable
contracts, and include `Args`, `Returns`, and `Raises` sections where they help
callers understand non-obvious behavior. Private helpers and nested functions
are documented too when their behavior is not self-evident.

## Common development tasks

### Change corpus preprocessing

Update the ingestion code and any affected lexicon or correction file, then run
the ingestion tests. Rebuild chunks and the semantic index for local evaluation;
re-check dataset-to-passage linkage if chunk boundaries or passage text change.
See the [corpus guide](../data/corpus/README.md).

### Change retrieval

Preserve the common retriever contract and explicit setup errors. Run the
retrieval tests, rebuild the index when corpus text, tags, or embedding model
changes, and compare strategies on the same labelled cases. See
[retrieval details](retrieval/hybrid_retrieval.md).

### Change generation or API persistence

Update the request/response model, validation, route, or repository layer that
owns the behavior. Run generation and API/database tests as applicable. A plan
is persisted only after successful validation; preserve account ownership
checks when changing persistence paths. See
[generation](generation/generation.md) and
[API/database behavior](generation/api_and_database.md).

### Change evaluation

Keep dataset validation, metric semantics, and report wording aligned. Run the
evaluation tests, then use a bounded smoke run before a full comparison when
local models and corpus artifacts are available. See the
[evaluation protocol](evaluation_protocol.md).
