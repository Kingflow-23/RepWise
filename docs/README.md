# RepWise documentation

Pick the guide for the task you’re working on:

## Get started

- [Local setup](setup.md) — Python, corpus, embeddings, Ollama, and launch steps.
- [Configuration reference](configuration.md) — environment variables and defaults.
- [Architecture](architecture.md) — system boundaries, request flow, and design choices.
- [Developer guide and code reference](development.md) — source map, test layout, and change workflows.
- [Corpus preparation](../data/corpus/README.md) — transcript cleaning, provenance, and corpus files.
- [Knowledge-base scope](ingestion/knowledge_base_scope.md) — what belongs in the searchable corpus.

## Understand the application

- [Generation and validation](generation/generation.md) — prompts, evidence, plan checks, and limits.
- [API, accounts, and database](generation/api_and_database.md) — routes, authentication, SQLite ownership, and workout feedback.
- [Retrieval](retrieval/hybrid_retrieval.md) — BM25, semantic search, hybrid fusion, and index maintenance.
- [Session 2 agent tools](agentic/README.md) — the three local tools, MCP server, and JSON schemas.

## Evaluate changes

- [Evaluation protocol](evaluation_protocol.md) — datasets, commands, metrics, and comparison limits.
- [Metric choices](evaluation/metrics_choice.md) — what the reported scores mean.
- [2026-11-02 results analysis](evaluation/results_analysis_2026-11-02.md) — measured trade-offs and recommended application settings.
- [Red-team checks](evaluation/red_teaming.md) — safety test coverage and known gaps.

The [project README](../README.md) is the short overview. Corpus files,
embedding indexes, Ollama models, and evaluation outputs are local artifacts;
they are not bundled with the source repository.
