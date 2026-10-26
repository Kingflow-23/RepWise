# Evaluation protocol

RepWise has one evaluation command and two datasets: retrieval questions with time-span relevance labels, and generation cases with a request, optional user profile, expected status, and targeted text constraints. The generation cases combine the former program scenarios and applicable safety cases. Cases describing an adaptation engine that does not exist are excluded rather than reported as if executed.

## Run the benchmark

```powershell
python -m pytest tests/evaluation
python -m src.evaluation.runner --mode retrieval --retriever bm25
python -m src.evaluation.runner --mode retrieval --retriever semantic
python -m src.evaluation.runner --mode retrieval --retriever hybrid
python -m src.evaluation.runner --mode generation --llm mock
python -m src.evaluation.runner --mode generation --llm ollama
python -m src.evaluation.runner --mode all --retriever hybrid --llm ollama
```

Retrieval mode never calls an LLM. It evaluates the same labelled retrieval queries for BM25, semantic, or hybrid search. Generation mode exercises the application pipeline using the generation cases; `mock` is a deterministic simulation, while `ollama` uses the configured local model. All mode runs both suites and uses the configured retriever for the generation pipeline. Runs write `report.md` and `results.json` under `outputs/eval_runs/`.

The retrieval corpus is local and is not committed. Without it, retrieval cannot run. Semantic and hybrid modes also require the local embedding model and a current semantic index. Generation with Ollama requires Ollama and the configured model. The deterministic mock is useful for runner checks, but its scores do not represent model quality.

## Datasets

- `evals/retrieval/retrieval_test_cases.jsonl` contains query text, answerability, and a topic label. No split field is used.
- `evals/expected_evidence/relevance_judgments.jsonl` links answerable query IDs to relevant video time spans.
- `evals/benchmark_cases.jsonl` contains executable generation cases. Profiles are stored inline only when required by the request; missing profiles are valid inputs for clarification cases.

Add cases as one JSON object per line. Retrieval cases marked answerable must have at least one relevance judgment; unanswerable cases must have none. Generation cases must include a unique ID, request, and one or more expected statuses. Optional regex checks are applied to normalized user-visible output.

## Metrics

Retrieval reports Recall@k, Precision@k, Hit@1, MRR, and latency. A retrieved chunk is relevant when it overlaps a labelled time span in the same video. Unanswerable requests are excluded from ranking metrics because they have no relevant span; generation cases cover abstention and out-of-scope behavior.

Generation reports expected-status pass rate, abstention accuracy and false-abstention rate, output schema validity, citation ID validity, optional quote-level citation precision and faithfulness when the pipeline exposes claims, and latency. Citation ID validity only confirms that a source ID was supplied to the model; it is not evidence of semantic support. Quote checks are only run when claim-level quotes are available, so these metrics may be N/A for the current plan generator.

Reports include whether generation was mocked or real and the selected retrieval, embedding, and generation models. A high score on the small encoded case set does not establish general safety or semantic correctness; review failures and sample outputs manually.
