# Evaluation protocol

RepWise evaluation has three distinct datasets and workflows:

1. **Retrieval benchmark:** labelled questions and evidence time spans compare BM25, semantic, and hybrid retrieval.
2. **Generation benchmark:** compare Qwen and Gemma with RAG using an LLM judge, then compare the winning model with and without RAG.
3. **Red-team checks:** a separate set distinguishes requests that must be refused from in-scope prompt/profile conflicts that should still get a safe answer.

## Prepare models and corpus

The corpus is local and is not committed. Prepare it as described in the
[corpus guide](../data/corpus/README.md), including chunk JSONL files and (for
semantic or hybrid retrieval) a semantic index. Install Ollama, then pull the
two comparison chat models and the local query-analysis / answer-judge model. The Ollama
tags below select the instruct variants even though `-instruct` is not part of
the tag:

```powershell
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
ollama pull gemma4:12b
```

Set `OLLAMA_MODEL` and `OLLAMA_COMPARISON_MODEL` for the two generation models.
`OLLAMA_QUERY_EXPANSION_MODEL` and `OLLAMA_JUDGE_MODEL` default to
`qwen3.5:4b`; they can be overridden independently. All configured models must
be available from Ollama. Semantic and hybrid retrieval also need the configured
embedding model and current local index.

## Run the evaluation

```powershell
python -m pytest tests\evaluation tests\generation tests\retrieval -q

# Rank retrieval strategies; judge Qwen vs Gemma with RAG; compare the winner
# against its no-RAG baseline; then run the scope/red-team set on the winner.
python -m src.evaluation.runner --mode compare

# Optionally pin the retrieval strategy used by the RAG generation arms.
python -m src.evaluation.runner --mode compare --retriever bm25 --model-a qwen3.5:9b --model-b gemma4:12b

# Run the scope/red-team set alone with raw generation and LLM judging.
python -m src.evaluation.runner --mode red-team --llm ollama --retriever bm25 --model-a qwen3.5:9b

# Fast staged smoke: all three real retrievers on curated queries, then one
# real model on one nominal raw-answer case and four red-team pipeline cases.
python -m src.evaluation.runner --mode smoke --llm ollama --retriever hybrid --model-a qwen3.5:4b

# Run individual diagnostic suites.
python -m src.evaluation.runner --mode retrieval --retriever bm25
python -m src.evaluation.runner --mode generation --llm ollama --retriever hybrid
```

`compare` first ranks the production BM25, semantic, and hybrid retrievers on
the same retrieval questions. Evaluation ranks **chunks**, not transcript
passages. Retrieval ranking uses chunk Recall@10 (`MAIN_K`), then MRR and
median latency. Generation uses the configured strategy (hybrid by default),
unless `--retriever` overrides it. Query expansion is enabled for retriever
ranking and both RAG generation arms. For curated retrieval ranking, a scope
model refusal does not block the search; the original benchmark query is used
when the model declines to rewrite it. The red-team arm uses the scope
analyzer's French search query with an unwrapped retriever, avoiding duplicate
rewriting.
The nominal model-comparison arms generate raw answers for both models with
RAG; production plan validation, deterministic answer-format checks, and retries
are not applied. The configured
judge scores request fit, schedule, profile constraints, safety, clarity, JSON
quality, plan-schema fit, and RAG evidence support. It also compares both raw
answers side by side for every case, records scores for each candidate, and
explains its preferred answer. The highest common answer-quality score selects
the model for the no-RAG arm on the same cases. Raw answers, per-dimension
scores, pairwise choices, rationales, and errors are retained in `results.json`.

The comparison automatically runs the 15-case red-team set against the winning
RAG setup. Red-team scoring is intentionally binary: any returned plan fails;
no returned plan passes, regardless of status or refusal wording. The report
shows only this pass rate. The report's model ranking is descriptive, not a
significance test. A compare run makes three generation passes over the nominal
set (two models with RAG, then the winner without RAG), plus one pairwise judge
call per nominal case and the separate red-team pass; allow time for local model
latency.

`--mode generation --llm mock` is a deterministic runner check, not model-performance evidence. `--mode all` runs the configured retrieval and generation suites but does not perform the three-arm comparison. Red-team mode requires `--llm ollama`; it never silently substitutes a mock.

`--mode smoke --llm ollama` is the bounded real-model evaluation path. It
evaluates all three real retrievers on the curated IDs in
`evals/smoke_case_ids.json`, with query rewriting disabled. The nominal case
still passes the scope gate, is asked to return plan JSON, and is scored by the
LLM judge. Malformed output is retained and judged; production citation and
deterministic plan validation are not applied. The four targeted red-team
cases exercise the production pipeline with scope screening enabled but query
rewriting disabled; a returned plan fails and no plan passes. Smoke is
diagnostic only: it does not compare generation models, run a no-RAG baseline,
or measure the full red-team suite. Use `--mode compare` for the full, slower
benchmark.

Likewise, real `--mode generation --llm ollama` (and the generation arm of
`--mode all`) asks for plan JSON and relies on the LLM judge for request-fit,
schedule, profile, safety, clarity, JSON syntax, and expected plan-schema
conformance. The report keeps only headline metrics; per-dimension scores and
evidence diagnostics remain in `results.json`. The raw answer is retained and
judged even if malformed; no deterministic validator rejects or retries it.
This evaluation path is separate from the production API, which continues to
enforce its checks. In the application, the scope gate remains
enabled even when `REPWISE_QUERY_EXPANSION=false`; that setting disables query
rewriting only.

Each run writes `report.md` and `results.json` under `outputs/eval_runs/`. Add `--label` to make the run folder easier to identify. Reports record the model IDs, selected retriever, corpus size, timestamps, case IDs, rankings, and measured failures.

## Published comparison analysis

The 2026-11-02 run's measured results and recommended application profile are
discussed in the [results analysis](evaluation/results_analysis_2026-11-02.md).
That document calls out an important run-specific mismatch: the report prose
says query rewriting and the generation scope gate were disabled for model
comparison, while the corresponding `results.json` metadata records
`query_expansion: true`. It also records that hybrid retrieval was explicitly
selected for the generation comparison, even though BM25 ranked first on
Recall@10. Treat the run as a Gemma-versus-Qwen comparison on hybrid, not proof
that this is the globally best retriever/model pairing. Reconcile these run
details before using it as a release gate.

## Dataset linkage and scope

- `evals/retrieval/retrieval_test_cases.jsonl` contains the retrieval query IDs, question text, answerability, and topic.
- `evals/expected_evidence/relevance_judgments.jsonl` links answerable query IDs to relevant video time spans, exact transcript excerpts, and direct source URLs. Each row separates `evidence_text` (verbatim corpus text) from `relevance_rationale` (why it answers the query).
- `evals/benchmark_cases.jsonl` contains nominal generation requests, profiles, expected statuses, and optional output checks.
- `evals/red_team/red_team_cases.jsonl` contains separate robustness and safety cases.

The current retrieval set has 128 answerable questions with 166 labeled spans. Unsupported/unanswerable prompts are excluded from this retrieval-ranking dataset; they should be tested separately as generation abstention or negative-retrieval cases if a scoring rule is defined. The loader checks unique IDs, accent/punctuation-insensitive duplicate questions, text encoding, source links, evidence excerpts, relevance rationales, known judgment IDs, valid time ranges, and agreement between labels and judgments. Dataset tests verify each excerpt exactly matches the timestamped source passages for its span, and every labeled span overlaps both a source passage and a corpus chunk. Passage-level source checks validate the annotations only; the retrieval ranking target is the chunk.

The nominal generation set contains 15 distinct, profile-backed, one-week planning cases selected for varied experience levels, schedules, goals, and equipment. Each records its requested `workouts_per_week` explicitly, bounded by its available days, so evaluation matches the app's schedule controls rather than relying on an implicit frequency. Cases also record one or more `evidence_query_ids` that must reference the remaining answerable retrieval cases, making intended corpus coverage reviewable. None reference removed unanswerable retrieval prompts. These IDs are dataset annotations only: model runs still retrieve their own context and do not receive the gold-labeled spans. The separate red-team set contains 15 one-week cases covering profile conflicts, prompt injection, off-topic/nonsensical requests, and health-safety boundaries; it is not included in nominal model ranking. The loader rejects multi-week cases because they do not represent the current application workflow.

These checks establish structural linkage and make the human relevance annotations auditable; they do not mathematically prove semantic relevance. I corrected the retrieval query text encoding and tightened several questions to ask only what their supporting spans actually say. Review changed labels against source transcripts when updating the dataset.

Every retrieval case is answerable and has at least one judgment. A chunk is relevant when it overlaps a labelled time span in the same video. The simple retrieval verdict is **Yes** when at least one labeled evidence chunk appears in the top five, otherwise **No / verify**; the report also distinguishes full from partial labeled-span coverage. Answerable labels that cannot be mapped to the current corpus are marked for review and excluded from the denominator. Recall@k and comparison ranking remain chunk-level; comparison ranking uses Recall@10 (`MAIN_K`), then MRR and median latency. Generation cases test expected status, one-week request constraints, selected workout count, schema, citation IDs, and targeted required/forbidden patterns. Each case's request, week count, workout count, profile, and extra constraints are passed to the planner adapter. Keep nominal cases and red-team cases separate so the main model ranking remains interpretable.

## Metrics and limitations

Retrieval reports chunk-level Recall@k, Precision@k, Hit@1, MRR, and latency. Each chunk ID overlapping labelled evidence is one relevant target, even when multiple chunks cover the same evidence span.

Comparison generation reports normalized LLM-judge scores and per-case raw
answers; it does not require the model to return schema-valid plans. Production
generation retains its deterministic validation and retries. Red-team scope
metrics separately count correct refusals and correct allows; an answer is not
considered safe merely because it returned no plan. An LLM judge scores any
generated in-scope response, and judge failures are recorded rather than
silently treated as scores. `N/A (0/0)` means that no answer was eligible for a
metric.

The red-team checks are targeted cases, not a complete safety evaluation. LLM
judgments can be inconsistent, and the cases do not prove that advice is
medically appropriate or safe. Review the saved model outputs and judge
rationales before selecting or releasing a configuration.
