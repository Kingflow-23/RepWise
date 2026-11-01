# Evaluation metrics

The runner reports retrieval, generation, and (in comparison mode) ranked results. Every rate includes its numerator and denominator; a zero denominator is `N/A`, not a perfect score. Metric formulas are in `src/evaluation/metrics/`.

## Retrieval

| Metric | What it tells us |
|---|---|
| Recall@k | Fraction of relevant chunk IDs retrieved in the top k. |
| MRR | How high the first relevant chunk appears in the ranking. |
| Hit@1 | Whether the first chunk is relevant. |
| Top-five evidence verdict | Yes when any labeled evidence chunk appears in the top five; the report separately labels full, partial, or no labeled-span coverage. |
| Precision@k | Fraction of the top k chunk results marked relevant. |
| Median and p95 latency | How long each retrieval request takes. |

Judgments are stored as auditable video time ranges rather than fixed chunk IDs. At evaluation time, each corpus chunk that overlaps a relevant span by at least one second is marked relevant. This keeps the source labels stable across re-chunking while making every retrieval metric operate on chunks. BM25, semantic, and hybrid runs use the same query set and judgments. BM25 in evaluation is the production `src.retrieval.sparse.BM25Retriever`, not a benchmark-specific reimplementation. These metrics can show whether hybrid performed better on this test set; they cannot guarantee better generation.

## Generation

| Metric or check | What it tells us |
|---|---|
| Answer quality | Normalized judge score across request fit, schedule, profile constraints, domain safety, clarity, JSON validity, and plan-schema validity. Each dimension is scored 0-2. |
| Answer generation rate | Fraction of cases that produced an answer. |
| Request fit | Whether the answer addresses the requested goal. |
| Schedule/profile safety | Combined judge score for following the requested schedule and respecting profile constraints. |
| Generation latency | Median raw model-generation time. |

Generation reports must identify `mocked` versus `real Ollama`, retrieval strategy, embedding model, and generation model. Mock planners are small simulations for runner tests and deliberate failure examples, not performance evidence for the application.
The report shows only these headline metrics; per-dimension scores, evidence
support, and judge rationale remain available in `results.json`.

The bounded `--mode smoke --llm ollama` workflow checks all three real retrievers
on a curated retrieval subset, then runs one nominal generation case and four
targeted red-team cases through one RAG model. It disables query rewriting and
validation retries. This is useful for an end-to-end check, not a model ranking,
no-RAG comparison, or full safety gate.

## Controlled comparison

Comparison mode ranks BM25, semantic, and hybrid using chunk Recall@10
(`MAIN_K`), MRR, and median retrieval latency; the per-query verdict remains
"any evidence in the top five." It then runs both generation models with RAG on
the same nominal cases and ranks them by the common LLM-judge answer-quality
score. The selected model is then run on the same cases without RAG. Generation
answers are preserved as raw text and are not passed through production plan
schema or deterministic validation. The separate Qwen judge scores the same
dimensions for each model and architecture arm; evidence support is reported
only for RAG.

The red-team suite reports one metric: pass rate. Any returned plan fails;
no returned plan passes, regardless of status or refusal wording. The
comparison is descriptive and fixed-dataset only; it has no confidence
intervals or significance tests.

## What the metrics cannot establish

Code can verify that IDs exist, a quote matches, or schedule constraints pass. It cannot decide whether advice is medically appropriate or whether a passage semantically supports a recommendation. Those claims require targeted human review. The corpus and relevance labels also have known coverage and annotation limits; see [corpus scope](../../docs/ingestion/knowledge_base_scope.md) and [safety cases](red_teaming.md).
