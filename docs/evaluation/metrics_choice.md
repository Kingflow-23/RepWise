# Evaluation metrics

The runner reports separate retrieval and generation results. Every rate includes its numerator and denominator; a zero denominator is `N/A`, not a perfect score. Metric formulas are in `src/evaluation/metrics/`.

## Retrieval

| Metric | What it tells us |
|---|---|
| Recall@k | Fraction of labelled evidence spans covered by the top k chunks. |
| MRR | How high the first relevant chunk appears in the ranking. |
| Hit@1 | Whether the first result overlaps any labelled span. |
| Precision@k | Fraction of the top k results overlapping labelled evidence. |
| Median and p95 latency | How long each retrieval request takes. |

Labels are video time ranges rather than chunk IDs, so the same relevance judgments can be used after re-chunking. When one relevant span overlaps multiple chunks, it is counted once for Recall. BM25, semantic, and hybrid runs must use the same query set and judgments. These metrics can show whether hybrid performed better on this test set; they cannot guarantee better generation.

## Generation

| Metric or check | What it tells us |
|---|---|
| Expected-status accuracy | Whether each case ended with an accepted status such as `ok`, `abstain`, or `clarify`. |
| Output-schema validity | Whether released plans have readable sessions and named exercises. The application also applies Pydantic and deterministic plan checks before release. |
| Citation ID validity | Whether cited passage IDs were actually supplied to the model. This is provenance validation, not proof of semantic grounding. |
| Citation precision and faithfulness | Optional quote and claim checks when the adapter provides claims and citations. They may be `N/A` for the real `PlanPipeline` adapter, which does not currently produce separate claim annotations. |
| Safety-case pass rate | Captured by generation pass rate across cases with expected status and required/forbidden text checks. |
| Median and p95 latency | Time spent in the evaluated generation call. |

Generation reports must identify `mocked` versus `real Ollama`, retrieval strategy, embedding model, and generation model. Mock planners are small simulations for runner tests and deliberate failure examples, not performance evidence for the application.

## What the metrics cannot establish

Code can verify that IDs exist, a quote matches, or schedule constraints pass. It cannot decide whether advice is medically appropriate or whether a passage semantically supports a recommendation. Those claims require targeted human review. The corpus and relevance labels also have known coverage and annotation limits; see [corpus scope](../../docs/ingestion/knowledge_base_scope.md) and [safety cases](red_teaming.md).
