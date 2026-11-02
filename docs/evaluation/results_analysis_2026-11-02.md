# Evaluation analysis and recommended application settings

This analysis interprets the comparison run in
[`outputs/eval_runs/20261102-034806/report.md`](../../outputs/eval_runs/20261102-034806/report.md).
The machine-readable scores, per-case outputs, and run metadata are in
[`results.json`](../../outputs/eval_runs/20261102-034806/results.json).

## Executive recommendation

For the current local-first application, keep the balanced profile already
represented by the defaults: **Qwen 3.5 9B + hybrid retrieval**, using the
existing multilingual E5 small index and five evidence chunks. This is a
practical trade-off, not the benchmark's scalar-score winner:

- **Gemma 4 12B + hybrid** had the highest average judged answer quality in
  this run, 0.886 versus Qwen's 0.871.
- **Qwen 3.5 9B + hybrid** generated answers about 49 seconds faster at the
  median and won 10 of the 13 valid pairwise judgments (Gemma won 3; 2 judge
  errors).
- **BM25** led Recall@10, while hybrid tied it on the top-five evidence hit
  count and ranked the first relevant chunk higher on average.

Choose Gemma instead when the small measured answer-quality advantage matters
more than local response time, and confirm the result with a larger, repeated
comparison first. Choose BM25 when maximizing this dataset's Recall@10 is the
priority and its latency is acceptable. The run does not establish a universally
optimal model or retriever.

## What the retrieval results say

All three retrieval strategies were scored on the same 128 answerable queries
and chunk-level evidence labels. The evaluator requested 50 results per query;
the headline Recall@10 measures the first 10, while the top-five verdict checks
whether any labeled evidence appears in the first five.

| Retriever | Recall@10 | Top-five evidence | MRR | Hit@1 | Median latency |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.755 (96.583/128) | 99/128 | 0.577 | 0.422 (54/128) | 1,966 ms |
| Hybrid | 0.729 (93.333/128) | 99/128 | 0.606 | 0.477 (61/128) | 25.3 ms |
| Semantic | 0.595 (76.167/128) | 79/128 | 0.475 | 0.344 (44/128) | 14.0 ms |

BM25 is the recall-first choice: its Recall@10 is 2.6 percentage points above
hybrid. Hybrid has the stronger early-rank signal: it has the same number of
top-five hits as BM25, but a higher MRR and seven more Hit@1 cases. This favors
hybrid when the best evidence near the top matters, while still combining
lexical matching with semantic matches. Semantic-only retrieval trails on every
reported quality measure here; that does not make embeddings useless, since
they still contribute to hybrid fusion.

The reported median BM25 latency is about 78 times hybrid's. Treat this
particular latency comparison as a warning, not a production promise: the run
also records markedly different retrieval medians between the two generation
arms despite using the same selected retriever (about 7,986 ms for Qwen versus
54 ms for Gemma). Re-run with warmed indexes, alternating model/strategy order,
and per-query timing before making a latency-based architecture decision.

## What the generation results say

| RAG model | Answer quality | Generated | Judge coverage | Median generation |
|---|---:|---:|---:|---:|
| Gemma 4 12B | 0.886 (186/210) | 15/15 | 15/15 | 166.6 s |
| Qwen 3.5 9B | 0.871 (183/210) | 15/15 | 15/15 | 117.8 s |

Gemma's lead is only 3 points out of 210 possible points (about 1.4 percentage
points on the normalized score), whereas its median generation time is about
49 seconds longer. Meanwhile, the pairwise judge preferred Qwen 10 times and
Gemma 3 times among the 13 cases it could score; it returned errors for two
cases. The aggregate score and pairwise choice therefore tell different
stories. With only 15 requests and a single local judge, neither result should
be treated as a definitive quality ordering.

The RAG/no-RAG comparison is a strong signal that retrieved context mattered
for these requests: Gemma with RAG scored 0.886, while its no-RAG arm scored
0.000 despite generating all 15 answers. The no-RAG arm scored zero for request
fit and schedule/profile safety. This supports retaining retrieval in the main
application, but it does not prove that every generated claim is grounded or
that RAG improves every future request.

The judge examined raw model text; production plan-schema validation and
deterministic plan checks were not applied in this comparison. The judge
rationales also repeatedly cite JSON formatting and evidence support, which
are useful review targets. Inspect the saved raw outputs and per-dimension
scores before treating the aggregate as a release-quality measure.

The red-team result was 15/15 under the runner's binary rule: no plan was
returned for any of the fixed test cases. This is a positive result for those
cases, not broad safety certification; the metric does not distinguish an
appropriate refusal from other no-plan outcomes.

## Recommended values for the main application

These recommendations distinguish settings exercised by this run from
defaults that were not independently tuned.

| Setting | Recommended value | Why / qualification |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3.5:9b` | Balanced default: much lower measured generation time and the more frequent valid pairwise preference. Use `gemma4:12b` for a quality-first experiment; it won the aggregate score by a small margin. |
| `REPWISE_RETRIEVAL` | `hybrid` | Balanced default: same top-five hit count as BM25, better MRR and Hit@1, and ranked evidence from both lexical and semantic search. BM25 is the recall-first alternative. |
| `REPWISE_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Keep the current multilingual embedding model and matching index. This run tested retrieval strategies, not competing embedding models. |
| `REPWISE_EMBEDDING_INDEX` | `data/corpus/index/semantic.npz` | Keep the index built from the current corpus and embedding model; rebuild it whenever either changes. |
| `REPWISE_TOP_K` | `5` | Keep the current evidence budget. It is not tuned by this report: Recall@10 is an evaluation metric, and the evaluator retrieved 50 candidates. Do not copy `MAIN_K=10` into the application setting. |
| `REPWISE_QUERY_EXPANSION` | `true` | Retain the application default provisionally; the run's metadata and prose disagree about whether expansion was enabled, so this is not a clean ablation result. |
| `OLLAMA_QUERY_EXPANSION_MODEL` | `qwen3.5:4b` | Retain the existing lightweight model for scope analysis and query rewriting. The run did not compare alternatives for this role. |
| `OLLAMA_COMPARISON_MODEL` | `gemma4:12b` | Keep as the evaluation challenger, not the normal application generator unless choosing the quality-first profile. |
| `OLLAMA_JUDGE_MODEL` | `qwen3.5:4b` | Evaluation-only judge; the run does not show that it should be part of the normal answer path. |
| Hybrid RRF constant | `60` | Keep the implemented value; this run compares retrievers, not RRF constants. |

In short, the score-winning benchmark configuration is **Gemma 4 12B + hybrid**.
The recommended balanced application configuration is **Qwen 3.5 9B + hybrid**,
with query expansion enabled as the current default but explicitly awaiting a
clean on/off verification. No code default is changed on the strength of this
single run.

## Limits and next evaluation

Two run details need resolution before a stronger recommendation:

1. The report says query rewriting and the generation scope gate were disabled
   for model comparison, while `results.json` records
   `"query_expansion": true`. The report also says hybrid was the selected
   retriever; metadata confirms that selection was a requested override, even
   though BM25 ranked first on Recall@10. Consequently, the model comparison
   establishes Gemma-versus-Qwen on hybrid, not the best automatic
   retriever/model combination.
2. The retrieval latency values differ sharply between otherwise comparable
   paths. Confirm timing boundaries, warm-up, and cache behavior before using
   these figures for capacity planning.

For a stronger follow-up, repeat the comparison with expansion explicitly
enabled and disabled, report per-case latency after warm-up, alternate run
order, validate outputs through the production checks, and use more varied
generation cases. Preserve the same corpus and relevance labels, and review the
two pairwise judge errors and the raw judge rationales. The current comparison
has no confidence intervals or significance tests.
