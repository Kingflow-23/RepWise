# RepWise evaluation report

- **Run:** 2026-10-30T16:55:14 · dataset-cleanup-validation
- **Mode:** retrieval
- **Retriever:** `bm25` · **Planner:** `mock-abstain`
- **Generation mode:** not used
- **Models:** embedding `not used` · generation `not used`
- **Corpus:** 603 chunks from `C:\Users\flori\OneDrive - aivancity\5th year\Generative AI\Project\RepWise\data\corpus\chunks`

Values are shown as `value (numerator/denominator)`. `N/A (0/0)` means no applicable case, not a perfect score. ↑ higher is better, ↓ lower is better.

## Retrieval

| Metric | Value |
|---|---|
| pass rate ↑ | 0.695 (89/128) |
| recall@1 ↑ | 0.367 (47/128) |
| precision@1 ↑ | 0.414 (53/128) |
| recall@3 ↑ | 0.581 (74.333/128) |
| precision@3 ↑ | 0.276 (35.333/128) |
| recall@5 ↑ | 0.742 (95/128) |
| precision@5 ↑ | 0.220 (28.2/128) |
| recall@10 ↑ | 0.793 (101.5/128) |
| precision@10 ↑ | 0.125 (16/128) |
| hit@1 ↑ | 0.414 (53/128) |
| mrr ↑ | 0.557 (71.284/128) |
| latency_ms_median ↓ | 6.9 |
| latency_ms_p95 ↓ | 9.2 |

Failure details are in `results.json`.
