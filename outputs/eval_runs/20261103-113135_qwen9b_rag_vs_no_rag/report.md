# RepWise RAG vs no-RAG benchmark

- **Run:** 2026-11-03T11:31:35 · qwen9b_rag_vs_no_rag
- **Generation model:** `qwen3.5:9b`
- **Answer judge:** `qwen3.5:4b`
- **RAG retriever:** `hybrid`
- **Corpus:** 586 chunks from `C:\Users\flori\OneDrive - aivancity\5th year\Generative AI\Project\RepWise\data\corpus\chunks`
- **Generation cases:** 15 identical cases per arm
- **Query expansion in RAG arm:** `False`

Both arms use the same model, unconstrained response prompt, and nominal cases. The no-RAG arm receives no retrieved evidence. Answer JSON and plan-schema formatting are not required or scored.

| System | Answer quality | Evidence support | Answer generated | Judge coverage | Request fit | Schedule/profile safety | Median generation latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen3.5:9b + RAG` | 0.883 (159/180) | 0.400 (12/30) | 1.000 (15/15) | 1.000 (15/15) | 1.000 (30/30) | 0.950 (57/60) | 82982.7 ms |
| `qwen3.5:9b without RAG` | 0.789 (142/180) | 0.000 (0/30) | 1.000 (15/15) | 1.000 (15/15) | 0.967 (29/30) | 0.900 (54/60) | 61965.9 ms |

## Interpretation

- Answer quality averages all five general quality dimensions plus evidence support. With no retrieved evidence, the judge cannot award evidence support; raw answers and per-dimension scores are retained in `results.json`.
- This is a fixed-dataset ablation, not a significance test or a complete safety evaluation.
- No production plan validation or retry is applied to either arm.
