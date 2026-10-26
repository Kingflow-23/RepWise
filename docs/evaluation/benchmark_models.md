# Benchmark: models and systems to compare

This document lists the models the benchmark will run, and in which order. The machine-readable list is [`evals/benchmark/benchmark_models.json`](../../evals/benchmark/benchmark_models.json). Metric definitions are in [`metrics_choice.md`](metrics_choice.md).

LLM availability was checked on **2026-10-24** on the provider pages ([Groq models](https://console.groq.com/docs/models), [Cerebras models](https://inference-docs.cerebras.ai/models/overview)). Providers retire models often: check these pages again before each benchmark run, and record the date in the run label.

## Rules of the benchmark

1. **One change at a time.** Each stage changes one component and freezes all the others, so every difference in score can be explained (course, lab report).
2. **Same inputs for everyone.** Same test split, same profiles, same prompt template, same k, same output budget.
3. **Tune on `dev`, report on `test`.** The winner of each stage is chosen on `dev`; the final table uses `test` only.
4. **Repeat stochastic runs.** LLM runs use temperature 0 and are repeated 3 times; the mean and the spread are reported (project note, section 6).
5. **Report cost next to quality.** Latency (median and 95th percentile) and, for LLMs, tokens per request.

The benchmark runs in three stages: retrieval, then generation, then the whole system.

## Stage A: retrieval (no LLM)

Goal: choose the retriever configuration. Metrics: Recall@1/3/5, MRR, nDCG@5, ms/query.

| ID | Type | Model | Size | Why it is in the list |
|---|---|---|---|---|
| `bm25` | Sparse | BM25, k1 = 1.5, b = 0.75 (built into `src/evaluation`) | – | Baseline. Already measured on the 150 draft queries: Recall@5 = 0.67, MRR = 0.51. French Recall@5 = 0.78, but English Recall@5 = 0.16: strong on exact words, blind to paraphrases and to other languages. |
| `bge-m3` | Dense | `BAAI/bge-m3` | 568 M | Multilingual, 8,192-token input (longer than our chunks), recommended in the course. Main candidate. |
| `me5-large` | Dense | `intfloat/multilingual-e5-large` | 560 M | Strong multilingual reference. **Caution:** maximum input of 512 tokens, but a 450-word French chunk is about 600–700 tokens, so the end of each chunk is cut off. Interesting to test precisely for that reason. Requires the `query: ` / `passage: ` prefixes. |
| `qwen3-emb-0.6b` | Dense | `Qwen/Qwen3-Embedding-0.6B` | 0.6 B | Recent, multilingual, long input; cited in the course. |
| `nomic-v2` | Dense | `nomic-ai/nomic-embed-text-v2-moe` | ~475 M | Light multilingual alternative; tests whether a smaller model is enough. |
| `hybrid` | Hybrid | BM25 + best dense model, fused with RRF (k = 60) | – | Course default: combines exact words and meaning. |
| `hybrid+rerank` | Rerank | best hybrid → `BAAI/bge-reranker-v2-m3` on the top 50 | 568 M | Course two-stage pipeline: should raise MRR and Hit@1. |
| `hybrid+rerank-jina` | Rerank | best hybrid → `jinaai/jina-reranker-v2-base-multilingual` | 278 M | Smaller reranker; checks whether a cheaper one is enough. Non-commercial licence: acceptable for this academic project. |

All of these run locally (`sentence-transformers`), so no document leaves the laptop.

Order of runs: BM25 → the 4 dense models → hybrid with the best dense model → the 2 rerankers on top of the best hybrid. That is 8 runs, each taking minutes on a laptop.

Also to report: **per-language results**. English and code-switched queries are where BM25 fails and multilingual embedders should help.

## Stage B: generation (LLM)

Goal: choose the LLM, with the retriever fixed to the winner of stage A. Metrics: abstention accuracy, false abstention rate, citation precision, unsupported claim rate, grounded claim rate, schema validity, hard constraint adherence, program validity, red-team pass rate, latency, tokens.

| ID | Provider | Model ID | Status | Why it is in the list |
|---|---|---|---|---|
| `llama-8b` | Groq | `llama-3.1-8b-instant` | Production | Small, fast model. Shows the quality floor and whether a cheap model is enough. |
| `llama-70b` | Groq | `llama-3.3-70b-versatile` | Production | Large general-purpose model, good French. Expected strong baseline. |
| `gpt-oss-20b` | Groq | `openai/gpt-oss-20b` | Production | Mid-size open-weight reasoning model. |
| `gpt-oss-120b` | Groq | `openai/gpt-oss-120b` | Production | Largest available open-weight model. Expected best on constraints and JSON. |
| `qwen-27b` (optional) | Groq | `qwen/qwen3.8-27b` | Preview | Different model family. Preview models can change or disappear, so it is reported apart and never chosen for the final system. |

**Backup provider:** Cerebras serves `gpt-oss-120b` and `qwen-3.8-27b` (`CEREBRAS_API_KEY` in `.env.example`). Use it if Groq free-tier rate limits block a run. Always record which provider served a run, because the same model can behave slightly differently.

Settings for all LLMs: temperature 0, same system prompt, same output budget, JSON output mode when the provider supports it, 3 repetitions.

## Stage C: system comparison (project note, section 6)

Goal: answer the research question. All systems get the same profile, the same logs and the same constraints at each decision point. Baselines must not be weakened by withholding information.

| ID | System | What it tests |
|---|---|---|
| `prompt-only` | Best LLM from stage B, no retrieval | What a general-purpose chatbot does alone |
| `basic-rag` | Dense retrieval + LLM, no validator | The value of retrieval |
| `v1` | Retrieval + LLM + deterministic validation and revision | The value of validators |
| `v2` | V1 + history + adaptation engine | The full system |
| `v2-no-adapt` | V2 with adaptation turned off (ablation) | The value of the adaptation engine alone |

`v2` and `v2-no-adapt` need the adaptation engine, which has not been built yet.

## LLM judge (later, for level-2 grounding)

- The judge must come from a **different model family** than the generator it judges, to limit self-preference. For example, `llama-3.3-70b-versatile` judges a `gpt-oss` generator, and `openai/gpt-oss-120b` judges a Llama generator.
- Before use, its verdicts are compared with about 30 human labels. If agreement is low, the judge is not used.

## Running a benchmark entry

Today the runner already supports the BM25 baseline:

```
python -m src.evaluation.runner --suite retrieval --retriever bm25 --label A_bm25_2026-10-24
```

The other entries need small adapters (`retrieve(query, k)` or `plan(case)` functions) that Florian's pipeline will provide, or that we write in `src/evaluation/adapters.py`. Name each run after its benchmark ID (`--label A_bge-m3_<date>`) so that the `results.json` files can be compared later.
