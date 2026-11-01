# Evaluation protocol

RepWise evaluation has three distinct datasets and workflows:

1. **Retrieval benchmark:** labelled questions and evidence time spans compare BM25, semantic, and hybrid retrieval.
2. **Generation benchmark:** the same nominal requests compare model A with RAG, model B with RAG, and model A without RAG.
3. **Red-team checks:** a separate set of robustness and safety requests exercises the real RAG generation path; it is not mixed into nominal model ranking.

## Prepare models and corpus

The corpus is local and is not committed. Prepare it as described in the README, including the chunk JSONL files and semantic index. Install Ollama and pull two different instruction-tuned chat models. The Ollama
tags below select the instruct variants even though `-instruct` is not part of
the tag:

```powershell
ollama pull qwen3.5:9b
ollama pull gemma4:12b
```

Set `OLLAMA_MODEL` to model A and `OLLAMA_COMPARISON_MODEL` to model B in `.env`, or pass both explicitly to the runner. The defaults are `qwen3.5:9b` and `gemma4:12b`; both must be available from the configured Ollama service. Semantic and hybrid retrieval also need the configured embedding model and current local index.

## Run the evaluation

```powershell
python -m pytest tests\evaluation tests\generation tests\retrieval -q

# Rank all three retrieval strategies and compare two RAG models plus a no-RAG
# baseline. Then red-team the highest-ranked RAG setup.
python -m src.evaluation.runner --mode compare

# Optionally pin the retrieval strategy used by the RAG generation arms.
python -m src.evaluation.runner --mode compare --retriever bm25 --model-a qwen3.5:9b --model-b gemma4:12b

# Run red-team alone; replace the model ID with best_generation_model from the compare report.
python -m src.evaluation.runner --mode red-team --llm ollama --retriever bm25 --model-a qwen3.5:9b

# Run individual diagnostic suites.
python -m src.evaluation.runner --mode retrieval --retriever bm25
python -m src.evaluation.runner --mode generation --llm ollama --retriever hybrid
```

`compare` first scores the production BM25, semantic, and hybrid retrievers on the same retrieval questions. Evaluation does not carry a second BM25 implementation: the benchmark calls `src.retrieval.sparse.BM25Retriever`, including its title and body-part fields. Comparison mode constructs BM25 and the semantic model once, then shares those components with hybrid and the selected generation retriever. It selects by Recall@5, then MRR, then median retrieval latency; `--retriever` overrides only the selection used for the generation arms. Both RAG arms share the same selected retriever instance and receive identical cases. The no-RAG baseline calls model A directly with the same request and profile but no retriever or evidence. It uses a separate no-RAG evidence policy: source-citation and retrieved-support requirements are removed, while shared schedule, safety, and output-schema rules remain. It must not invent citations or claim that recommendations are evidence-backed. All arms use the same schema and deterministic plan validation.

Only the two RAG arms compete for the best generation setup; the no-RAG arm is a baseline, not a candidate for a RAG red-team run. Generation ranking is by case pass rate, request-constraint compliance, schema validity, then median latency. The comparison automatically runs the separate 15-case red-team suite using the top-ranked RAG model and selected retriever, and records the winning setup and red-team results together. This is a descriptive ordering, not a significance test. The report preserves per-case failures and metric numerators/denominators so that a rank is not mistaken for a release decision. Compare runs call both Ollama models on nominal cases and the winner on red-team cases; allow several minutes.

`--mode generation --llm mock` is a deterministic runner check, not model-performance evidence. `--mode all` runs the configured retrieval and generation suites but does not perform the three-arm comparison. Red-team mode requires `--llm ollama`; it never silently substitutes a mock.

Each run writes `report.md` and `results.json` under `outputs/eval_runs/`. Add `--label` to make the run folder easier to identify. Reports record the model IDs, selected retriever, corpus size, timestamps, case IDs, rankings, and measured failures.

## Dataset linkage and scope

- `evals/retrieval/retrieval_test_cases.jsonl` contains the retrieval query IDs, question text, answerability, and topic.
- `evals/expected_evidence/relevance_judgments.jsonl` links answerable query IDs to relevant video time spans, exact transcript excerpts, and direct source URLs. Each row separates `evidence_text` (verbatim corpus text) from `relevance_rationale` (why it answers the query).
- `evals/benchmark_cases.jsonl` contains nominal generation requests, profiles, expected statuses, and optional output checks.
- `evals/red_team/red_team_cases.jsonl` contains separate robustness and safety cases.

The current retrieval set has 150 questions: 128 answerable with 166 labeled spans and 22 unanswerable. The loader checks unique IDs, accent/punctuation-insensitive duplicate questions, text encoding, answerability rationale, source links, evidence excerpts, relevance rationales, known judgment IDs, valid time ranges, and agreement between labels and judgments. Dataset tests verify each excerpt exactly matches the timestamped source passages for its span, and every labeled span overlaps both a source passage and a corpus chunk.

The nominal generation set contains 30 distinct, profile-backed, one-week planning cases. Each records its requested `workouts_per_week` explicitly, bounded by its available days, so evaluation matches the app's schedule controls rather than relying on an implicit frequency. Cases also record one or more `evidence_query_ids` that must reference answerable retrieval cases, making the intended corpus coverage reviewable. These IDs are dataset annotations only: model runs still retrieve their own context and do not receive the gold-labeled spans. The separate red-team set contains 15 one-week cases covering profile conflicts, prompt injection, off-topic/nonsensical requests, and health-safety boundaries; it is not included in nominal model ranking. The loader rejects multi-week cases because they do not represent the current application workflow.

These checks establish structural linkage and make the human relevance annotations auditable; they do not mathematically prove semantic relevance. I corrected the retrieval query text encoding and tightened several questions to ask only what their supporting spans actually say. Review changed labels against source transcripts when updating the dataset.

Retrieval cases marked answerable must have at least one judgment; unanswerable cases must have none. A chunk is relevant when it overlaps a labelled time span in the same video. Retrieval scores use answerable cases; unanswerable cases have no relevant span to rank. Generation cases test expected status, one-week request constraints, selected workout count, schema, citation IDs, and targeted required/forbidden patterns. Each case's request, week count, workout count, profile, and extra constraints are passed to the planner adapter. Keep nominal cases and red-team cases separate so the main model ranking remains interpretable.

## Metrics and limitations

Retrieval reports Recall@k, Precision@k, Hit@1, MRR, and latency. Each relevant span is counted once for recall even if it overlaps multiple chunks.

Generation reports case pass rate, expected-status/abstention results, request-constraint compliance, output-schema validity, citation ID validity, optional claim-level citation precision/faithfulness, and latency. `N/A (0/0)` means a metric was not applicable. The no-RAG baseline is expected to leave `source_ids` empty, so citation-ID validity is N/A for it; it is not penalized for lacking citations. In RAG arms, citation-ID validity only confirms that a cited passage ID was supplied, not that it semantically supports the exercise. The current plan adapter does not expose claim-level annotations, so semantic grounding may be N/A.

The red-team checks are targeted cases, not a complete safety evaluation. Regex checks can miss paraphrases or match harmless wording; passing does not prove that advice is medically appropriate or safe. Review real model outputs and failures before using results to choose or release a configuration.
