# Evaluation protocol

Owner: Ephraïm. Code in `src/evaluation/`, datasets in `evals/`, tests in `tests/evaluation/`.

The goal is to measure whether RepWise works with numbers that can be reproduced, rather than by looking at a few convincing examples. Each layer is scored separately, so a failure points to the component that has to change.

| Layer | What is measured | Needs an LLM? | Fixed by |
|---|---|---|---|
| Retrieval | Recall@k, Precision@k, Hit@1, MRR, nDCG@k, latency | No | ingestion / retrieval |
| Grounding | Citation precision, unsupported and grounded claim rates, abstention | No (quote check) | generation |
| Program | Schema validity, hard constraints, preferences, completeness, duration overrun, program validity | No | generation / validation |
| Adaptation | Red-team trajectories RT9–RT15 | No | adaptation engine (not built yet) |

## Running it

```
pip install -r requirements.txt
python -m pytest tests/evaluation                 # unit tests of the metrics and datasets
python -m src.evaluation.runner --suite all       # full evaluation, BM25 baseline + mock planner
```

Useful options:

| Option | Meaning |
|---|---|
| `--suite retrieval\|generation\|all` | Which suites to run |
| `--split dev\|test\|all` | Tune on `dev`, report on `test` |
| `--retriever bm25` or `--retriever pkg.module:function` | System under test for retrieval |
| `--planner mock-naive\|mock-abstain` or `--planner pkg.module:function` | System under test for generation |
| `--chunks-dir PATH` | Chunk corpus (default `data/corpus/chunks`) |
| `--k 1 3 5 10`, `--main-k 5` | Cut-offs, and the k that decides pass/fail for a retrieval case |
| `--label NAME` | Name of the run folder |
| `--gates` | Exit code 1 if a release gate fails (for CI) |

Each run writes `outputs/eval_runs/<timestamp>_<label>/report.md` (readable) and `results.json` (everything, for comparisons).

## Interface contract with the other components

```python
retrieve(query: str, k: int) -> list[dict]
# best first; each chunk has video_id, start, end, text, chunk_id

plan(case: dict) -> dict
# input  {"request": str, "lang": str, "profile": dict}
# output {"status": "ok" | "abstain" | "clarify" | "blocked" | "hold",
#         "message": str,
#         "plan": {"sessions": [{"name", "warmup_min",
#                                "exercises": [{"name", "equipment": [...], "sets", "reps",
#                                               "rest_s", "effort": {"scale": "RIR"|"RPE", "value"},
#                                               "tempo"}]}]} | None,
#         "claims": [{"text", "kind": "evidence"|"assumption"|"policy",
#                     "citations": [{"chunk_id", "quote"}]}],
#         "retrieved_ids": [chunk_id, ...]}
```

`quote` must contain words copied from the cited chunk (at least 4). That is what makes citations checkable without an LLM (course slide 27, level 1). Equipment uses this vocabulary: `barbell, rack, bench, dumbbells, cable, machine, pullup_bar, bodyweight`.

**Note for ingestion:** chunk records do not store a `chunk_id` yet. Until they do, the runner derives `<video_id>_<first passage>-<last passage>` (for example `BBzZktps5D4_p0001-p0012`). The generation pipeline must cite the same IDs.

## Datasets (`evals/`)

| File | One line = |
|---|---|
| `retrieval/retrieval_test_cases.jsonl` | a query: `id, query, lang, split, answerable, topic, review_status` |
| `expected_evidence/relevance_judgments.jsonl` | a relevant span: `query_id, video_id, start, end, grade (1-3), note` |
| `profiles/fixed_profiles.jsonl` | a user profile: days, max minutes, equipment, exclusions, preferences |
| `scenarios/program_scenarios.jsonl` | a program request and its expected status |
| `red_team/safety_cases.jsonl` | RT1–RT15: request or trajectory, expected status, forbidden and required patterns |

Design choices, following the course and the project note:

- **Relevance is labelled as time spans, not chunk IDs.** A chunk counts as relevant when it overlaps a labelled span by at least one second. Labels therefore survive re-chunking, and a span covered by two overlapping chunks is counted once in recall and nDCG.
- **10–25 % of the queries are unanswerable.** They measure abstention instead of invention (slide 35).
- **Queries are written the way a student would ask,** not copied from the transcript, which would favour keyword search. They include English and code-switched queries, and metrics are broken down by language (slide 16).
- **`split`:** tune on `dev`, report on `test`. Freeze `test` before tuning anything.
- **`review_status: draft`:** labels proposed from the transcripts. Every label must be checked by a teammate and set to `reviewed` before the test set is frozen.
- **Patterns** (`forbidden_patterns`, `required_patterns`) are regexes matched against the normalised answer: lower case, no accents, no punctuation. Write `reeducation`, not `rééducation`.

## Metric definitions

Every metric is reported as `value (numerator/denominator)`. A zero denominator prints `N/A (0/0)`, never 100 %.

| Metric | Definition |
|---|---|
| Recall@k ↑ | labelled spans covered by the top k / labelled spans; mean over answerable queries |
| Precision@k ↑ | relevant chunks in the top k / k |
| Hit@1 ↑ | first result relevant |
| MRR ↑ | mean of 1 / rank of the first relevant chunk, 0 if none |
| nDCG@k ↑ | DCG@k / ideal DCG@k with grades 1–3 and a log2 discount; each span earns its gain once |
| Citation precision ↑ | (claim, citation) pairs whose quote matches a real chunk / all pairs |
| Unsupported claim rate ↓ | evidence claims without any matching quote / evidence claims |
| Grounded claim rate ↑ | evidence claims supported by a chunk that was in the retrieved context / evidence claims |
| Abstention accuracy ↑ | correct abstentions / cases requiring abstention |
| False abstention rate ↓ | abstentions / answerable cases |
| Output schema validity ↑ | released plans with a valid structure / released plans |
| Hard constraint adherence ↑ | satisfied checks / applicable checks (days, duration per session, equipment per exercise, exclusions, sets bounds, effort scale) |
| Plans with zero violations ↑ | valid plans breaking no hard constraint / valid plans |
| Preference satisfaction ↑ | preferred exercises included and disliked ones avoided / applicable preferences |
| Parameter completeness ↑ | non-placeholder values among name, sets, reps, rest_s, effort / required fields |
| Duration overrun ↓ | mean of max(0, estimated − available) / available |
| Program validity rate ↑ | released plans with valid schema, zero violations and complete parameters / released plans |
| Red-team pass rate ↑ | red-team cases with no failure / red-team cases run |

Duration estimate (project note, section 4): `T = warm-up + Σ [sets × reps_max × s/rep + (sets − 1) × rest] + (n_exercises − 1) × transition`. The defaults (10 min warm-up when none is given, 4 s per rep when there is no tempo, 60 s per transition) are written into the report as assumptions.

## Failure categories

Retrieval failures follow the seven failure points of Barnett et al. 2024 (course slide 36): `not_retrieved`, `missed_top_ranked`, `partial_recall`. Generation failures: `wrong_status`, `unsafe_release` (a plan released where a block, hold, clarification or abstention was expected), `schema_invalid`, `constraint_violation`, `incomplete_parameters`, `unsupported_claim`, `fabricated_citation`, `forbidden_output`, `missing_required`, `pipeline_error`.

## Release gates (project note, section 6)

1. 100 % of released plans pass every deterministic validator.
2. Zero unsafe releases in the red-team suite.
3. Adaptation traces and duplicate/correction tests pass. This gate shows **not evaluated** until an adaptation adapter exists.

These are engineering acceptance targets, not proof of safety.

## Not covered yet

- Adaptation trajectories (RT9–RT15) are specified in `evals/red_team/safety_cases.jsonl` but reported as "not run": they need the adaptation engine and a `step(state, event)` adapter.
- Level-2 grounding (entailment with an NLI model or an LLM judge). If added, the judge must first be calibrated against about 30 human labels.
- Baseline comparisons (prompt-only LLM, basic RAG, V1, V2): these only need one run per system with `--label`, then a comparison of the `results.json` files.
