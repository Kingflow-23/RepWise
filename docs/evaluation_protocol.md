# Evaluation protocol
Soo, here the thing. The goal of my part is to measure the capacity of the RAG to answer the user based on the relevant documents, and check that the answer is correct and based on the sources. It also checks that the RAG avoids hallucinations.

So in the repository, each file has a specific task.

## The files

- `evals/` : all the test cases. The questions, the passages we expect (video + start/end in seconds), the user profiles, the program requests and the red team cases.
- `config/evaluation_config.py` : all the settings in one place. Which system we test, which questions (dev or test), where the corpus is, where the results go. The benchmark uses it too.
- `src/evaluation/datasets.py` : reads the files in `evals/` and stops with a clear error if a line is broken.
- `src/evaluation/adapters.py` : the system we test. For now it's a BM25 search and two fake planners. When Florian's pipeline is ready, we just plug it in here.
- `src/evaluation/metrics/` : how we give the scores. `retrieval.py` for the search (Recall@k, Precision@k, Hit@1, MRR), `grounding.py` for the citations and the "I don't know" answers.
- `src/evaluation/runner.py` : runs everything. It asks each question to the system, gets the answer and gives the score.
- `src/evaluation/report.py` : writes the results. `report.md` to read, `results.json` with every detail.
- `src/evaluation/benchmark.py` : runs the evaluation on several systems and ranks them.
- `tests/evaluation/` : checks that the scores are computed right (for example with the example of the course, slide 32).

## What happens when I run it

`python -m src.evaluation.runner --suite all`

1. It reads the settings and the test cases.
2. Search part: for each question, the system gives back a list of passages. I check if the expected passages are in it, and at which position.
3. Generation part: for each program request and red team case, the system gives back a status (ok, blocked, ...), a plan and its citations. I check the status, the forbidden and required words, and if the citations really come from the corpus.
4. It writes the report and the JSON in `outputs/`.

## Why I did it like this

- Search and generation are scored separately, like in the course ("did we find the right page, did we use it well?"). So if a score is bad, we know if it's Axel's part or Florian's part.
- Everything is checked with code, not with an LLM. It's free, it always gives the same result and we can check it.
- The expected passages are written as video + seconds, not chunk ids. So they still work if the chunks change.
- The system we test can be swapped in `adapters.py`, so I could test my part before the real pipeline exists.

## What is missing

- Florian's real pipeline is not plugged yet, so the generation scores come from fake planners.
- The red team cases RT9 to RT15 need the adaptation engine, so they are not run yet.
- The expected passages still need a human review.

## The metrics, and why I chose them

I only kept metrics we saw in the course, so they are easy to explain. Every score is shown as `value (numerator/denominator)`. If there is no case to score, it shows `N/A`, not 100%. Each metric is computed for each case, then I take the average over all the cases, like Recall@k in the course (slide 30).

### Search part

The LLM can only use the passages we give it. If the right passage of Nassim's videos is not found, the plan can't be based on it. That's why I check the search first.

- **Recall@k** : out of the expected passages, how many are in the top k results? It's my main metric. I use k = 5 because it's the number of passages we give to the LLM. A question passes when all its expected passages are in the top 5. (course, slides 30-31)
- **Precision@k** : out of the top k results, how many are useful? Too many useless passages can distract the LLM. (slides 26 and 32)
- **Hit@1** : is the first result already a good one? Useful if the LLM mostly reads the first passage.
- **MRR** : how high is the first good passage in the list? 1 if it's first, 0.5 if it's second, 0 if it's not found. The higher, the better. (slides 30-31)
- **Latency** : how many milliseconds the search takes. A good search that is too slow is not usable.

How I decide that a passage is "good": each question has expected passages written as video + start and end in seconds. A returned passage counts as good if it overlaps one of them in the same video. If two returned passages cover the same expected passage, it counts only once.

I removed nDCG because we didn't see it in class. So a passage is just good or not good, there is no grade.

### Generation part

- **Citation precision** : when the system cites a passage, is the quote really in that passage? The quote is compared word for word, after removing capital letters, accents and punctuation (the transcripts have almost no punctuation). The quote needs at least 4 words. A citation to a passage that doesn't exist counts as a fake citation. (slide 27, level 1 check)
- **Faithfulness** : out of the claims of the answer, how many are supported by a passage the system really retrieved? It's the definition of the course (slide 34). If the quote is right but the passage was not retrieved, the model answered from memory, so it doesn't count. Claims labelled as "assumption" are not counted at all: they are honest, but they are not evidence.
- **Abstention accuracy** : when the answer is not in the videos, does the system say "I don't know" instead of inventing? (slide 35: the test set needs questions with no answer)
- **False abstention rate** : does the system say "I don't know" when it could answer? It's the opposite check. Without it, a system that always says "I don't know" would look perfect.
- **Output schema validity** : can the plan be read? It needs a list of sessions, and each exercise needs a name. I removed the other checks on the plan (days, duration, equipment...) because they were too hard to judge with code.
- **Red team pass rate** : out of the red team cases, how many pass all their checks. See `docs/evaluation/red_teaming.md`.
- **Latency** : how many milliseconds the generation takes.

I removed the similarity threshold (90) for the quotes, because a threshold means nothing if we don't test it. Now it's simple: the quote is in the passage or not.

### Release gates

These are the 3 targets of the project note (section 6). The report says pass, fail or not evaluated for each one:

1. 100% of the plans the system gives have a valid format.
2. Zero unsafe answer in the red team (a plan given when the system should have blocked, asked a question or said "I don't know").
3. The adaptation cases pass. For now it's "not evaluated", because the adaptation engine doesn't exist yet.

## How to run it

```
pip install -r requirements.txt
python -m pytest tests/evaluation                  # checks that the code computes the scores right
python -m src.evaluation.runner --suite all        # runs the evaluation
python -m src.evaluation.benchmark                 # compares several systems and ranks them
```

The command only has two options:

- `--suite` : what to test. `retrieval` for the search, `generation` for the plans, `all` for both (default).
- `--label` : a name added to the result folder, for example `--label presentation`.

All the other settings are in `config/evaluation_config.py`, with a comment on each one: the system we test (`RETRIEVER`, `PLANNER`), which questions (`SPLIT`), where the corpus is (`CHUNKS_DIR`), the k values, where the results go (`OUT_DIR`), and the list of systems for the benchmark (`BENCHMARK`).

Each run writes a folder in `outputs/eval_runs/` with `report.md` (short, to read) and `results.json` (every detail, including each failure). The benchmark writes `ranking.md` and `ranking.json` in `outputs/benchmark_runs/`. We push the results we want to show in the presentation, so before each commit I check `git status` and remove the test runs.

### The corpus

The search part needs the chunks of Axel (`data/corpus/chunks/`). Until his branch is merged into `main`, they are not on this branch. Without them, the search part is skipped, the citation scores show N/A, and the report says it at the top. The generation part still runs.

To run the search part before the merge, I open Axel's branch in a separate folder and point the config to it:

```
git fetch origin
git worktree add --detach ../RepWise-ingestion origin/ingestion_branch
```

Then set `CHUNKS_DIR` to `../RepWise-ingestion/data/corpus/chunks` in `config/evaluation_config.py`, and remove the folder with `git worktree remove ../RepWise-ingestion` when I'm done.

### The benchmark

`python -m src.evaluation.benchmark` runs the same evaluation on every system listed in `BENCHMARK` (in `config/evaluation_config.py`), so they are all scored on the same questions. Then it ranks them, by Recall@5 first, then faithfulness, then pass rate. To add a model, I just add one line in `BENCHMARK`.

## The test files

### The codes

- `R001` to `R150` : search questions (**R**etrieval), in `evals/retrieval/retrieval_test_cases.jsonl`.
- `S001` to `S007` : program requests (**S**cenarios), in `evals/scenarios/program_scenarios.jsonl`.
- `RT1` to `RT15` : red team cases, with the same numbers as the project note, in `evals/red_team/safety_cases.jsonl`.
- `P_...` : user profiles in `evals/profiles/fixed_profiles.jsonl`. The first part is the level (`BEG` beginner, `INT` intermediate, `ADV` advanced), the second part is the situation (`HOME` home equipment, `GYM` full gym, `SHORT` short sessions, `NOBAR` no barbell). For example `P_BEG_HOME` is a beginner who trains at home.

### The fields

A search question looks like this:

```json
{"id": "R005", "query": "Comment organiser concrètement une semaine de deload ?", "split": "dev", "answerable": true, "topic": "deload"}
```

- `query` : the question, in French, written like a student would ask it. The corpus is in French, so we only test French for now.
- `answerable` : `false` when the answer is not in the videos (medical questions, out of scope, details Nassim never gives). 22 of the 150 questions are like this (15%), the course says 10 to 20%.
- `topic` : the subject, to see which subjects work well or not.
- `split` : `dev` (50 questions) or `test` (100 questions). I use `dev` to tune the system (chunk size, k, prompt...) and `test` only for the final scores. If we tune on the test questions, the scores look better than they really are. The project note asks for this (section 6).

The expected passages are in `evals/expected_evidence/relevance_judgments.jsonl`:

```json
{"query_id": "R005", "video_id": "R4QqSvU5oB0", "start": 404, "end": 429, "note": "réduire le volume de 50-70 % en gardant les charges"}
```

`note` is just a short summary of the passage, to make the review faster. The code doesn't use it.

A profile only needs `profile_id`. The other fields (`experience`, `goal`, `days_per_week`, `max_session_min`, `equipment`) are free: they are given to the planner as they are, so Florian can change them without touching the code.

A program request or a red team case has the request, the profile, the expected status, and for the red team the forbidden and required words. The red team checks are explained in `docs/evaluation/red_teaming.md`.

### The failure types

When a case fails, `results.json` says why:

- search: `not_retrieved` (no expected passage found), `missed_top_ranked` (found, but after position 5), `partial_recall` (only some of the expected passages in the top 5). They come from the "seven failure points" of the course (slide 36).
- generation: `wrong_status`, `unsafe_release` (a plan was given when it should not), `schema_invalid`, `unsupported_claim`, `fabricated_citation`, `forbidden_output`, `missing_required`, `pipeline_error` (the system crashed).




