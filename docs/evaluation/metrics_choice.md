# Why these metrics

This document explains why each evaluation metric was chosen, what question it answers, and why some common alternatives were left out. Formulas and dataset formats are in [`../evaluation_protocol.md`](../evaluation_protocol.md).

## Guiding principles

1. **One metric answers one question about one component.** The course separates two questions: did we find the right passage (retrieval), and did the model use it correctly (generation)? The project adds a third: is the plan executable and safe? Each layer is therefore scored separately, so that a bad score tells us *who* has to fix *what*.
2. **Deterministic first, LLM judges last.** Everything that can be checked by code (rank of a passage, quote present in a chunk, session duration, equipment) is checked by code. It is free, reproducible and cannot be fooled. LLM judges are only added where code cannot decide, and only after calibration on human labels.
3. **Numerator and denominator are always reported.** "100 %" on 2 cases and on 200 cases are not the same result. A zero denominator is "N/A", not a perfect score (project note, section 5).
4. **The metrics come from the project note.** The V2 note fixes the metric families (sections 5 and 6). The choices below implement them; they do not replace them.
5. **Measure one change at a time.** Each benchmark run changes one setting (embedder, k, LLM…). The metrics must therefore be stable and cheap enough to rerun after every change.

## Retrieval layer

| Metric | Question it answers | Why it was chosen |
|---|---|---|
| **Recall@k** (main: k = 5) | Is the evidence among the chunks given to the LLM? | The key RAG metric: the generator cannot use a passage it never sees (course, slides 30–31). k = 5 matches the number of chunks that fit in the prompt with ~450-word chunks. |
| **MRR** | How high is the first relevant chunk? | Rewards putting the evidence first. LLMs use the start and end of the context best ("lost in the middle", slide 26). |
| **nDCG@k** | Is the *most* useful evidence ranked highest? | Our labels are graded (1 = related, 3 = directly answers). Recall and MRR ignore grades; nDCG uses them. Required by the project note. |
| **Hit@1** | Is the very first result right? | Simple and intuitive; useful if the generator ends up reading only the top chunk. |
| **Precision@k** | How much noise reaches the prompt? | Irrelevant chunks distract the model (slide 26). Reported as a diagnostic, not a target: with 1–3 relevant spans per question, its maximum is low by construction. |
| **Latency (ms/query)** | Is it fast enough? | The lab report asks for ms/query next to quality, because rerankers and query rewriting cost time. |

**Why relevance is labelled as time spans, not chunk IDs.** Axel's chunking can change (size, overlap). If labels pointed to chunk IDs, every re-chunking would invalidate them. A label "video X, from 404 s to 429 s" stays true whatever the chunking. It also lets us *compare* chunking strategies with the same test set. A span covered by two overlapping chunks counts once, so overlap cannot inflate the score.

**Why some questions have no answer (20 %).** They measure whether the system says "not found" instead of inventing (slide 35). They are excluded from retrieval scores, which need evidence to exist, and scored in the generation layer as abstention.

**Left out:**
- *MAP (mean average precision):* it duplicates what Recall and nDCG already show, and is harder to explain.
- *Retrieval scored by an LLM judge (e.g. RAGAS "context precision"):* labelled spans let us score retrieval without any model, which is cheaper and fully reproducible.

## Grounding layer

| Metric | Question it answers | Why it was chosen |
|---|---|---|
| **Citation precision** | When the system cites a chunk, is the citation real? | Required by the note. A valid link alone does not count: the quoted words must appear in the cited chunk (quote check, slide 27). |
| **Unsupported claim rate** ↓ | How many factual or prescriptive claims have no support? | The main hallucination signal. "Faithful" is what the generator controls (slides 33–34). |
| **Grounded claim rate** | How many claims are supported by the *retrieved* context? | Separates "supported by the corpus" from "supported by what the model was actually given". A claim supported by a chunk that was not retrieved means the model answered from memory and was lucky. |
| **Fabricated citations** (count) | Does the system invent sources? | The worst failure for a system whose value is traceability. Tracked on its own so it is never averaged away. |
| **Abstention accuracy** | When the corpus has no answer, does it abstain? | Required by the note (V1 definition). Covers RT2 (unsupported detail) and out-of-scope requests. |
| **False abstention rate** ↓ | Does it refuse questions it could answer? | The counterweight. A system that always says "not found" would score 100 % on abstention accuracy; this metric catches that. |

**Why a quote check instead of an LLM judge.** The course gives three levels of citation checking: quote match, entailment and coverage (slide 27). Level 1 is deterministic, cheap and, according to the course, very effective. It requires the generator to return a short verbatim quote with each citation, which is part of the interface contract with Florian. Matching ignores case, accents and punctuation, because 48 of the 60 transcripts are unpunctuated speech-to-text. The 90/100 threshold comes from the course example.

**"Assumption" claims are counted apart.** The note says an honest label ("implementation assumption") is not evidence. Such claims are neither penalised nor counted as grounded.

**Left out:**
- *BLEU, ROUGE, BERTScore:* they compare the answer to one reference text. A training plan has many valid forms; similarity to one reference says nothing about correctness or support.
- *LLM-judge faithfulness as the only measure:* LLM judges can be biased (for example, favouring their own model family) and are not reproducible. It is planned as level 2, calibrated against about 30 human labels first (slides 34 and 43).

## Program layer

| Metric | Question it answers | Why it was chosen |
|---|---|---|
| **Output schema validity** | Can the application read the output at all? | A plan that cannot be parsed cannot be executed. Covers "wrong format", failure point 5 in slide 36. |
| **Hard constraint adherence** | Are days, session time, equipment, exclusions and bounds respected? | The note makes hard constraints mandatory. Each one is checked by code, per session and per exercise, so the failure report says exactly which constraint broke. |
| **Plans with zero violations** | How many plans are fully usable? | Adherence averages over many small checks: 95 % can hide that every plan has one problem. This metric counts whole plans. |
| **Preference satisfaction** | Are soft preferences honoured? | Required by the note. Kept separate from hard constraints, because preferences may be traded off but constraints may not. |
| **Parameter completeness** | Is every exercise executable (sets, reps, rest, effort)? | Required by the note. A placeholder such as "TBD" does not count as filled. |
| **Duration overrun** ↓ | By how much does a session exceed the time budget? | The note replaces the V1 absolute difference: a shorter session is valid and must not be penalised. Computed with the note's formula, and the defaults used are written in the report. |
| **Program validity rate** | How many released plans pass every validator? | The release gate: it must be 100 %. Strict on purpose: a single violation makes the plan invalid. |

## Safety and red team

| Metric | Why it was chosen |
|---|---|
| **Red-team pass rate** | One number for the 15 scenarios of the note (RT1–RT15). A case passes only if *every* check passes: expected status, no forbidden output, constraints respected. |
| **Unsafe releases** (count, gate = 0) | Releasing a plan where the system should have blocked, held, asked or abstained (for example pain reported, RT4). This is the "zero blocked increases" gate of the note. Counted separately because one case is already too many. |

**Why "expected status" instead of reading the text.** Whether the system blocked, asked a question or abstained is decided by the `status` field, not by parsing its prose. Regular-expression checks on the text are used only for forbidden content (diagnosis, attempts to reveal other users' data) and required content (referral to a professional).

## Adaptation layer (to be done)

The longitudinal metrics of the note (adaptation agreement, unsafe increase rate, longitudinal consistency, decision traceability, replay integrity) need the adaptation engine. The trajectories RT9–RT15 are already written; they are reported as "not run" rather than silently skipped or counted as passed.

## Known limits

- **Labels are drafts.** They must be reviewed by a second person before the test split is frozen.
- **150 queries, 128 with evidence.** One query now moves Recall by less than 1 point, which is enough to compare embedders. The `test` split has 100 queries and `dev` has 50, so a difference of a few points on `test` is still within noise.
- **Corpus coverage is uneven.** The transcripts say little about missed sessions and returning after a break, so those topics have only one or two queries. This is a gap in the corpus, not in the test set, and it matters for the adaptation rules of the project note.
- **Validators check rules, not truth.** They confirm that encoded constraints are respected. They cannot prove that the advice is correct or safe for a given person (note, section 7).
- **Pattern checks are coarse.** A forbidden-word regex can miss a paraphrase or flag a negation. They are a safety net, not a judge.
