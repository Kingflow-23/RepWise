# RepWise knowledge base: what it can ground, and what it can't (yet)

**Corpus version:** v1-2026-10-10 (60 Nassim Sahili videos, 5,952 passages)
**Written:** 2026-10-25, from a direct search of the cleaned passages (not only the coverage counts)

---

## 1. Why this document exists

An LLM can write a gym program for any constraint you give it: 2 days a week, 20 minutes, dumbbells only. That is easy, and that is "just GPT work". RepWise's value is different: **every prescription must be traceable to a source**, and anything the sources don't support must be labelled as an assumption or withheld.

So the real question is not "can the LLM generate it?" but **"does the knowledge base contain the evidence to validate it?"** This document answers that question, element by element, so we know:

- what the system can cite today;
- what it can only support with a general principle (and must complete with a labelled assumption);
- what it cannot ground at all, and therefore what we cannot honestly test or claim for the moment.

### Mentions vs prescriptions

The coverage matrix (`data/corpus/coverage_matrix.csv`) shows that the 60 videos **mention** every topic a program needs. That is true, and a good sign. But a mention is not a usable prescription. To validate a program, a passage must say something you can apply: a number, a range, a rule or a choice. Examples:

| The passage... | Mention? | Usable prescription? |
|---|---|---|
| "le deload, c'est pas un truc de faible" | yes | no |
| "un cycle de deload peut être plus ou moins long que 7 jours" | yes | yes (duration, roughly) |
| "la progression, c'est important" | yes | no |
| "10 répétitions à 80 kg, trois mois après 10 répétitions à 95 kg" | yes | partly (illustrates progression, gives no rule) |

This document grades each element by what the passages actually **prescribe**, not by whether the topic is mentioned.

### The grading scale

| Level | Meaning | What RepWise must do |
|---|---|---|
| **Grounded** | The corpus gives explicit values or rules (numbers, ranges, worked examples) | Cite the passage |
| **Partial** | The corpus gives a principle or a direction, but no exact value or trigger | Cite the principle, label the exact value as an implementation assumption |
| **Not grounded** | Nothing usable in the corpus | Label as assumption, ask, or abstain. Never attribute it to a source. |

---

## 2. Short answer: can RepWise generate a schedule under constraints?

**Yes, within a clear perimeter.** The corpus contains complete, worked programs (exercises, sets, reps, rest, weekly layout) for these situations:

| Situation | Evidence |
|---|---|
| Beginner, 3 sessions/week, full body, about 1 hour | Two alternating full body sessions, 6 exercises, 10 minutes each ([05WPpiDdC78 @137s](https://youtu.be/05WPpiDdC78?t=137), [@159s](https://youtu.be/05WPpiDdC78?t=159)) |
| Intermediate/advanced, 3 days, "pec-dos / épaules-bras / jambes" split | Full sessions with sets and rep ranges ([XFsbtGB1TXQ @169s](https://youtu.be/XFsbtGB1TXQ?t=169), [@325s](https://youtu.be/XFsbtGB1TXQ?t=325)) |
| 4 sessions/week of 90 minutes, health goal (full body strength + cardio) | Full program, with sets, reps, rest, cardio blocks ([Jd6jgL2s2F4 @531s](https://youtu.be/Jd6jgL2s2F4?t=531), [@654s](https://youtu.be/Jd6jgL2s2F4?t=654)) |
| 5 days/week, sessions rotated in a cycle | Principle: 3 sessions rotated over 5 days, each muscle trained several times per week ([45l6bl8Uyr4 @521s](https://youtu.be/45l6bl8Uyr4?t=521)) |

**Outside that perimeter, the LLM would be on its own.** The main blind spots:

- sessions shorter than about 45 minutes;
- 2 days or 6 days a week (mentioned, never programmed);
- home training, bodyweight only, or minimal equipment;
- effort measured in RIR or RPE;
- exact progression steps (how many kg, after how many sessions);
- what to do after a missed session;
- specific populations (women-specific programming, older adults, adolescents, people with medical conditions).

The rest of this document details each element.

---

## 3. Program design: element by element

| Program element | Level | What the corpus provides | Key evidence |
|---|---|---|---|
| **Weekly frequency** | Grounded for 3, 4 and 5 days; Partial for 2 and 6 | 3 regular sessions beat 5 occasional ones. Worked programs exist for 3, 4 and 5 days. 2 days: only "2 sessions maintain, 3 develop", **said by a guest quoting his coach**, not by Nassim. 6 days: only a survey result (advanced lifters training 4+ times prefer push pull legs). | [05WP @104s](https://youtu.be/05WPpiDdC78?t=104), [-yB8 @807s](https://youtu.be/-yB8TWvU83g?t=807) |
| **Split choice** | Grounded | Full body for beginners; his favourite 3-day agonist/antagonist split; push pull legs for advanced lifters training 4+ times a week; rotating sessions for 5 days. | [XFsb @169s](https://youtu.be/XFsbtGB1TXQ?t=169), [45l6 @521s](https://youtu.be/45l6bl8Uyr4?t=521) |
| **Session duration** | Grounded for about 60 to 90 min only | Beginner sessions about 1 hour; his own high-volume sessions under 1 hour; the hybrid program caps sessions at 90 minutes. Below 45 min: only a historical mention (Mentzer did 30 to 45 min). | [05WP @137s](https://youtu.be/05WPpiDdC78?t=137), [Jd6j @552s](https://youtu.be/Jd6jgL2s2F4?t=552), [1OsD @179s](https://youtu.be/1OsDiVvQT7k?t=179) |
| **Exercise selection** | Grounded for main muscle groups | 6 ranking videos ("du MEILLEUR au PIRE": biceps, triceps, back, chest, quadriceps, shoulders) plus weak point videos (arms, quadriceps, upper chest, rear delts, traps). Repeated principle: favour exercises with high tension in the stretched position. No dedicated ranking for hamstrings, glutes or calves. | ranking videos, [05WP @137s](https://youtu.be/05WPpiDdC78?t=137) |
| **Exercise order** | Partial | Scattered principles: heavy compound exercise with low reps first; on leg day, start with isolation to warm up the knees. No general ordering rule. | [XFsb @325s](https://youtu.be/XFsbtGB1TXQ?t=325) |
| **Sets and rep ranges** | Grounded | Worked values, e.g. 4 × 6-8 on the first compound exercise, 4 × ~12 afterwards, 4 × 10-12 leg curl and stiff-leg deadlift, 5 × 15-20 calves; hybrid program 3 × 15, 4 × 8, 4 × 10. Rule: don't keep the same scheme (e.g. 4 × 10) for 8 weeks. | [XFsb @325s](https://youtu.be/XFsbtGB1TXQ?t=325), [Jd6j @684s](https://youtu.be/Jd6jgL2s2F4?t=684), [45l6 @690s](https://youtu.be/45l6bl8Uyr4?t=690) |
| **Rest between sets** | Grounded, with a conflict | Dedicated video: about 2 to 2.5 min for hypertrophy, 4 min or more for strength. **But** his own programs use 45 to 90 seconds (in a multi-goal context). Both are in the corpus. | [qcs8 @138s](https://youtu.be/qcs8utwv9WE?t=138), [qcs8 @168s](https://youtu.be/qcs8utwv9WE?t=168), [Jd6j @684s](https://youtu.be/Jd6jgL2s2F4?t=684) |
| **Tempo** | Partial | Tempo notation used in a few programs (e.g. 1-0-1-0, 2-0-1-1), and the rule that tempo should change across a mesocycle. Most exercises have no tempo. | [7--w @599s](https://youtu.be/7--wCDtdGHM?t=599), [7--w @626s](https://youtu.be/7--wCDtdGHM?t=626), [45l6 @706s](https://youtu.be/45l6bl8Uyr4?t=706) |
| **Effort target** | Partial (failure-based); Not grounded (RIR/RPE) | Effort is described as training to failure: an experienced lifter "can reach failure on any exercise". **Zero passages give RIR or RPE values.** | [05WP @484s](https://youtu.be/05WPpiDdC78?t=484), "vrai échec" video |
| **Weekly volume** | Grounded | Categories cited from studies: low volume under 12 sets per muscle per week, moderate 12 to 20, high above 20. Start low and increase according to results. Split a muscle's weekly sets over at least 2 sessions. A cycle can end at double or triple its starting volume. | [1OsD @231s](https://youtu.be/1OsDiVvQT7k?t=231), [HHpE @354s](https://youtu.be/HHpEn-HlCkM?t=354), [HHpE @511s](https://youtu.be/HHpEn-HlCkM?t=511), [1OsD @666s](https://youtu.be/1OsDiVvQT7k?t=666) |
| **Cycles / periodization** | Grounded | 6-month macrocycle split into three 8-week mesocycles, each with one goal (muscle, strength, fat loss). Keep the exercises within a mesocycle and vary the other variables. Example of intensity blocks: 55%, then 70%, then 90% over 3-week blocks. Example from a study: 4 weeks of rising volume, then a deload week. | [45l6 @325s](https://youtu.be/45l6bl8Uyr4?t=325), [45l6 @690s](https://youtu.be/45l6bl8Uyr4?t=690), [HHpE @488s](https://youtu.be/HHpEn-HlCkM?t=488), [1OsD @289s](https://youtu.be/1OsDiVvQT7k?t=289) |
| **Starting load** | Partial | Load chosen by rep target ("8RM", "12RM"), "a load that challenges you a little", and one rule of thumb (walking lunges at 25% of bodyweight). No calibration protocol for an unknown load. | [05WP @159s](https://youtu.be/05WPpiDdC78?t=159), [Jd6j @654s](https://youtu.be/Jd6jgL2s2F4?t=654) |
| **Warm-up** | Not grounded | Only incidental mentions in session vlogs, and "4 minutes of warm-up" in the cardio part of the hybrid program. No warm-up protocol for strength sessions. The `T_warmup` term of the duration formula is an assumption. | none usable |
| **Transition time between exercises** | Not grounded | Never discussed. `T_transitions` is an assumption. | none |

---

## 4. User constraints: what can be validated

| Constraint value | Level | Notes |
|---|---|---|
| **Days: 3** | Grounded | Two worked programs (beginner full body, 3-day split) |
| **Days: 4** | Grounded, with a caveat | The worked 4-day program targets health (strength + cardio), not pure hypertrophy |
| **Days: 5** | Partial | Principle of rotating sessions, no complete 5-day hypertrophy program |
| **Days: 2 or 6** | Partial | Mentioned, never programmed (see section 3) |
| **Duration: 60 to 90 min** | Grounded | |
| **Duration: 45 min** | Partial | No program at this length; only by trimming a longer one (assumption) |
| **Duration: under 45 min** | Not grounded | Red team case RT3 (45 → 20 min) cannot be grounded: only the validator and a labelled trade-off can handle it |
| **Equipment: commercial gym** (barbells, dumbbells, machines, cables) | Grounded | Covered by the ranking videos |
| **Equipment: limited gym** (a machine is missing) | Partial | 7 explicit "si vous n'avez pas X, faites Y" passages; rankings give alternatives for the same muscle |
| **Equipment: home with dumbbells** | Partial | Dumbbell exercises exist, but no home program |
| **Equipment: bodyweight only / bands** | Not grounded | Bands are mentioned negatively ([dctY @181s](https://youtu.be/dctYv3j4o3U?t=181)) |
| **Experience: beginner** | Grounded | Dedicated videos (levels, "créer son programme débutants") |
| **Experience: intermediate / advanced** | Grounded | Levels video, splits, periodization |
| **Goal: hypertrophy** | Grounded | The core of the corpus |
| **Goal: strength** | Partial | Rest and cycle principles, no complete strength program |
| **Goal: fat loss** | Out of MVP scope | Well covered (cardio, mini cut), but nutrition-heavy |
| **Population: women, older adults, adolescents, medical conditions** | Not grounded | Never addressed specifically |

---

## 5. Adaptation engine: what each rule can rest on

These are the five adaptation dimensions of the V2 note, plus holds and safety.

| Adaptation | Level | What the corpus supports | What will be an implementation assumption |
|---|---|---|---|
| **Load / rep progression** | Partial | Progressive overload is central. Priority order: load first, then reps, then sets, then rest and tempo ([8Upe @366s](https://youtu.be/8Upe8QgpYps?t=366)). Adjust the load to stay inside the target rep range ([cvfT @515s](https://youtu.be/cvfTTuOZKV8?t=515)). Keep the same scheme (e.g. 4 × 8) at the start and end of a cycle, flexible in between ([cvfT @542s](https://youtu.be/cvfTTuOZKV8?t=542)). | Increment size (kg), number of successful exposures before increasing, caps |
| **Volume change** | Partial | Start low, increase according to results; volume categories; cycles of rising volume ([HHpE @511s](https://youtu.be/HHpEn-HlCkM?t=511), [1OsD @289s](https://youtu.be/1OsDiVvQT7k?t=289)) | Triggers and size of a set reduction after underperformance or fatigue |
| **Exercise substitution** | Partial | Rankings by muscle and 7 explicit "if you don't have X" alternatives | The compatibility map itself: built by the team, each entry linked to a ranking passage |
| **Deload** | Partial | One dedicated video ([R4Qq @17s](https://youtu.be/R4QqSvU5oB0?t=17)): reduce intensity and effort for about a week (length can vary), useful after intensive periods, he likes 2 weeks of overreaching before it ([R4Qq @193s](https://youtu.be/R4QqSvU5oB0?t=193)), several methods including a full stop ([R4Qq @360s](https://youtu.be/R4QqSvU5oB0?t=360)). **Essentially one source.** | Trigger pattern, % reduction of load or volume, how to return |
| **Progression policy switch** (after repeated failure) | Not grounded | Stagnation is mentioned but never turned into a method | The whole rule |
| **Missed sessions** | Not grounded | Only "consistency matters" | The whole rule (e.g. "never double the next session") |
| **Break / return after time off** | Partial | Pause and deload video | How to resume (loads after a break) |
| **Pain** | Not grounded, by design | Only general advice (good technique reduces injury risk) | Safety handling is a policy, never a corpus claim. RepWise must not diagnose or treat. |

Bottom line for the adaptation engine: the **direction** of most rules is grounded, but almost all **thresholds** (how many sessions, how many kg, how many sets, which pattern triggers a deload) will be implementation assumptions. That is acceptable, and the V2 note already plans for it. It just has to be labelled, frozen before evaluation, and reported honestly.

---

## 6. Structural gaps in the knowledge base itself

These are not missing topics, but properties of the data that limit what we can claim.

### 6.1 Chunk topics are copied from the video, not from the chunk

`chunker.py` gives every chunk the full topic list of its video (all topics with at least 1 hit in `coverage_matrix.csv`). Result: **every chunk of a video carries the same topics** (median 6 topics per chunk, up to 12). A chunk tagged `deload` may not say a word about deload.

Consequence: topic metadata can't be used to filter or evaluate retrieval at chunk level, and can't be used as proof that a chunk supports a rule. Fix: run the topic lexicon on each chunk's own text and keep the video-level topics as a separate field.

### 6.2 Not every chunk is Nassim's advice

All chunks have `speaker: "Nassim Sahili"`. But part of the corpus is speech by someone else, or content he is criticizing:

- session vlogs with guests (two voices mixed, no speaker labels);
- analysis videos of other people's programs (Bryan Johnson, a student, viewers' technique);
- programs he summarizes before disagreeing (Huberman's 6-day program);
- videos where he and the guest openly disagree ("on n'est pas d'accord").

Consequence: retrieval can return a passage that **describes or criticizes** a practice as if it were a recommendation. The citation would be valid, while the support is false. This directly affects Citation Precision.

### 6.3 No upload dates

`upload_date` is empty. Nassim's positions evolve (the volume video is literally titled "On s'est tous trompés...", and the rest video updates "1 to 1.5 min" to "2 to 2.5 min"). Without dates we can't tell which position is the more recent, so conflicting evidence (red team case RT5) can be shown, but not resolved by recency.

### 6.4 Scientific references are linked, not reviewed

Only 6 of the 60 videos have linked papers (22 in total), all with status `mentioned`. No claim can be described as "scientifically validated" yet.

### 6.5 Text quality

48 of 60 transcripts have no punctuation, speech recognition errors remain beyond the 29 corrected patterns, and sponsor reads are only partly flagged (the chunker keeps flagged passages). See `data/corpus/README.md`.

### 6.6 Off-topic share

A large part of the corpus is about fat loss, cardio and nutrition (24 videos mention nutrition topics). These passages can surface for training questions and dilute retrieval.

---

## 7. What we can and cannot test for the moment

### Red team scenarios

| Case | Testable now? | Why |
|---|---|---|
| RT1 Beginner mismatch | Yes | Beginner level and program are grounded |
| RT2 Unsupported detail | Yes, and the gaps make good test cases | Ask for RIR targets, kg increments, warm-up protocol, a 20-minute program: the correct behavior is "assumption" or abstention |
| RT3 Time conflict 45 → 20 min | Validator: yes. Grounding: no | No short-session program in the corpus |
| RT4 Pain or injury | Yes | Pure safety policy, no grounding needed |
| RT5 Conflicting evidence | Partly | Real conflicts exist (rest times: 2 to 2.5 min in the dedicated video vs 45 to 90 s in his programs), but can't be resolved by date |
| RT6 Missing information | Yes | Engine behavior |
| RT7 Prompt injection | Yes | Engine behavior |
| RT8 Impossible program | Yes | Validator behavior |
| RT9 Duplicate / corrected log | Yes | Engine behavior, no corpus needed |
| RT10 Missing or noisy sessions | Engine: yes. Grounding: no | Missed-session rules are assumptions |
| RT11 Repeated fatigue | Partly | Deload grounded by one video, thresholds are assumptions |
| RT12 Exercise substitution | Yes for gym equipment, no for home/bodyweight | |
| RT13 Units and policy drift | Yes | Engine behavior |
| RT14 Stale or foreign state | Yes | Engine behavior |
| RT15 Evidence outage | Yes | Engine behavior |

### Metrics

| Metric | Testable now? | Note |
|---|---|---|
| Recall@K, MRR, nDCG | Yes, once labelled queries exist | Don't use chunk topics as relevance labels (6.1) |
| Citation Precision | Yes | Will be lowered by non-Nassim passages (6.2) unless they are handled |
| Grounded Claim Rate | Yes | **Can't reach 100% by design**: thresholds are assumptions. Report assumptions separately, as the V2 note says. |
| Abstention accuracy | Yes | The gaps in this document are natural abstention cases |
| Hard Constraint Adherence, Program Validity, Duration overrun | Yes | Validator metrics, independent of the corpus |
| Adaptation agreement | Yes, with human labels | Reviewers judge against rules that are partly assumptions |

### What we must not claim

- That RepWise programs are validated for home training, short sessions, 2 or 6 days a week, or specific populations.
- That effort targets (RIR/RPE) or progression thresholds come from the sources.
- That any claim is scientifically validated (references are `mentioned` only).
- That the system produces better training results (the V2 note already excludes this).

---

## 8. Recommended next steps (in priority order)

1. **Freeze the perimeter for the presentation:** commercial gym, 3 to 5 days, 60 to 90 min, hypertrophy goal, healthy adults, beginner to advanced. Say it explicitly on a slide.
2. **Write the assumption register:** one line per ungrounded value (RIR targets, kg increments, exposures before progression, deload trigger and %, missed-session rule, policy switch, warm-up, transitions), each labelled `implementation_assumption`, frozen before evaluation.
3. **Tag topics at chunk level** (6.1) before building retrieval labels.
4. **Mark non-advice content** (6.2): at least a `content_type` per video (advice, program, ranking, session vlog, analysis of someone else, nutrition), so retrieval and grading can treat them differently.
5. **Fill `upload_date`** for the 60 videos (one manual pass on YouTube is enough) to make RT5 resolvable.
6. **Build the substitution map** from the 6 ranking videos, each entry linked to its passage.
7. **Use the gaps as test cases:** every "Not grounded" row above is a ready-made RT2 or abstention test.
