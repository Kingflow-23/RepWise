# RepWise corpus: how the raw transcripts are cleaned

This folder holds the cleaned version of the RepWise corpus: 60 French YouTube videos by Nassim Sahili about gym training. This document explains, step by step and in plain words, what `src/ingestion/build_corpus.py` does to the raw transcripts, why each step exists, and what it does **not** do.

| | |
|---|---|
| Source | 60 YouTube videos, Nassim Sahili, French |
| Raw material | YouTube captions (mostly automatic speech recognition), extracted 2026-10-10 |
| Volume | 16.5 hours of video, 29,828 caption fragments, 210,919 words |
| Result | 5,952 cleaned passages (about 37 words each), 501 speech recognition errors corrected |
| Use | Academic prototype only (aivancity PGE5 GenAI project). The transcripts remain the creator's content and are not redistributed. |

---

## 1. The problem: what raw captions look like

A raw transcript file looks like this (rolling layout):

```
00:01 On s'est tous déjà posé cette question
00:02 

00:02 au moins une fois. Quel est le meilleur
00:04 
```

Many issues make this text unusable as it is:

| Problem | Example | Why it matters |
|---|---|---|
| Text arrives in tiny 1 to 2 second pieces | `au moins une fois. Quel est le meilleur` | A piece is not a full idea. Searching on it gives meaningless results. |
| Two different file layouts | "rolling" files have an empty duplicate timestamp after every line (46 files). "Flat" files have one line per timestamp (14 files). | The reader must understand both. |
| Lines without a timestamp | a line that just continues the previous one | Must be attached to the right piece, not lost. |
| Sound tags in the text | `[Musique]`, `[Rires]`, `[Applaudissements]`, `[ __ ]` (censored word) | Noise inside sentences and citations. |
| No punctuation | 48 of the 60 files are one long lowercase stream | Sentences can't be detected. |
| Misheard gym vocabulary | `dilode` for deload, `push pool legs` for push pull legs, `axquat` for hack squat, `mass grass` for masse grasse | A search for "deload" would find nothing, even though one video is entirely about it. |
| Odd timestamps | `24:60` instead of `25:00` (453 times, in 59 files) | Must still give the right time. |
| Sponsor reads and calls to action | "code promo", "abonnez-vous" | Off-topic text that should not be used as training advice. |
| Inconsistent characters | curly apostrophes `’` vs straight `'`, non-breaking spaces, accents encoded two different ways (é as one character or as e + accent) | The same word looks like two different words to a computer. File names have the same issue. |

---

## 2. Design principles

These rules guided every choice in the cleaning:

1. **The raw files are never modified.** The script only reads them and writes new files. If something goes wrong, you can always start over from the original.
2. **Never invent content.** Cleaning removes noise and fixes known errors. It never adds words, guesses missing text or rewrites what the speaker meant.
3. **Every passage stays traceable to the video.** Each passage keeps its start and end time and a link that opens the video at that exact second. Anyone can check a claim against the source in a few seconds.
4. **Every correction is counted.** Each fix is logged per video, so you can see what was changed and how often.
5. **Same input, same output.** The script is deterministic: run it twice on the same files and you get identical results. The fingerprint of each raw file (SHA-256) is stored, so anyone can check that they are working on the same corpus.
6. **Flag, don't delete.** Sponsor reads and calls to action are marked, not removed. The decision to exclude them belongs to the next steps.
7. **No dependencies.** The script uses only the Python standard library, so it runs anywhere without installing anything.

---

## 3. The cleaning pipeline, step by step

```
raw .txt file
   │
   ├─ 0. Match the file to its video (id, title, papers)
   ├─ 1. Read the captions          -> list of fragments (time + text)
   ├─ 2. Clean each fragment        -> no sound tags, uniform characters
   ├─ 3. Fix speech errors          -> correction map, on the whole transcript
   ├─ 4. Build passages             -> groups of fragments with start/end time
   ├─ 5. Flag sponsor / CTA         -> labels on passages
   └─ 6. Write outputs              -> passages, readable text, manifest, log
```

### Step 0. Match each file to its video

The transcript files are named after the video title, and the video id only exists in `dictionnary_links.txt`. To link them, the script:

- reads `dictionnary_links.txt` (one `https://youtu.be/<id> : <title>` per line);
- builds a simplified key from each title and each file name: lowercase, accents normalized, everything except letters and digits removed. Example: `Quel TEMPS DE REPOS prendre pour la PRISE DE MUSCLE ?` and the file name `Quel TEMPS DE REPOS prendre pour la PRISE DE MUSCLE.txt` give the same key, even though Windows removed the `?` from the file name;
- if no exact key matches, it accepts a near match (at least 85% similar) and prints a warning;
- prints a warning for any video without a file, and for any file not in the dictionary.

It also reads `links.txt` to attach the scientific papers linked to each video (the tab-indented URLs under each video link). Current result: all 60 files matched exactly, and 6 videos have linked papers (22 papers in total).

### Step 1. Read the captions into fragments

Each line is read with one rule: a timestamp (`mm:ss` or `h:mm:ss`) followed by text.

- **Timestamps become seconds:** hours × 3600 + minutes × 60 + seconds. This arithmetic also handles the odd timestamps: `24:60` gives 1500 seconds, which is exactly 25:00.
- **Empty timestamp lines are skipped.** In rolling files, every line is followed by an empty duplicate timestamp. These carry no text.
- **Lines without a timestamp** are glued to the end of the previous fragment, so no text is lost.
- **Text is normalized as it is read:** accents are stored the same way everywhere (Unicode NFC form), curly apostrophes `’` become straight `'`, and non-breaking spaces become normal spaces.
- **The layout is detected:** if empty timestamp lines outnumber half of the text lines, the file is labelled `rolling`, otherwise `flat`. This label is stored in the manifest.

Output of this step: a list of fragments, each with a start time in seconds and a short piece of text.

### Step 2. Clean each fragment

- **Sound tags are removed:** anything between square brackets (`[Musique]`, `[musique]`, `[Rires]`, `[Applaudissements]`, `[soupir]`, `[ __ ]`...). These are descriptions of sounds, not speech.
- **Spaces are tidied:** multiple spaces collapse into one, and spaces at the start and end are removed.

### Step 3. Fix speech recognition errors

The automatic captions get gym vocabulary wrong in consistent ways. The fixes are listed in `config/term_corrections.json`. Each entry has:

| Field | Meaning |
|---|---|
| `pattern` | what to look for (a regular expression, case-insensitive, whole words only) |
| `replacement` | what to write instead |
| `status` | `confirmed` (checked in context) or `needs_review` (likely, still to check against the video) |
| `example` | a real phrase where the error appears |

Current map: 29 rules. Main fixes and how often they fired:

| Heard | Corrected to | Count |
|---|---|---|
| mass grass / masse grass | masse grasse | 156 |
| alters / altères | haltères | 57 |
| soulever terre / soulev de terre | soulevé de terre | 48 |
| développer couché / incliné... | développé couché / incliné... | 43 |
| dorseaux | dorsaux | 25 |
| dilode / dilo / diload / dit laude | deload | 23 |
| axquat / axwat | hack squat | 12 |
| push pool legs / push poule legs | push pull legs | 13 |
| face pool / facepool | face pull | 11 |

Two rules are still `needs_review`: `legl` (probably leg curl) and `drakker` (probably drag curl).

**The key design choice: corrections run on the whole transcript, not fragment by fragment.** Errors are often cut across two caption pieces. In this real case, the word "push" ends one fragment and "pool legs" starts the next one:

```
...sans surprise, le push       <- fragment at 00:59
pool legs, c'est le format...   <- fragment at 01:02
```

Checked fragment by fragment, "push pool legs" would never be found. So the script:

1. joins all fragments of the video into one text, remembering where each fragment starts;
2. applies every correction rule, in the order of the file, on that full text;
3. recomputes where each fragment starts in the corrected text (corrections can make the text longer or shorter);
4. if a correction spans two fragments, the corrected words go to the fragment where the error started.

Each rule's hits are counted per video and saved in `corrections_log.csv`.

### Step 4. Build passages

Fragments are too short to be useful, so they are grouped into **passages** of a few sentences. A passage is the smallest unit of cleaned text. It's what you read, cite and later group into chunks.

First, the script decides if the file is **punctuated**: it is if more than 20% of its fragments contain `.`, `?`, `!` or `,`. Then:

| File type | When a passage is closed |
|---|---|
| Punctuated (12 files) | At the end of a sentence (`.`, `?`, `!` or `…`), as soon as the passage has at least 8 words, or at 60 words in any case. Passages follow real sentences. |
| Not punctuated (48 files) | Every 30 words or so, always at a fragment boundary. Passages are windows of text, since sentences can't be detected. |

While grouping, empty fragments and fragments identical to the previous one are skipped.

**Times:** each passage starts at the start time of its first fragment. It ends at the start time of the fragment that comes right after it (or 2 seconds after the last fragment, for the end of the video). A passage always lasts at least 1 second.

### Step 5. Flag sponsor reads and calls to action

Each passage is checked against two lists of expressions (case-insensitive):

| Flag | Triggered by | Count |
|---|---|---|
| `sponsor` | "sponsoris(é/e)", "code promo", "mon code", "premier lien", "partenaire de la/cette vidéo", "X % de réduction", "-X % avec/sur" | 70 passages, in 32 videos |
| `cta` (call to action) | "abonnez-vous", "abonne-toi", "lâchez/mettez un like", "activez la cloche", "en/dans les commentaires", "lien en/dans la description" | 125 passages |

A passage can have both flags. Flagged passages are kept, with their flags, so the next steps decide what to do with them. The coverage report already skips `sponsor` passages.

### Step 6. Write the outputs

For each video the script writes the passages (JSON Lines) and a human-readable text file, then a manifest row and its correction counts.

---

## 4. Output files

| File | What it contains | In git? |
|---|---|---|
| `passages/<video_id>.jsonl` | The cleaned passages, one per line. The input of every next step. | No (transcript text) |
| `clean_txt/<video_id>.txt` | The same passages in a readable form, to review by eye | No (transcript text) |
| `manifest.csv` | One row per video: metadata and provenance | Yes |
| `corrections_log.csv` | Number of times each correction fired, per video | Yes |

### Passage format (`passages/<video_id>.jsonl`)

```json
{"video_id": "XFsbtGB1TXQ", "passage_id": "XFsbtGB1TXQ_p0001", "start": 1, "end": 14,
 "url_t": "https://youtu.be/XFsbtGB1TXQ?t=1", "text": "On s'est tous déjà posé cette question ...", "flags": []}
```

| Field | Meaning |
|---|---|
| `video_id` | YouTube id of the video |
| `passage_id` | `<video_id>_p<number>`, numbered from 0001 in reading order |
| `start`, `end` | start and end time in seconds |
| `url_t` | link that opens the video at the start of the passage |
| `text` | cleaned and corrected text |
| `flags` | `sponsor`, `cta`, both, or empty |

### Readable format (`clean_txt/<video_id>.txt`)

A header with the title and the link, then one passage per line with its time and any flags:

```
# Ce format d'entraînement est parfait (Programme complet)
# https://youtu.be/XFsbtGB1TXQ

[00:01] On s'est tous déjà posé cette question au moins une fois. ...
[01:20] {sponsor} ...
```

### Manifest columns (`manifest.csv`)

| Column | Meaning |
|---|---|
| `video_id`, `url`, `title` | identity of the video |
| `source_file` | name of the raw transcript file |
| `raw_sha256` | fingerprint of the raw file. If yours differs, you don't have the same corpus. |
| `upload_date` | empty for now (not in the raw files) |
| `duration_s` | start time of the last fragment, in seconds |
| `caption_layout` | `rolling` or `flat` |
| `punctuated` | `True` if the transcript has punctuation |
| `n_passages`, `n_words` | size after cleaning |
| `n_corrections` | number of speech errors fixed in this video |
| `sponsor_passages` | start times of the passages flagged `sponsor` |
| `scientific_references` | papers linked to the video, separated by `\|` |
| `n_references` | number of linked papers |
| `reference_review_status` | `mentioned` when papers exist: they are linked, not yet checked against the claims |
| `corpus_version` | label of this build (default `v1-<date of the run>`) |

---

## 5. Inputs and how to run it

| Input | Location (from the RepWise folder) | Shared through |
|---|---|---|
| Raw transcripts | `../corpus/*.txt` | private zip, never git |
| Video id ↔ title | `../dictionnary_links.txt` | private zip |
| Video ↔ papers | `../links.txt` | private zip |
| Correction map | `config/term_corrections.json` | git |

To set up, unzip the shared archive **next to** the RepWise folder, so that `corpus/`, `dictionnary_links.txt` and `links.txt` sit beside `RepWise/`. Then, from the RepWise folder:

```bash
python src/ingestion/build_corpus.py        # step 1: clean the transcripts (this document)
python src/ingestion/coverage_report.py     # step 2: map which videos cover which topic
python src/ingestion/chunker.py             # step 3: group passages into chunks for retrieval
```

Run them in this order, since each step reads the previous one's output. Other options are listed by `python src/ingestion/build_corpus.py --help`: other input or output paths, and a custom `--corpus-version` label.

**Adding a correction:** open `config/term_corrections.json`, add an entry with a pattern, its replacement, a status and an example, then rerun all three steps. Check `corrections_log.csv` to see how many times the new rule fired.

---

## 6. What the cleaning does NOT do (known limitations)

Being clear about this matters for the evaluation and the presentation.

- **No punctuation is added.** The 48 unpunctuated files stay lowercase without punctuation. Their passages are word windows, so a passage can start or end mid-sentence.
- **Filler words are kept:** "euh", "bah", "ben", "hein", "ouais" and the like.
- **Repeated words are kept:** stutters such as "le le" or "ça ça", and rolling captions that repeat the end of the previous line, both stay in the text.
- **Speakers are not separated.** In training session videos with a guest, both voices are mixed in the same text with no label for who is speaking.
- **Silences don't split passages.** A passage can join speech from before and after a long pause (music, an exercise set).
- **Sponsor detection is partial.** Only passages containing a trigger expression are flagged. The rest of a sponsor read, in the neighbouring passages, can remain unflagged.
- **The correction map only fixes known errors.** Other misheard words certainly remain. The map grows as new errors are found.
- **Timestamps are precise to the second only,** as in the raw captions.
- **`upload_date` is empty,** and the linked papers have not been checked yet (`mentioned` status).
- **Sponsor detection limits carry over to chunks:** `chunker.py` removes every passage flagged `sponsor` (and never joins the text before and after one), but the unflagged parts of a sponsor read stay in the chunks. Passages flagged `cta` are kept on purpose, since they usually also contain real content.
