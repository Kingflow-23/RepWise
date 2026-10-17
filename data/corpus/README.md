# RepWise corpus

**Source:** 60 French-language YouTube videos by Nassim Sahili (gym training education). The transcripts were extracted from YouTube captions on 2026-10-10. Raw files live in `../../../corpus/` and are never modified.

**Usage:** academic prototype only (aivancity PGE5 GenAI project). The transcripts remain the creator's content, and the corpus is not redistributed.

## Build

From the RepWise folder:

```
python scripts/build_corpus.py      # raw transcripts -> cleaned passages + manifest
python scripts/coverage_report.py   # topic coverage map
```

Both scripts use the Python standard library only. Rerun both after editing `config/term_corrections.json` or `config/topic_lexicon.json`.

## Pipeline

1. **Parse.** Handles both caption layouts in the raw files (rolling captions with empty duplicate timestamps, and flat captions), plus continuation lines that have no timestamp.
2. **Clean.** Normalizes text to NFC, unifies apostrophes and strips sound tags (`[Musique]`, `[Rires]`, `[ __ ]` and similar).
3. **Correct.** Applies the ASR correction map across the whole transcript, so errors split across two caption fragments are also fixed. Every hit is logged in `corrections_log.csv`.
4. **Passage.** Merges the 1 to 2 second caption fragments into passages. Punctuated files are split on sentence ends. Unpunctuated files are split into windows of about 30 words.
5. **Flag.** Marks passages with sponsor reads (`sponsor`) and calls to action (`cta`), so chunking can exclude them. Detection is keyword-based: a sponsor read can span neighbouring passages that carry no flag.

## Files

| File | Content |
|---|---|
| `manifest.csv` | One row per video: id, url, title, raw file hash, duration, caption layout, punctuation, counts, sponsor timestamps, linked papers, reference review status, corpus version |
| `passages/<video_id>.jsonl` | `passage_id`, `start`, `end` (seconds), `url_t` (timestamped link), `text`, `flags` |
| `clean_txt/<video_id>.txt` | Human-readable version for review |
| `corrections_log.csv` | Number of times each correction fired, per video |
| `coverage_matrix.csv` / `coverage_hits.csv` / `coverage_summary.md` | Topic coverage. A hit is a lead to read, not evidence |

## Known limitations

- `upload_date` is empty: the raw files don't contain it.
- 48 of the 60 transcripts have no punctuation, so their passages are word windows rather than sentences.
- Linked scientific references exist for 6 of the 60 videos. Their status is `mentioned`, not reviewed.
