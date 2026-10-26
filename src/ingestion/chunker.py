"""Step 3 of the RepWise data ingestion pipeline: group passages into chunks.

WHAT THIS SCRIPT DOES (in simple words)
    Passages (about 37 words) are too short to carry a full piece of advice,
    and whole videos are far too long to search. This script groups the
    passages of each video into chunks of about 250 words: the units that
    get embedded and retrieved.

    For each video it:
      1. Reads the cleaned passages (data/corpus/passages/<video_id>.jsonl).
      2. Removes the passages flagged 'sponsor' (sponsor reads are not
         training content). A removed sponsor block also cuts the video in
         two: a chunk never joins the text before and after an ad.
         Passages flagged 'cta' (calls to action) are KEPT: they usually
         also contain real content ("on passe maintenant a la selection
         d'exercices"), and the generation step drops any chunk flagged
         'cta', which would throw that content away.
      3. Groups consecutive passages until the next one would exceed
         MAX_WORDS, with OVERLAP_PASSAGES passages shared between two
         neighbouring chunks so an idea cut at a boundary appears whole in
         at least one chunk.
      4. Tags each chunk with the topics found in ITS OWN text (topic
         lexicon), and keeps the video-level topics in a separate field.
      5. Adds the video metadata: title, links, content type, speaker,
         scientific references.

WHY ~250 WORDS
    250 French words are roughly 350 to 400 tokens, which fits the 512-token
    limit of common multilingual embedding models (e5 family). Longer chunks
    would be silently truncated at embedding time, and their end never
    searched. Smaller chunks also make citations more precise.

INPUT FILES
    data/corpus/passages/<video_id>.jsonl   cleaned passages (build_corpus.py)
    data/corpus/manifest.csv                video metadata (build_corpus.py)
    data/corpus/coverage_matrix.csv         video-level topics (coverage_report.py)
    config/topic_lexicon.json               topic patterns, applied per chunk
    config/video_content_types.json         content type of each video

OUTPUT FILES
    data/corpus/chunks/<video_id>.jsonl     one chunk per line (gitignored:
                                            contains transcript text)

CHUNK FORMAT
    chunk_id                <video_id>_p<first>_p<last> (passage numbers)
    text                    passages joined with spaces
    word_count              number of words
    passage_ids             passages included, in order
    start, end              start / end time in seconds
    video_url, start_url    video link, and link at the chunk's start
    video_id, title
    content_type            advice, program, ranking, session_vlog,
                            coaching_case, third_party_analysis,
                            fat_loss_nutrition
    speaker                 who is speaking (mixed voices in vlogs)
    topics                  topics whose patterns match THIS chunk's text
    topic_counts            number of pattern matches per topic in the chunk
    video_topics            topics of the whole video (coverage_matrix)
    flags                   remaining passage flags (sponsor and cta are
                            never propagated)
    n_cta_passages          how many kept passages contained a call to action
    scientific_references   papers linked to the video
    reference_review_status 'mentioned' when papers exist
    corpus_version          build label from the manifest

USAGE (from the RepWise folder, after build_corpus.py and coverage_report.py)
    python src/ingestion/chunker.py
    python src/ingestion/chunker.py --max-words 300 --overlap 1
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent

MAX_WORDS = 250
OVERLAP_PASSAGES = 1
TOPIC_MIN_COUNT = 1
NON_TOPIC_COLS = {"video_id", "title"}
EXCLUDED_FLAGS = {"sponsor"}       # passages removed from chunks
NOT_PROPAGATED = {"sponsor", "cta"}  # passage flags not copied onto chunks
SPEAKER_BY_TYPE = {
    "advice": "single",
    "program": "single",
    "ranking": "single",
    "fat_loss_nutrition": "single",
    "session_vlog": "mixed",
    "coaching_case": "mixed",
    "third_party_analysis": "commentary",
}


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def read_passages(path: Path) -> list[dict]:
    """Read the passages of one video (one JSON object per line)."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_manifest(manifest_path: Path, coverage_path: Path):
    return pd.read_csv(manifest_path), pd.read_csv(coverage_path)


def load_topic_lexicon(path: Path) -> dict[str, re.Pattern]:
    topics = json.loads(path.read_text(encoding="utf-8"))["topics"]
    return {
        name: re.compile("|".join(f"(?:{p})" for p in spec["patterns"]), re.I)
        for name, spec in topics.items()
    }


def load_content_types(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["videos"], data["_about"]["speakers"]


def split_on_excluded(passages: list[dict]) -> list[list[dict]]:
    """Cut the video into runs of usable passages. Excluded passages
    (sponsor reads) are dropped and act as hard boundaries."""
    runs, current = [], []
    for p in passages:
        if EXCLUDED_FLAGS.intersection(p["flags"]):
            if current:
                runs.append(current)
            current = []
        else:
            current.append(p)
    if current:
        runs.append(current)
    return runs


def create_chunks(run: list[dict], max_words: int, overlap: int) -> list[list[dict]]:
    """Group consecutive passages into chunks of at most max_words words
    (a single longer passage forms its own chunk), sharing `overlap`
    passages between neighbours."""
    chunks, start = [], 0
    n = len(run)
    while start < n:
        end, total = start, 0
        while end < n:
            w = word_count(run[end]["text"])
            if end > start and total + w > max_words:
                break
            total += w
            end += 1
        chunks.append(run[start:end])
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def passage_number(passage_id: str) -> str:
    return re.search(r"_(p\d+)$", passage_id).group(1)


def build_records(video_id, passages, manifest_df, coverage_df, lexicon,
                  content_types, speakers, max_words, overlap) -> list[dict]:
    man = manifest_df.loc[manifest_df["video_id"] == video_id].iloc[0]
    cov = coverage_df.loc[coverage_df["video_id"] == video_id].iloc[0]
    topic_cols = [c for c in coverage_df.columns if c not in NON_TOPIC_COLS]
    video_topics = [t for t in topic_cols if int(cov[t]) >= TOPIC_MIN_COUNT]

    refs_raw = man["scientific_references"]
    references = [] if pd.isna(refs_raw) or not str(refs_raw).strip() else str(refs_raw).split("|")
    review = None if pd.isna(man["reference_review_status"]) else man["reference_review_status"]
    content_type = content_types.get(video_id, "unclassified")
    speaker = speakers[SPEAKER_BY_TYPE.get(content_type, "single")]
    video_url = str(man["url"])

    records = []
    for run in split_on_excluded(passages):
        for group in create_chunks(run, max_words, overlap):
            text = " ".join(p["text"] for p in group)
            counts = {t: len(rx.findall(text)) for t, rx in lexicon.items()}
            flags = sorted({f for p in group for f in p["flags"]} - NOT_PROPAGATED)
            start, end = int(group[0]["start"]), int(group[-1]["end"])
            records.append({
                "chunk_id": f"{video_id}_{passage_number(group[0]['passage_id'])}"
                            f"_{passage_number(group[-1]['passage_id'])}",
                "text": text,
                "word_count": word_count(text),
                "passage_ids": [p["passage_id"] for p in group],
                "start": start,
                "end": end,
                "video_url": video_url,
                "start_url": f"{video_url}?t={start}",
                "video_id": video_id,
                "title": man["title"],
                "content_type": content_type,
                "speaker": speaker,
                "topics": [t for t, c in counts.items() if c >= TOPIC_MIN_COUNT],
                "topic_counts": {t: c for t, c in counts.items() if c},
                "video_topics": video_topics,
                "flags": flags,
                "n_cta_passages": sum("cta" in p["flags"] for p in group),
                "scientific_references": references,
                "reference_review_status": review,
                "corpus_version": man["corpus_version"],
            })
    return records


def save_chunks(records: list[dict], out_dir: Path, video_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{video_id}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-words", type=int, default=MAX_WORDS)
    ap.add_argument("--overlap", type=int, default=OVERLAP_PASSAGES)
    args = ap.parse_args()

    corpus = REPO / "data" / "corpus"
    manifest_df, coverage_df = read_manifest(corpus / "manifest.csv", corpus / "coverage_matrix.csv")
    lexicon = load_topic_lexicon(REPO / "config" / "topic_lexicon.json")
    content_types, speakers = load_content_types(REPO / "config" / "video_content_types.json")

    missing = sorted(set(manifest_df["video_id"]) - set(content_types))
    all_records = []
    for video_file in sorted((corpus / "passages").glob("*.jsonl")):
        video_id = video_file.stem
        records = build_records(video_id, read_passages(video_file), manifest_df, coverage_df,
                                lexicon, content_types, speakers, args.max_words, args.overlap)
        if records:
            save_chunks(records, corpus / "chunks", video_id)
            all_records.extend(records)

    words = sorted(r["word_count"] for r in all_records)
    print(f"videos: {len(set(r['video_id'] for r in all_records))}  chunks: {len(all_records)}  "
          f"words per chunk: median {words[len(words) // 2]}, max {words[-1]}")
    print(f"chunks with no topic of their own: {sum(1 for r in all_records if not r['topics'])}  "
          f"chunks still flagged: {sum(1 for r in all_records if r['flags'])}")
    for vid in missing:
        print(f"WARN no content type for {vid} (set it in config/video_content_types.json)")


if __name__ == "__main__":
    main()
