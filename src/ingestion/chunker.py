import re
import json
from pathlib import Path
import pandas as pd

MAX_WORD_CHUNK = 450
REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
TOPIC_MIN_COUNT = 1
NON_TOPIC_COLS = {"video_id", "title"}

def read_json_file(text_json_path : str):
    """
    Read Json file of one video
    """
    json_object = pd.read_json(text_json_path, lines=True)
    return json_object

def read_manifest(manifest_path : str, coverage_path : str):
    manifest_dataframe, coverage_dataframe = pd.read_csv(manifest_path), pd.read_csv(coverage_path)
    return manifest_dataframe, coverage_dataframe


def word_count(text : str):
    return len(re.findall(r'\w+', text))

def create_chunk(text_json_path : str):
    json_object = read_json_file(text_json_path)

    n_rows = len(json_object)
    start_index = 0
    chunk_id = 1
    chunk_dict = {}

    MAX_WORD_CHUNK = 450
    OVERLAP_PASSAGES = 1

    while start_index < n_rows:
        row_index = start_index
        word_total = 0
        texts, flags, passage_ids = [], set(), []

        start_time = json_object.iloc[row_index]["start"]
        start_passage_id = re.search(r"_p(\d+)$", json_object.iloc[row_index]["passage_id"]).group()

        while row_index < n_rows:
            row = json_object.iloc[row_index]
            n_words = word_count(row["text"])

            if texts and word_total + n_words > MAX_WORD_CHUNK:
                break

            texts.append(row["text"])
            flags.update(row["flags"])
            passage_ids.append(row["passage_id"])
            word_total += n_words
            row_index += 1

        end_time = json_object.iloc[row_index - 1]["end"]
        end_time_passage_id = re.search(r"_p(\d+)$", json_object.iloc[row_index - 1]["passage_id"]).group()
        url = json_object.iloc[row_index - 1]["url_t"]
        video_url = re.match(r"[^?]+", url).group()
        video_id = json_object.iloc[row_index - 1]["video_id"]
        chunk_ID = video_id + f"_{str(start_passage_id)}" + f"_{str(end_time_passage_id)}"
        

        
        chunk_dict[f"chunk_{chunk_ID}"] = {
            "text": " ".join(texts),
            "flags": sorted(flags),
            "word_count": word_total,
            "passage_ids": passage_ids,
            "start": start_time,
            "end": end_time,
            "video_url": video_url,
            "video_id": video_id
        }
        chunk_id += 1

        if row_index >= n_rows:  
            break

        start_index = max(row_index - OVERLAP_PASSAGES, start_index + 1)
    return chunk_dict


def final_metadata_match(chunk_dict, coverage_df, manifest_df,
                         topic_min_count=TOPIC_MIN_COUNT,
                         speaker="Nassim Sahili"):
    """Attach video-level metadata to every chunk of one video.
    Returns a list of flat chunk records."""

    video_id = next(iter(chunk_dict.values()))["video_id"]

    cov = coverage_df.loc[coverage_df["video_id"] == video_id]
    man = manifest_df.loc[manifest_df["video_id"] == video_id]

    cov_row, man_row = cov.iloc[0], man.iloc[0]

    topic_cols = [c for c in coverage_df.columns if c not in NON_TOPIC_COLS]
    topic_counts = {t: int(cov_row[t]) for t in topic_cols}
    topics = [t for t, n in topic_counts.items() if n >= topic_min_count]

    refs_raw = man_row["scientific_references"]
    references = [] if pd.isna(refs_raw) or not str(refs_raw).strip() \
        else str(refs_raw).split("|")

    review_status = man_row["reference_review_status"]
    review_status = None if pd.isna(review_status) else review_status


    records = []
    for chunk in chunk_dict.values():
        records.append({
            **chunk,
            "start": int(chunk["start"]),
            "end": int(chunk["end"]),
            "start_url": f'{chunk["video_url"]}?t={int(chunk["start"])}',
            "title": man_row["title"],
            "speaker": speaker,
            "topics": topics,
            "topic_counts": topic_counts,
            "scientific_references": references,
            "reference_review_status": review_status,
        })
    return records


def save_chunk(records: list, out_dir: str | Path | None = None):
    """Save flattened chunk records into data/corpus/chunks as JSONL.

    Each record is a dict produced by final_metadata_match() and contains all the
    metadata needed for retrieval and citation.
    """
    if out_dir is None:
        out_dir = REPO / "data" / "corpus" / "chunks"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not records:
        return []

    video_id = records[0]["video_id"]
    output_path = out_dir / f"{video_id}.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return [str(output_path)]


def main():
    """Build chunk files for all videos in the cleaned corpus."""
    corpus_dir = REPO / "data" / "corpus"
    manifest_path = corpus_dir / "manifest.csv"
    coverage_path = corpus_dir / "coverage_matrix.csv"
    passages_dir = corpus_dir / "passages"

    manifest_df, coverage_df = read_manifest(str(manifest_path), str(coverage_path))

    chunk_files = []
    for video_file in sorted(passages_dir.glob("*.jsonl")):
        video_id = video_file.stem
        chunk_dict = create_chunk(str(video_file))
        if not chunk_dict:
            continue

        records = final_metadata_match(
            chunk_dict,
            coverage_df,
            manifest_df,
            topic_min_count=TOPIC_MIN_COUNT,
            speaker="Nassim Sahili",
        )
        saved = save_chunk(records, corpus_dir / "chunks")
        chunk_files.extend(saved)

    print(f"saved {len(chunk_files)} chunk files")
    return chunk_files


if __name__ == "__main__":
    main()