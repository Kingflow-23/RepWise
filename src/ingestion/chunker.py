"""Build retrieval-ready chunks from cleaned, timestamped video passages."""

import re
import json
import logging
from pathlib import Path
import pandas as pd

from ..body_parts import detect_body_parts
from ..observability import configure_logging

MAX_WORD_CHUNK = 450
REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
TOPIC_MIN_COUNT = 1
NON_TOPIC_COLS = {"video_id", "title"}
logger = logging.getLogger(__name__)


def read_json_file(text_json_path: str):
    """Read one video's line-delimited passage records into a DataFrame.

    Args:
        text_json_path: Path to the video's JSONL passage file.

    Returns:
        A pandas DataFrame containing the decoded records.

    Raises:
        FileNotFoundError: If the input path does not exist.
        ValueError: If pandas cannot decode the JSONL records.
    """
    json_object = pd.read_json(text_json_path, lines=True)
    return json_object


def read_manifest(manifest_path: str, coverage_path: str):
    """Read the video manifest and topic coverage tables.

    Args:
        manifest_path: CSV path containing video metadata and references.
        coverage_path: CSV path containing per-video topic counts.

    Returns:
        ``(manifest_dataframe, coverage_dataframe)`` in that order.

    Raises:
        FileNotFoundError: If either CSV is missing.
        pandas.errors.ParserError: If either CSV is malformed.
    """
    manifest_dataframe, coverage_dataframe = pd.read_csv(manifest_path), pd.read_csv(
        coverage_path
    )
    return manifest_dataframe, coverage_dataframe


def word_count(text: str):
    """Count regex word tokens in a passage.

    Args:
        text: Passage text to count.

    Returns:
        Number of ``\\w+`` matches, rather than whitespace-delimited tokens.
    """
    return len(re.findall(r"\w+", text))


def _is_unflagged(flags):
    """Return whether a flags cell is empty or missing.

    Args:
        flags: A decoded collection, null-like value, or CSV cell value.

    Returns:
        ``True`` for an empty collection, missing value, or blank string;
        otherwise ``False``.
    """
    if isinstance(flags, (list, tuple, set)):
        return not flags
    if flags is None or pd.isna(flags):
        return True
    return not str(flags).strip()


def create_chunk(text_json_path: str):
    """Merge a video's passages into overlapping chunks under the word cap.

    Flagged passages are excluded when a ``flags`` column is present. Chunk
    identifiers derive from the video and passage range, making them stable
    while that source range remains unchanged.

    Args:
        text_json_path: Path to one video's JSONL passage file.

    Returns:
        An insertion-ordered mapping from internal keys to chunk records. Each
        record contains text, word count, source passage IDs, times, URL, and
        video ID; adjacent chunks overlap by one passage.

    Raises:
        FileNotFoundError: If the passage file is missing.
        ValueError: If pandas cannot decode the passage records.
        AttributeError: If a passage ID or URL does not match the expected
            format.
    """
    json_object = read_json_file(text_json_path)
    if "flags" in json_object:
        json_object = json_object.loc[json_object["flags"].map(_is_unflagged)]
        json_object = json_object.reset_index(drop=True)

    n_rows = len(json_object)
    start_index = 0
    chunk_id = 1
    chunk_dict = {}

    MAX_WORD_CHUNK = 450
    OVERLAP_PASSAGES = 1

    while start_index < n_rows:
        row_index = start_index
        word_total = 0
        texts, passage_ids = [], []

        start_time = json_object.iloc[row_index]["start"]
        start_passage_id = re.search(
            r"_p(\d+)$", json_object.iloc[row_index]["passage_id"]
        ).group()

        while row_index < n_rows:
            row = json_object.iloc[row_index]
            n_words = word_count(row["text"])

            if texts and word_total + n_words > MAX_WORD_CHUNK:
                break

            texts.append(row["text"])
            passage_ids.append(row["passage_id"])
            word_total += n_words
            row_index += 1

        end_time = json_object.iloc[row_index - 1]["end"]
        end_time_passage_id = re.search(
            r"_p(\d+)$", json_object.iloc[row_index - 1]["passage_id"]
        ).group()
        url = json_object.iloc[row_index - 1]["url_t"]
        video_url = re.match(r"[^?]+", url).group()
        video_id = json_object.iloc[row_index - 1]["video_id"]
        chunk_ID = (
            video_id + f"_{str(start_passage_id)}" + f"_{str(end_time_passage_id)}"
        )

        chunk_dict[f"chunk_{chunk_ID}"] = {
            # Stable while the source video and passage range stay unchanged.
            "chunk_id": f"{video_id}{start_passage_id}-{end_time_passage_id}",
            "text": " ".join(texts),
            "word_count": word_total,
            "passage_ids": passage_ids,
            "start": start_time,
            "end": end_time,
            "video_url": video_url,
            "video_id": video_id,
        }
        chunk_id += 1

        if row_index >= n_rows:
            break

        start_index = max(row_index - OVERLAP_PASSAGES, start_index + 1)
    return chunk_dict


def final_metadata_match(
    chunk_dict,
    coverage_df,
    manifest_df,
    topic_min_count=TOPIC_MIN_COUNT,
    speaker="Nassim Sahili",
):
    """Attach topic, source, citation, and timing metadata to every chunk.

    Topics are included when their passage counts meet ``topic_min_count``.
    Scientific references are split from the manifest's pipe-delimited value;
    body parts are detected from each chunk's text.

    Args:
        chunk_dict: Non-empty chunk mapping returned by :func:`create_chunk`.
        coverage_df: Coverage matrix with one row per video and topic columns.
        manifest_df: Manifest with matching video metadata and references.
        topic_min_count: Minimum coverage count required to attach a topic.
        speaker: Speaker label stored on each output chunk.

    Returns:
        Flat chunk records enriched with citation and retrieval metadata.

    Raises:
        StopIteration: If ``chunk_dict`` is empty.
        IndexError: If the video has no matching row in either input table.
        KeyError: If a required metadata column is absent.
    """

    video_id = next(iter(chunk_dict.values()))["video_id"]

    cov = coverage_df.loc[coverage_df["video_id"] == video_id]
    man = manifest_df.loc[manifest_df["video_id"] == video_id]

    cov_row, man_row = cov.iloc[0], man.iloc[0]

    topic_cols = [c for c in coverage_df.columns if c not in NON_TOPIC_COLS]
    topic_counts = {t: int(cov_row[t]) for t in topic_cols}
    topics = [t for t, n in topic_counts.items() if n >= topic_min_count]

    refs_raw = man_row["scientific_references"]
    references = (
        []
        if pd.isna(refs_raw) or not str(refs_raw).strip()
        else str(refs_raw).split("|")
    )

    review_status = man_row["reference_review_status"]
    review_status = None if pd.isna(review_status) else review_status

    records = []
    for chunk in chunk_dict.values():
        records.append(
            {
                **chunk,
                "start": int(chunk["start"]),
                "end": int(chunk["end"]),
                "start_url": f'{chunk["video_url"]}?t={int(chunk["start"])}',
                "title": man_row["title"],
                "speaker": speaker,
                "topics": topics,
                "topic_counts": topic_counts,
                "body_parts": detect_body_parts(chunk["text"]),
                "scientific_references": references,
                "reference_review_status": review_status,
            }
        )
    return records


def save_chunk(records: list, out_dir: str | Path | None = None):
    """Save flattened chunk records into data/corpus/chunks as JSONL.

    Each record is a dict produced by final_metadata_match() and contains all the
    metadata needed for retrieval and citation.

    Args:
        records: Flat records for a single video. An empty list is a no-op.
        out_dir: Destination directory; defaults to the repository's chunks
            directory.

    Returns:
        A one-element list containing the written path, or an empty list when
        there are no records.

    Raises:
        OSError: If the output directory or file cannot be created or written.
        KeyError: If a non-empty record set lacks ``video_id``.
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
    """Build chunk files from the repository's cleaned corpus.

    Returns:
        Paths of the chunk JSONL files written.
    """
    return build_chunks(REPO / "data" / "corpus")


def build_chunks(corpus_dir: str | Path):
    """Build and enrich chunk files for each passage file in a corpus.

    Empty videos have stale chunk files removed. Files are processed in sorted
    filename order for reproducible outputs.

    Args:
        corpus_dir: Directory containing ``manifest.csv``,
            ``coverage_matrix.csv``, and ``passages``.

    Returns:
        Paths of chunk JSONL files written for non-empty videos.

    Raises:
        FileNotFoundError: If required corpus files or passages are unavailable.
        KeyError: If a passage or metadata record lacks a required field.
        IndexError: If a video's manifest or coverage row is absent.
        OSError: If stale files cannot be removed or output files cannot be
            written.
    """
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir / "manifest.csv"
    coverage_path = corpus_dir / "coverage_matrix.csv"
    passages_dir = corpus_dir / "passages"

    manifest_df, coverage_df = read_manifest(str(manifest_path), str(coverage_path))

    chunk_files = []
    logger.info("Chunk building started corpus_dir=%s", corpus_dir)
    for video_file in sorted(passages_dir.glob("*.jsonl")):
        video_id = video_file.stem
        chunk_dict = create_chunk(str(video_file))
        if not chunk_dict:
            stale_chunk_file = corpus_dir / "chunks" / f"{video_id}.jsonl"
            if stale_chunk_file.exists():
                stale_chunk_file.unlink()
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
        logger.info(
            "Video chunks built video_id=%s chunks=%s",
            video_id,
            len(records),
        )

    logger.info("Chunk building completed video_files=%s", len(chunk_files))
    print(f"saved {len(chunk_files)} chunk files")
    return chunk_files


if __name__ == "__main__":
    configure_logging("ingestion")
    main()
