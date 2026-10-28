"""Fast fixture tests for the three local corpus-ingestion steps."""

import csv
import json

import pandas as pd

from src.ingestion import build_corpus, chunker, coverage_report

VIDEO_ID = "abcdefghijk"


def test_build_corpus_writes_clean_passages_and_provenance(tmp_path, capsys):
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    transcript = transcripts / "Demo video.txt"
    transcript.write_text(
        "00:00:01 push\n00:00:02 pool legs [Musique]\n"
        "00:00:03 code promo 10% de réduction avec le lien partenaire\n"
        "00:00:04 encore une phrase assez longue pour continuer et créer une nouvelle ligne de texte\n",
        encoding="utf-8",
    )
    titles = tmp_path / "titles.txt"
    titles.write_text(f"https://youtu.be/{VIDEO_ID} : Demo video\n", encoding="utf-8")
    refs = tmp_path / "refs.txt"
    refs.write_text(
        f"https://youtu.be/{VIDEO_ID}\n\thttps://example.org/paper\n", encoding="utf-8"
    )
    corrections = tmp_path / "corrections.json"
    corrections.write_text(
        json.dumps(
            {
                "corrections": [
                    {"pattern": r"push pool legs", "replacement": "push pull legs"}
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    build_corpus.main(
        [
            "--corpus-dir",
            str(transcripts),
            "--titles",
            str(titles),
            "--refs",
            str(refs),
            "--corrections",
            str(corrections),
            "--out",
            str(out),
            "--corpus-version",
            "test-v1",
        ]
    )

    passages = [
        json.loads(line)
        for line in (out / "passages" / f"{VIDEO_ID}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert "push pull legs" in passages[0]["text"]
    assert "Musique" not in passages[0]["text"]
    assert any("sponsor" in passage["flags"] for passage in passages)
    assert passages[0]["passage_id"] == f"{VIDEO_ID}_p0001"
    assert passages[0]["url_t"].endswith("?t=2")

    with (out / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        manifest = next(csv.DictReader(stream))
    assert manifest["raw_sha256"]
    assert manifest["scientific_references"] == "https://example.org/paper"
    assert manifest["corpus_version"] == "test-v1"
    assert "videos: 1" in capsys.readouterr().out


def test_coverage_report_ignores_sponsors_and_writes_empty_hits(tmp_path):
    passages = tmp_path / "passages"
    passages.mkdir()
    (passages / f"{VIDEO_ID}.jsonl").write_text(
        json.dumps(
            {
                "video_id": VIDEO_ID,
                "start": 1,
                "url_t": "https://youtu.be/x?t=1",
                "text": "repos entre les séries",
                "flags": [],
            }
        )
        + "\n"
        + json.dumps(
            {
                "video_id": VIDEO_ID,
                "start": 2,
                "url_t": "https://youtu.be/x?t=2",
                "text": "repos entre les séries",
                "flags": ["sponsor"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with (tmp_path / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", "title"])
        writer.writerow([VIDEO_ID, "Demo video"])
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(
        json.dumps(
            {
                "topics": {
                    "rest": {"label": "Rest", "patterns": ["repos entre les séries"]},
                    "deload": {"label": "Deload", "patterns": ["deload"]},
                }
            }
        ),
        encoding="utf-8",
    )

    coverage_report.main(tmp_path, lexicon)

    with (tmp_path / "coverage_matrix.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        matrix = list(csv.DictReader(stream))
    assert matrix[0]["rest"] == "1"
    assert matrix[0]["deload"] == "0"
    with (tmp_path / "coverage_hits.csv").open(encoding="utf-8", newline="") as stream:
        hits = list(csv.DictReader(stream))
    assert len(hits) == 1
    assert "Rest | 1 | 1 | 0" in (tmp_path / "coverage_summary.md").read_text(
        encoding="utf-8"
    )


def test_chunker_groups_passages_and_preserves_metadata(tmp_path):
    passages = tmp_path / "passages"
    passages.mkdir()
    (passages / f"{VIDEO_ID}.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "video_id": VIDEO_ID,
                    "passage_id": f"{VIDEO_ID}_p{i:04}",
                    "start": (i - 1) * 30,
                    "end": i * 30,
                    "url_t": f"https://youtu.be/{VIDEO_ID}?t={(i - 1) * 30}",
                    "text": f"passage {i}",
                    "flags": [],
                }
            )
            for i in range(1, 4)
        )
        + "\n",
        encoding="utf-8",
    )
    with (tmp_path / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "video_id",
                "title",
                "scientific_references",
                "reference_review_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_id": VIDEO_ID,
                "title": "Demo",
                "scientific_references": "https://example.org/paper",
                "reference_review_status": "reviewed",
            }
        )
    with (tmp_path / "coverage_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "title", "rest"])
        writer.writeheader()
        writer.writerow({"video_id": VIDEO_ID, "title": "Demo", "rest": 2})

    written = chunker.build_chunks(tmp_path)

    assert written == [str(tmp_path / "chunks" / f"{VIDEO_ID}.jsonl")]
    chunks = [
        json.loads(line)
        for line in (tmp_path / "chunks" / f"{VIDEO_ID}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == f"{VIDEO_ID}_p0001-_p0003"
    assert chunks[0]["passage_ids"] == [
        f"{VIDEO_ID}_p0001",
        f"{VIDEO_ID}_p0002",
        f"{VIDEO_ID}_p0003",
    ]
    assert chunks[0]["topics"] == ["rest"]
    assert chunks[0]["scientific_references"] == ["https://example.org/paper"]
    assert chunks[0]["start_url"].endswith("?t=0")
