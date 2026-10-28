"""Map which videos and passages cover each adaptation topic.

Reads data/corpus/passages/*.jsonl (built by build_corpus.py) and the lexicon in
config/topic_lexicon.json. A hit means a passage mentions the topic and is
worth reading; it is NOT proof that the passage supports a rule.

Outputs (under data/corpus/):
  coverage_matrix.csv   video x topic passage counts
  coverage_hits.csv     every hit with timestamp link, for manual review
  coverage_summary.md   per-topic totals and the strongest videos
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data" / "corpus"


def main(corpus: Path = CORPUS, lexicon_path: Path | None = None) -> None:
    corpus = Path(corpus)
    lexicon_path = lexicon_path or REPO / "config" / "topic_lexicon.json"
    lex = json.loads(lexicon_path.read_text(encoding="utf-8"))["topics"]
    rx = {
        t: re.compile("|".join(f"(?:{p})" for p in v["patterns"]), re.I)
        for t, v in lex.items()
    }
    titles = {}
    with open(corpus / "manifest.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            titles[row["video_id"]] = row["title"]

    matrix = {vid: defaultdict(int) for vid in titles}
    hits = []
    for vid in titles:
        with open(corpus / "passages" / f"{vid}.jsonl", encoding="utf-8") as stream:
            for line in stream:
                p = json.loads(line)
                if "sponsor" in p["flags"]:
                    continue
                for topic, r in rx.items():
                    m = r.search(p["text"])
                    if m:
                        matrix[vid][topic] += 1
                        hits.append(
                            {
                                "topic": topic,
                                "video_id": vid,
                                "title": titles[vid],
                                "start": p["start"],
                                "url_t": p["url_t"],
                                "match": m.group(0),
                                "text": p["text"],
                            }
                        )

    topics = list(lex)
    with open(corpus / "coverage_matrix.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "title", *topics])
        for vid, counts in matrix.items():
            w.writerow([vid, titles[vid], *[counts[t] for t in topics]])
    hit_fields = ["topic", "video_id", "title", "start", "url_t", "match", "text"]
    with open(corpus / "coverage_hits.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hit_fields)
        w.writeheader()
        w.writerows(hits)

    lines = [
        "# Corpus topic coverage",
        "",
        "Passage-level keyword hits (sponsor passages excluded). A hit is a lead to read, not evidence.",
        "",
        "| Topic | Passages | Videos with >=1 hit | Videos with >=5 hits | Top videos (hits) |",
        "|---|---|---|---|---|",
    ]
    for t in topics:
        per_vid = sorted(
            ((matrix[v][t], v) for v in titles if matrix[v][t]), reverse=True
        )
        total = sum(c for c, _ in per_vid)
        strong = sum(1 for c, _ in per_vid if c >= 5)
        top = "; ".join(f"{titles[v][:45]} ({c})" for c, v in per_vid[:3])
        lines.append(
            f"| {lex[t]['label']} | {total} | {len(per_vid)} | {strong} | {top} |"
        )
    (corpus / "coverage_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
