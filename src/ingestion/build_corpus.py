"""Step 1 of the RepWise data ingestion pipeline: clean the raw transcripts.

WHAT THIS SCRIPT DOES 
    YouTube captions are messy: the text comes in tiny 1-2 second pieces,
    sentences are cut in half, most files have no punctuation, sound tags
    like [Musique] appear in the middle of sentences, and the speech
    recognition gets gym words wrong ("dilode" instead of "deload",
    "push pool legs" instead of "push pull legs").

    This script takes each raw transcript and turns it into clean,
    timestamped passages that the rest of the pipeline can use. It never
    modifies the raw files: it reads them and writes new files.

    For each video it:
      1. Reads the raw captions. Two caption layouts exist in the corpus
         (rolling captions with empty duplicate timestamps, and flat
         captions), plus lines that continue the previous one without a
         timestamp. All are handled.
      2. Cleans the text: removes sound tags ([Musique], [Rires], ...),
         normalises accents and apostrophes, removes duplicates.
      3. Fixes known speech recognition errors using the correction map.
         Corrections run on the whole transcript, so an error split across
         two caption pieces ("push" | "pool legs") is still fixed.
      4. Groups the tiny caption pieces into passages of a few sentences
         (about 30 words when the file has no punctuation). Each passage
         keeps its start and end time in seconds and a link that opens the
         video at that moment.
      5. Flags passages that are sponsor reads ("sponsor") or calls to
         action ("cta") so later steps can leave them out.
      6. Writes one manifest row per video with its provenance: url, title,
         raw file hash, duration, caption quality, linked scientific papers.

INPUT FILES
    ../corpus/*.txt                 raw transcripts, one per video
                                    (outside the repo, never committed)
    ../dictionnary_links.txt        "video url : title" for each video,
                                    used to match each file to its video id
    ../links.txt                    video urls with the scientific papers
                                    linked to each video (tab-indented)
    config/term_corrections.json    speech recognition correction map
                                    (edit it when we find new errors)

    Paths are relative to the RepWise folder. The raw inputs are shared
    privately (zip), not through git.

OUTPUT FILES (in data/corpus/)
    manifest.csv                one row per video: metadata and provenance
                                (committed: contains no transcript text)
    corrections_log.csv         how many times each correction fired, per
                                video (committed)
    passages/<video_id>.jsonl   cleaned, timestamped passages: the input of
                                the next steps (gitignored: transcript text)
    clean_txt/<video_id>.txt    human-readable version of the passages, for
                                reviewing by eye (gitignored: transcript text)

FULL DATA INGESTION PIPELINE (run from the RepWise folder, in this order)
    1. python src/ingestion/build_corpus.py
           raw transcripts -> passages/, clean_txt/, manifest.csv,
           corrections_log.csv
    2. python src/ingestion/coverage_report.py
           passages + config/topic_lexicon.json -> coverage_matrix.csv,
           coverage_hits.csv, coverage_summary.md (which videos discuss
           which training topic)
    3. python src/ingestion/chunker.py
           passages + manifest.csv + coverage_matrix.csv -> chunks/
           (passages merged into ~450-word chunks with one passage of
           overlap, enriched with title, topics, papers and timestamp
           links: the units that get embedded for retrieval)
    4. Embedding and indexing of the chunks (not built yet).

    Rerun all steps in order after changing the raw files,
    term_corrections.json or topic_lexicon.json. The pipeline is
    deterministic: the same inputs always give the same outputs, and the
    raw_sha256 column of manifest.csv lets anyone check that their raw
    files match.

USAGE
    python src/ingestion/build_corpus.py
    python src/ingestion/build_corpus.py --corpus-version v1.1
    python src/ingestion/build_corpus.py --help     (all path options)
"""


from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent

TS_LINE = re.compile(r"^\s*(?:(\d+):)?(\d{1,3}):(\d{2})\s?(.*)$")
SOUND_TAG = re.compile(r"\[[^\]]*\]")
SENT_END = re.compile(r"[.?!…]$")
SPONSOR = re.compile(
    r"sponsoris|code promo|mon code|premier lien|partenaire de (?:la|cette) vidéo|"
    r"\d+\s?% de réduction|-\s?\d+\s?% (?:avec|sur)",
    re.I,
)
CTA = re.compile(
    r"abonnez-vous|abonne-toi|lâchez un like|mettez un like|activez la cloche|"
    r"(?:en|dans les) commentaires?|lien (?:en|dans la) description",
    re.I,
)

UNPUNCT_WORDS = 30   # passage size when the file has no punctuation
MAX_WORDS = 60       # hard cap for punctuated files
MIN_SENT_WORDS = 8   # do not close a passage on a very short sentence


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s).replace("’", "'").replace(" ", " ")


def key(s: str) -> str:
    """Alphanumeric-only key used to match file names to video titles."""
    s = nfc(s).lower()
    return "".join(ch for ch in s if ch.isalnum())


def fmt_ts(sec: int) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_raw(text: str) -> tuple[list[dict], str]:
    """Return caption fragments [{start, text}] and the detected layout."""
    frags: list[dict] = []
    empty_ts = 0
    for line in nfc(text).splitlines():
        m = TS_LINE.match(line)
        if m:
            h, mi, se, body = m.groups()
            start = int(h or 0) * 3600 + int(mi) * 60 + int(se)
            body = body.strip()
            if not body:
                empty_ts += 1
                continue
            frags.append({"start": start, "text": body})
        elif line.strip() and frags:
            frags[-1]["text"] += " " + line.strip()  # wrapped continuation line
    layout = "rolling" if empty_ts > len(frags) * 0.5 else "flat"
    return frags, layout


def clean_fragment(t: str) -> str:
    t = SOUND_TAG.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_punctuated(frags: list[dict]) -> bool:
    if not frags:
        return False
    with_punct = sum(1 for f in frags if re.search(r"[.?!,]", f["text"]))
    return with_punct / len(frags) > 0.2


def correct_fragments(frags: list[dict], rules, counter: Counter) -> list[dict]:
    """Apply corrections on the whole transcript so errors split across two
    caption fragments ("push" | "pool legs") are still fixed, then map the
    corrected text back onto fragments. A match spanning a boundary is
    assigned to the fragment where it starts."""
    texts = [clean_fragment(f["text"]) for f in frags]
    full, bounds, pos = [], [], 0
    for t in texts:
        bounds.append(pos)
        full.append(t)
        pos += len(t) + 1
    text = " ".join(full)
    for rx, rep, name in rules:
        matches = list(rx.finditer(text))
        if not matches:
            continue
        counter[name] += len(matches)
        out, last, shifts = [], 0, []
        for m in matches:
            new = m.expand(rep)
            out += [text[last:m.start()], new]
            shifts.append((m.start(), m.end(), len(new) - (m.end() - m.start())))
            last = m.end()
        out.append(text[last:])
        text = "".join(out)
        new_bounds = []
        for b in bounds:
            delta, nb = 0, None
            for s, e, d in shifts:
                if s < b < e:          # boundary inside a match: snap to match start
                    nb = s + delta
                    break
                if e <= b:
                    delta += d
            new_bounds.append(nb if nb is not None else b + delta)
        bounds = new_bounds
    bounds.append(len(text) + 1)
    return [
        {"start": f["start"], "text": text[bounds[i]:bounds[i + 1]].strip()}
        for i, f in enumerate(frags)
    ]


def to_passages(frags: list[dict], punctuated: bool) -> list[dict]:
    passages, buf, start = [], [], None
    prev = None
    for i, f in enumerate(frags):
        txt = f["text"]
        if not txt or txt == prev:
            continue
        prev = txt
        if start is None:
            start = f["start"]
        buf.append(txt)
        n = len(" ".join(buf).split())
        if punctuated:
            close = (SENT_END.search(txt) and n >= MIN_SENT_WORDS) or n >= MAX_WORDS
        else:
            close = n >= UNPUNCT_WORDS
        if close:
            end = frags[i + 1]["start"] if i + 1 < len(frags) else f["start"] + 2
            passages.append({"start": start, "end": max(end, start + 1), "text": " ".join(buf)})
            buf, start = [], None
    if buf:
        last = frags[-1]["start"] + 2
        passages.append({"start": start, "end": max(last, start + 1), "text": " ".join(buf)})
    return passages


def load_corrections(path: Path) -> list[tuple[re.Pattern, str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(re.compile(c["pattern"], re.I), c["replacement"], c["pattern"]) for c in data["corrections"]]


def parse_title_map(path: Path) -> dict[str, str]:
    """dictionnary_links.txt: 'https://youtu.be/<id> : <title>' per line."""
    out = {}
    for line in nfc(path.read_text(encoding="utf-8")).splitlines():
        m = re.match(r"\s*https?://youtu\.be/([\w-]{11})\S*\s*:\s*(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def parse_refs(path: Path) -> dict[str, list[str]]:
    """links.txt: a video URL followed by tab-indented paper URLs."""
    refs: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"youtu\.be/([\w-]{11})", line)
        if m and not line.startswith(("\t", " ")):
            current = m.group(1)
            refs.setdefault(current, [])
        elif line.strip().startswith("http") and current:
            refs[current].append(line.strip())
    return refs


def match_files(titles: dict[str, str], files: list[Path]) -> tuple[dict[str, Path], list[str]]:
    by_key = {key(f.stem): f for f in files}
    matched, issues = {}, []
    for vid, title in titles.items():
        k = key(title)
        if k in by_key:
            matched[vid] = by_key[k]
            continue
        best = difflib.get_close_matches(k, list(by_key), n=1, cutoff=0.85)
        if best:
            matched[vid] = by_key[best[0]]
            issues.append(f"fuzzy match: {vid} '{title}' -> '{by_key[best[0]].name}'")
        else:
            issues.append(f"NO TRANSCRIPT FILE for {vid} '{title}'")
    used = set(matched.values())
    issues += [f"transcript file not in dictionary: {f.name}" for f in files if f not in used]
    return matched, issues


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, default=PROJECT / "corpus")
    ap.add_argument("--titles", type=Path, default=PROJECT / "dictionnary_links.txt")
    ap.add_argument("--refs", type=Path, default=PROJECT / "links.txt")
    ap.add_argument("--corrections", type=Path, default=REPO / "config" / "term_corrections.json")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "corpus")
    ap.add_argument("--corpus-version", default=f"v1-{date.today().isoformat()}")
    args = ap.parse_args()

    titles = parse_title_map(args.titles)
    refs = parse_refs(args.refs)
    files = sorted(args.corpus_dir.glob("*.txt"))
    matched, issues = match_files(titles, files)
    rules = load_corrections(args.corrections)

    (args.out / "passages").mkdir(parents=True, exist_ok=True)
    (args.out / "clean_txt").mkdir(parents=True, exist_ok=True)

    manifest, corr_rows = [], []
    for vid, path in matched.items():
        raw = path.read_bytes()
        frags, layout = parse_raw(raw.decode("utf-8"))
        punct = is_punctuated(frags)
        counter: Counter = Counter()
        passages = to_passages(correct_fragments(frags, rules, counter), punct)
        sponsor_ts = []
        url = f"https://youtu.be/{vid}"
        with open(args.out / "passages" / f"{vid}.jsonl", "w", encoding="utf-8") as fj, \
             open(args.out / "clean_txt" / f"{vid}.txt", "w", encoding="utf-8") as ft:
            ft.write(f"# {titles[vid]}\n# {url}\n\n")
            for n, p in enumerate(passages, 1):
                text = p["text"]
                flags = []
                if SPONSOR.search(text):
                    flags.append("sponsor")
                    sponsor_ts.append(fmt_ts(p["start"]))
                if CTA.search(text):
                    flags.append("cta")
                rec = {
                    "video_id": vid,
                    "passage_id": f"{vid}_p{n:04d}",
                    "start": p["start"],
                    "end": p["end"],
                    "url_t": f"{url}?t={p['start']}",
                    "text": text,
                    "flags": flags,
                }
                fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ft.write(f"[{fmt_ts(p['start'])}]{' {' + ','.join(flags) + '}' if flags else ''} {text}\n")
        for name, c in counter.items():
            corr_rows.append({"video_id": vid, "pattern": name, "count": c})
        words = sum(len(p["text"].split()) for p in passages)
        manifest.append({
            "video_id": vid,
            "url": url,
            "title": titles[vid],
            "source_file": path.name,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "upload_date": "",  # not in the raw files; fill from YouTube if needed
            "duration_s": frags[-1]["start"] if frags else 0,
            "caption_layout": layout,
            "punctuated": punct,
            "n_passages": len(passages),
            "n_words": words,
            "n_corrections": sum(counter.values()),
            "sponsor_passages": "|".join(sponsor_ts),
            "scientific_references": "|".join(refs.get(vid, [])),
            "n_references": len(refs.get(vid, [])),
            "reference_review_status": "mentioned" if refs.get(vid) else "",
            "corpus_version": args.corpus_version,
        })

    with open(args.out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    with open(args.out / "corrections_log.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "pattern", "count"])
        w.writeheader()
        w.writerows(corr_rows)

    total_s = sum(m["duration_s"] for m in manifest)
    print(f"videos: {len(manifest)}  passages: {sum(m['n_passages'] for m in manifest)}  "
          f"words: {sum(m['n_words'] for m in manifest)}  hours: {total_s / 3600:.1f}")
    print(f"punctuated: {sum(m['punctuated'] for m in manifest)}  "
          f"with refs: {sum(1 for m in manifest if m['n_references'])}  "
          f"corrections applied: {sum(m['n_corrections'] for m in manifest)}")
    for i in issues:
        print("WARN", i)


if __name__ == "__main__":
    main()
