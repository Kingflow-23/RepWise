"""Command line helper to create the local semantic retrieval index."""

from __future__ import annotations

import argparse
from pathlib import Path

from .semantic import DEFAULT_MODEL, build_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build RepWise's local embedding index."
    )
    parser.add_argument(
        "--chunks",
        default="data/corpus/chunks",
        help="folder containing per-video JSONL chunks",
    )
    parser.add_argument(
        "--index",
        default="data/corpus/index/semantic.npz",
        help="where to write the local index",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="locally installed Sentence Transformers model",
    )
    parser.add_argument(
        "--build", action="store_true", help="create or replace the index"
    )
    args = parser.parse_args(argv)
    if not args.build:
        parser.error("pass --build to create the index")
    result = build_index(args.chunks, args.index, args.model)
    print(f"Saved embedding index: {Path(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
