"""Shared body-region matching for corpus tags and retrieval query expansion."""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

LEXICON_PATH = Path(__file__).resolve().parents[1] / "config" / "body_part_lexicon.json"


def _fold_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


@lru_cache(maxsize=1)
def _compiled_lexicon() -> dict[str, re.Pattern[str]]:
    data = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    return {
        body_part: re.compile(
            "|".join(f"(?:{_fold_accents(pattern)})" for pattern in definition["patterns"])
        )
        for body_part, definition in data["body_parts"].items()
    }


def detect_body_parts(text: str) -> list[str]:
    """Return canonical regions explicitly mentioned in text."""
    folded = _fold_accents(text)
    return [
        body_part
        for body_part, pattern in _compiled_lexicon().items()
        if pattern.search(folded)
    ]
