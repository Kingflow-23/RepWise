"""Shared body-region matching for corpus tags and retrieval query expansion."""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

LEXICON_PATH = Path(__file__).resolve().parents[1] / "config" / "body_part_lexicon.json"


def _fold_accents(text: str) -> str:
    """Case-fold and remove combining accents from text.

    Args:
        text: Input text to normalize before matching.

    Returns:
        A case-folded string without combining marks; punctuation and spacing
        remain unchanged.
    """
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


@lru_cache(maxsize=1)
def _compiled_lexicon() -> dict[str, re.Pattern[str]]:
    """Load and compile the body-part patterns once per process.

    The one-entry cache avoids reparsing the shared JSON lexicon for each query.

    Returns:
        A mapping from canonical body-part names to accent-folded regexes.

    Raises:
        OSError: If the configured lexicon cannot be read.
        json.JSONDecodeError: If the lexicon is invalid JSON.
        KeyError: If the expected ``body_parts`` data is absent.
        re.error: If a lexicon pattern is not a valid regular expression.
    """
    data = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    return {
        body_part: re.compile(
            "|".join(
                f"(?:{_fold_accents(pattern)})" for pattern in definition["patterns"]
            )
        )
        for body_part, definition in data["body_parts"].items()
    }


def detect_body_parts(text: str) -> list[str]:
    """Return canonical regions whose lexicon patterns occur in ``text``.

    Matching is case- and accent-insensitive; names retain the lexicon's
    canonical spelling and order.

    Args:
        text: Query or passage to scan.

    Returns:
        Canonical body-part names with at least one match.

    Raises:
        OSError: If the body-part lexicon cannot be read.
        json.JSONDecodeError: If the lexicon is invalid JSON.
        re.error: If a lexicon pattern cannot be compiled.
    """
    folded = _fold_accents(text)
    return [
        body_part
        for body_part, pattern in _compiled_lexicon().items()
        if pattern.search(folded)
    ]
