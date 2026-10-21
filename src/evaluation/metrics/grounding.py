"""Grounding metrics: citation precision, unsupported / grounded claim rates, abstention.

A generated answer exposes its claims as:

    {"text": "...", "kind": "evidence" | "assumption" | "policy",
     "citations": [{"chunk_id": "...", "quote": "verbatim words from the chunk"}]}

Level-1 support check (course slide 27): the quote must fuzzy-match the text of
the cited chunk. Transcripts are unpunctuated ASR, so both sides are normalised
(lower case, accents folded, punctuation dropped) before matching.

``assumption`` and ``policy`` claims are honest labels, not evidence: they are
counted separately and never inflate the grounded rate.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .common import Ratio

try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:  # optional dependency
    _fuzz = None

QUOTE_MATCH_THRESHOLD = 90
MIN_QUOTE_WORDS = 4
EVIDENCE_KINDS = {"evidence"}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"\w+", text))


def quote_score(quote: str, source: str) -> float:
    """0-100 similarity of the quote to its best-matching region of source."""
    q, s = normalise(quote), normalise(source)
    if not q:
        return 0.0
    if q in s:
        return 100.0
    if _fuzz is not None:
        return float(_fuzz.partial_ratio(q, s))
    blocks = SequenceMatcher(None, q, s, autojunk=False).get_matching_blocks()
    return 100.0 * sum(b.size for b in blocks) / len(q)


@dataclass
class CitationCheck:
    claim_index: int
    chunk_id: str | None
    status: str  # supported | quote_mismatch | missing_quote | unknown_chunk | not_retrieved
    score: float = 0.0


@dataclass
class GroundingReport:
    n_evidence_claims: int = 0
    n_assumption_claims: int = 0
    n_uncited_claims: int = 0
    supported_claims: int = 0
    grounded_claims: int = 0
    checks: list[CitationCheck] = field(default_factory=list)

    @property
    def citation_precision(self) -> Ratio:
        return Ratio(sum(c.status in {"supported", "not_retrieved"} for c in self.checks), len(self.checks))

    @property
    def unsupported_claim_rate(self) -> Ratio:
        return Ratio(self.n_evidence_claims - self.supported_claims, self.n_evidence_claims)

    @property
    def grounded_claim_rate(self) -> Ratio:
        return Ratio(self.grounded_claims, self.n_evidence_claims)

    @property
    def fabricated_citations(self) -> int:
        return sum(c.status == "unknown_chunk" for c in self.checks)


def check_claims(
    claims: list[dict],
    corpus: dict[str, str],
    retrieved_ids: set[str] | None = None,
    threshold: float = QUOTE_MATCH_THRESHOLD,
) -> GroundingReport:
    """corpus maps chunk_id -> chunk text; retrieved_ids is what the generator was given.

    supported: the quote matches a real chunk.
    grounded:  supported AND that chunk was part of the retrieved context.
    """
    report = GroundingReport()
    for i, claim in enumerate(claims):
        kind = claim.get("kind", "evidence")
        if kind not in EVIDENCE_KINDS:
            report.n_assumption_claims += 1
            continue
        report.n_evidence_claims += 1
        citations = claim.get("citations") or []
        if not citations:
            report.n_uncited_claims += 1
            continue
        supported = grounded = False
        for cit in citations:
            chunk_id, quote = cit.get("chunk_id"), cit.get("quote") or ""
            if chunk_id not in corpus:
                report.checks.append(CitationCheck(i, chunk_id, "unknown_chunk"))
                continue
            if len(normalise(quote).split()) < MIN_QUOTE_WORDS:
                report.checks.append(CitationCheck(i, chunk_id, "missing_quote"))
                continue
            score = quote_score(quote, corpus[chunk_id])
            if score < threshold:
                report.checks.append(CitationCheck(i, chunk_id, "quote_mismatch", score))
                continue
            supported = True
            in_context = retrieved_ids is None or chunk_id in retrieved_ids
            grounded = grounded or in_context
            report.checks.append(CitationCheck(i, chunk_id, "supported" if in_context else "not_retrieved", score))
        report.supported_claims += supported
        report.grounded_claims += grounded
    return report


# ---------------------------------------------------------------- abstention

ABSTAIN_STATUSES = {"abstain"}


def abstention_outcome(expected: str, actual: str) -> str | None:
    """Classify one case for abstention accuracy / false abstention rate.

    Returns "correct_abstention", "missed_abstention", "false_abstention",
    "answered" or None when the case is not about abstention.
    """
    if expected in ABSTAIN_STATUSES:
        return "correct_abstention" if actual in ABSTAIN_STATUSES else "missed_abstention"
    if expected == "ok":
        return "false_abstention" if actual in ABSTAIN_STATUSES else "answered"
    return None


def abstention_metrics(outcomes: list[str | None]) -> dict[str, Ratio]:
    needs = [o for o in outcomes if o in {"correct_abstention", "missed_abstention"}]
    answerable = [o for o in outcomes if o in {"false_abstention", "answered"}]
    return {
        "abstention_accuracy": Ratio(needs.count("correct_abstention"), len(needs)),
        "false_abstention_rate": Ratio(answerable.count("false_abstention"), len(answerable)),
    }
