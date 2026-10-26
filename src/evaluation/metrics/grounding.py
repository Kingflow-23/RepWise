"""Grounding metrics: citation precision, faithfulness and abstention.

A generated answer exposes its claims as:

    {"text": "...", "kind": "evidence" | "assumption" | "policy",
     "citations": [{"chunk_id": "...", "quote": "verbatim words from the chunk"}]}

The quote must appear word for word in the cited chunk. Transcripts are unpunctuated ASR, so both sides are normalised
(lower case, accents folded, punctuation dropped) before matching.

``assumption`` and ``policy`` claims are honest labels, not evidence: they are
counted separately and never inflate the grounded rate.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from .common import Ratio

MIN_QUOTE_WORDS = 4
EVIDENCE_KINDS = {"evidence"}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"\w+", text))


def quote_found(quote: str, source: str) -> bool:
    """True when the quote appears word for word in the source, ignoring case, accents and punctuation."""
    return normalise(quote) in normalise(source)


@dataclass
class CitationCheck:
    claim_index: int
    chunk_id: str | None
    status: str  # supported | quote_mismatch | missing_quote | unknown_chunk | not_retrieved


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
        return Ratio(
            sum(c.status in {"supported", "not_retrieved"} for c in self.checks),
            len(self.checks),
        )

    @property
    def faithfulness(self) -> Ratio:
        return Ratio(self.grounded_claims, self.n_evidence_claims)

    @property
    def fabricated_citations(self) -> int:
        return sum(c.status == "unknown_chunk" for c in self.checks)


def check_claims(
    claims: list[dict], corpus: dict[str, str], retrieved_ids: set[str] | None = None
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
            if not quote_found(quote, corpus[chunk_id]):
                report.checks.append(CitationCheck(i, chunk_id, "quote_mismatch"))
                continue
            supported = True
            in_context = retrieved_ids is None or chunk_id in retrieved_ids
            grounded = grounded or in_context
            report.checks.append(
                CitationCheck(
                    i, chunk_id, "supported" if in_context else "not_retrieved"
                )
            )
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
        return (
            "correct_abstention" if actual in ABSTAIN_STATUSES else "missed_abstention"
        )
    if expected == "ok":
        return "false_abstention" if actual in ABSTAIN_STATUSES else "answered"
    return None


def abstention_metrics(outcomes: list[str | None]) -> dict[str, Ratio]:
    needs = [o for o in outcomes if o in {"correct_abstention", "missed_abstention"}]
    answerable = [o for o in outcomes if o in {"false_abstention", "answered"}]
    return {
        "abstention_accuracy": Ratio(needs.count("correct_abstention"), len(needs)),
        "false_abstention_rate": Ratio(
            answerable.count("false_abstention"), len(answerable)
        ),
    }
