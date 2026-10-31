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
    """Lowercase, fold accents, and retain only space-separated word tokens.

    Args:
        text: Claim quote or source text to normalize.

    Returns:
        Text with combining marks and punctuation removed.
    """
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"\w+", text))


def quote_found(quote: str, source: str) -> bool:
    """Check whether a normalized quote is a substring of normalized source.

    Args:
        quote: Citation excerpt supplied for a claim.
        source: Full cited chunk text.

    Returns:
        ``True`` when the quote occurs contiguously after case, accent, and
        punctuation normalization; this does not establish claim entailment.
    """
    return normalise(quote) in normalise(source)


@dataclass
class CitationCheck:
    """Outcome for one citation attempt attached to a claim.

    Attributes:
        claim_index: Zero-based position of the claim in the input list.
        chunk_id: Referenced chunk ID, or ``None`` if absent.
        status: One of ``supported``, ``quote_mismatch``, ``missing_quote``,
            ``unknown_chunk``, or ``not_retrieved``.
    """

    claim_index: int
    chunk_id: str | None
    status: str  # supported | quote_mismatch | missing_quote | unknown_chunk | not_retrieved


@dataclass
class GroundingReport:
    """Accumulated citation and evidence-claim counts for one output.

    Attributes:
        n_evidence_claims: Number of claims labeled as evidence.
        n_assumption_claims: Number of non-evidence claims.
        n_uncited_claims: Evidence claims without citation records.
        supported_claims: Evidence claims with at least one source quote match.
        grounded_claims: Supported claims whose source was in retrieved context.
        checks: Per-citation outcomes in claim and citation order.
    """

    n_evidence_claims: int = 0
    n_assumption_claims: int = 0
    n_uncited_claims: int = 0
    supported_claims: int = 0
    grounded_claims: int = 0
    checks: list[CitationCheck] = field(default_factory=list)

    @property
    def citation_precision(self) -> Ratio:
        """Return the fraction of citation checks that match real source quotes.

        Returns:
            Ratio counting ``supported`` and ``not_retrieved`` as valid quote
            matches over all citation checks; undefined when there are none.
        """
        return Ratio(
            sum(c.status in {"supported", "not_retrieved"} for c in self.checks),
            len(self.checks),
        )

    @property
    def faithfulness(self) -> Ratio:
        """Return grounded evidence claims divided by evidence claims.

        Returns:
            Ratio of retrieved-context-supported evidence claims to all
            evidence claims; undefined when no evidence claims were evaluated.
        """
        return Ratio(self.grounded_claims, self.n_evidence_claims)

    @property
    def fabricated_citations(self) -> int:
        """Count citations that refer to chunk IDs absent from the corpus.

        Returns:
            Number of ``unknown_chunk`` citation checks.
        """
        return sum(c.status == "unknown_chunk" for c in self.checks)


def check_claims(
    claims: list[dict], corpus: dict[str, str], retrieved_ids: set[str] | None = None
) -> GroundingReport:
    """Check citation quotes against the corpus and retrieved context.

    Only ``kind="evidence"`` claims are scored; other claim kinds are counted
    as assumptions/policies and excluded from grounded rates. A quote must
    contain at least :data:`MIN_QUOTE_WORDS` normalized words.

    Args:
        claims: Claim dictionaries, optionally containing ``kind`` and a list
            of citations with ``chunk_id`` and ``quote``.
        corpus: Mapping from known chunk IDs to their full text.
        retrieved_ids: Chunk IDs supplied to the generator. ``None`` means
            context membership is not being checked; an empty set means no
            cited source was retrieved.

    Returns:
        Counts and per-citation statuses. A quote match is ``supported``;
        it is ``grounded`` only when the matching chunk is in retrieved
        context.
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

    Args:
        expected: Expected case status.
        actual: Status returned by the system.

    Returns:
        One of the documented outcome strings for ``abstain`` or ``ok`` cases;
        otherwise ``None`` for statuses outside this abstention comparison.
    """
    if expected in ABSTAIN_STATUSES:
        return (
            "correct_abstention" if actual in ABSTAIN_STATUSES else "missed_abstention"
        )
    if expected == "ok":
        return "false_abstention" if actual in ABSTAIN_STATUSES else "answered"
    return None


def abstention_metrics(outcomes: list[str | None]) -> dict[str, Ratio]:
    """Aggregate abstention accuracy and false-abstention rate.

    Args:
        outcomes: Per-case labels from :func:`abstention_outcome`; ``None`` and
            unrelated outcomes are ignored.

    Returns:
        Ratios named ``abstention_accuracy`` and ``false_abstention_rate``.
        Each denominator includes only cases applicable to that metric.
    """
    needs = [o for o in outcomes if o in {"correct_abstention", "missed_abstention"}]
    answerable = [o for o in outcomes if o in {"false_abstention", "answered"}]
    return {
        "abstention_accuracy": Ratio(needs.count("correct_abstention"), len(needs)),
        "false_abstention_rate": Ratio(
            answerable.count("false_abstention"), len(answerable)
        ),
    }
