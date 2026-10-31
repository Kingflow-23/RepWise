"""Verify quote matching, citation grounding, and abstention metric behavior."""

from src.evaluation.metrics.grounding import (
    abstention_metrics,
    abstention_outcome,
    check_claims,
    normalise,
    quote_found,
)

CORPUS = {
    "c1": "si vous êtes débutant il est préférable de commencer avec des temps de repos assez longs "
    "vous pouvez compter entre deux et trois minutes de récup entre chaque série",
    "c2": "généralement on recommande de réduire le volume de 50 à 70% mais de garder les mêmes charges",
}


def claim(quote, chunk_id="c1", kind="evidence"):
    """Create one cited claim for grounding-metric test cases.

    Args:
        quote: Quoted source text to associate with the claim.
        chunk_id: Corpus chunk named by the citation.
        kind: Claim category, such as evidence or assumption.
    """
    return {
        "text": "claim",
        "kind": kind,
        "citations": [{"chunk_id": chunk_id, "quote": quote}],
    }


def test_normalise_folds_accents_case_and_punctuation():
    """Normalize accents, case, and punctuation before quote comparison."""
    assert normalise("Été, Absences !") == "ete absences"


def test_exact_quote_is_found_even_with_punctuation_and_accents():
    """Match an exact source quote despite punctuation and accent differences."""
    assert quote_found("Entre deux et trois minutes de récup !", CORPUS["c1"])


def test_unrelated_quote_is_not_found():
    """Do not accept a quote that is absent from the source chunk."""
    assert not quote_found(
        "il faut toujours aller à l'échec sur le squat", CORPUS["c1"]
    )


def test_supported_and_grounded_claim():
    """Give full citation precision and faithfulness to retrieved source evidence."""
    r = check_claims([claim("entre deux et trois minutes de récup")], CORPUS, {"c1"})
    assert r.citation_precision.value == 1
    assert r.faithfulness.value == 1


def test_fabricated_citation_is_detected():
    """Count citations to absent chunks as fabricated and unsupported."""
    r = check_claims(
        [claim("selon harvard il faut sept séries", chunk_id="fake")], CORPUS, {"c1"}
    )
    assert r.fabricated_citations == 1
    assert r.faithfulness.value == 0
    assert r.citation_precision.value == 0


def test_wrong_quote_on_real_chunk_is_not_support():
    """Reject a valid chunk citation when its quoted text is not in that chunk."""
    r = check_claims(
        [claim("réduire le volume de 50 à 70%", chunk_id="c1")], CORPUS, {"c1", "c2"}
    )
    assert r.checks[0].status == "quote_mismatch"
    assert r.supported_claims == 0


def test_too_short_quote_does_not_count():
    """Treat quotes below the minimum evidence length as missing."""
    r = check_claims([claim("trois minutes")], CORPUS, {"c1"})
    assert r.checks[0].status == "missing_quote"


def test_supported_but_not_retrieved_is_not_grounded():
    """Require cited support to have been retrieved for a claim to be grounded."""
    r = check_claims([claim("garder les mêmes charges", chunk_id="c2")], CORPUS, {"c1"})
    assert r.faithfulness.value == 0


def test_uncited_evidence_claim_is_unsupported():
    """Count evidence claims without citations as uncited and unfaithful."""
    r = check_claims(
        [{"text": "faites 4 séries", "kind": "evidence", "citations": []}], CORPUS
    )
    assert r.n_uncited_claims == 1
    assert r.faithfulness.value == 0


def test_assumptions_are_counted_apart_and_never_grounded():
    """Exclude assumptions from evidence support and report no-evidence faithfulness as N/A."""
    r = check_claims(
        [{"text": "tempo 3-0-1-0", "kind": "assumption", "citations": []}], CORPUS
    )
    assert r.n_assumption_claims == 1
    assert r.faithfulness.value is None  # no evidence claim: N/A, not 100 %


def test_abstention_metrics():
    """Compute abstention accuracy and false-abstention rate from mixed outcomes."""
    outcomes = [
        abstention_outcome("abstain", "abstain"),
        abstention_outcome("abstain", "ok"),
        abstention_outcome("ok", "abstain"),
        abstention_outcome("ok", "ok"),
        abstention_outcome("ok", "ok"),
        abstention_outcome("clarify", "clarify"),
    ]
    m = abstention_metrics(outcomes)
    assert m["abstention_accuracy"].fmt() == "0.500 (1/2)"
    assert m["false_abstention_rate"].fmt() == "0.333 (1/3)"
