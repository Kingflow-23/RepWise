from src.evaluation.metrics.grounding import (
    abstention_metrics, abstention_outcome, check_claims, normalise, quote_found,
)

CORPUS = {
    "c1": "si vous êtes débutant il est préférable de commencer avec des temps de repos assez longs "
          "vous pouvez compter entre deux et trois minutes de récup entre chaque série",
    "c2": "généralement on recommande de réduire le volume de 50 à 70% mais de garder les mêmes charges",
}


def claim(quote, chunk_id="c1", kind="evidence"):
    return {"text": "claim", "kind": kind, "citations": [{"chunk_id": chunk_id, "quote": quote}]}


def test_normalise_folds_accents_case_and_punctuation():
    assert normalise("Été, Absences !") == "ete absences"


def test_exact_quote_is_found_even_with_punctuation_and_accents():
    assert quote_found("Entre deux et trois minutes de récup !", CORPUS["c1"])


def test_unrelated_quote_is_not_found():
    assert not quote_found("il faut toujours aller à l'échec sur le squat", CORPUS["c1"])



def test_supported_and_grounded_claim():
    r = check_claims([claim("entre deux et trois minutes de récup")], CORPUS, {"c1"})
    assert r.citation_precision.value == 1
    assert r.faithfulness.value == 1


def test_fabricated_citation_is_detected():
    r = check_claims([claim("selon harvard il faut sept séries", chunk_id="fake")], CORPUS, {"c1"})
    assert r.fabricated_citations == 1
    assert r.faithfulness.value == 0
    assert r.citation_precision.value == 0


def test_wrong_quote_on_real_chunk_is_not_support():
    r = check_claims([claim("réduire le volume de 50 à 70%", chunk_id="c1")], CORPUS, {"c1", "c2"})
    assert r.checks[0].status == "quote_mismatch"
    assert r.supported_claims == 0


def test_too_short_quote_does_not_count():
    r = check_claims([claim("trois minutes")], CORPUS, {"c1"})
    assert r.checks[0].status == "missing_quote"


def test_supported_but_not_retrieved_is_not_grounded():
    r = check_claims([claim("garder les mêmes charges", chunk_id="c2")], CORPUS, {"c1"})
    assert r.faithfulness.value == 0


def test_uncited_evidence_claim_is_unsupported():
    r = check_claims([{"text": "faites 4 séries", "kind": "evidence", "citations": []}], CORPUS)
    assert r.n_uncited_claims == 1
    assert r.faithfulness.value == 0


def test_assumptions_are_counted_apart_and_never_grounded():
    r = check_claims([{"text": "tempo 3-0-1-0", "kind": "assumption", "citations": []}], CORPUS)
    assert r.n_assumption_claims == 1
    assert r.faithfulness.value is None  # no evidence claim: N/A, not 100 %


def test_abstention_metrics():
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
