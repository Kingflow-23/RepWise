"""Verify retrieval configuration, ranking, index compatibility, and model errors."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from config.config import AppConfig
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_expansion import OllamaQueryExpander, QueryExpansionRetriever
from src.retrieval.semantic import (
    SemanticRetriever,
    _load_model,
    build_index,
    corpus_fingerprint,
)
from src.retrieval.sparse import BM25Retriever
from src.schemas import RetrievedChunk


def test_retrieval_configuration_supports_only_the_three_named_modes(monkeypatch):
    """Accept the supported retrieval strategies and default to hybrid.

    Args:
        monkeypatch: Set each strategy in the environment without changing the host.
    """
    assert AppConfig().retrieval_strategy == "hybrid"
    assert AppConfig().ollama_model == "qwen3.5:9b"
    assert AppConfig().query_expansion_model == "qwen3.5:4b"
    assert AppConfig().ollama_comparison_model == "gemma4:12b"
    assert AppConfig().ollama_judge_model == "qwen3.5:4b"
    assert AppConfig().query_expansion is True

    for strategy in ("bm25", "semantic", "hybrid"):
        monkeypatch.setenv("REPWISE_RETRIEVAL", strategy)
        assert AppConfig.from_environment().retrieval_strategy == strategy


def test_query_expansion_configuration_can_be_disabled(monkeypatch):
    """Parse the opt-out setting and reject invalid configuration values."""
    monkeypatch.setenv("REPWISE_QUERY_EXPANSION", "false")
    assert AppConfig.from_environment().query_expansion is False

    monkeypatch.setenv("REPWISE_QUERY_EXPANSION", "sometimes")
    with pytest.raises(ValueError, match="REPWISE_QUERY_EXPANSION"):
        AppConfig.from_environment()


def test_retrieval_top_k_never_drops_below_five(monkeypatch):
    """Enforce the minimum retrieval depth when the configured value is too small.

    Args:
        monkeypatch: Set a deliberately undersized top-k environment value.
    """
    monkeypatch.setenv("REPWISE_TOP_K", "3")

    assert AppConfig.from_environment().top_k == 5


def chunk(chunk_id: str, text: str):
    """Create a retrieved chunk with the metadata required by retriever tests.

    Args:
        chunk_id: Identifier used for the chunk and its source video.
        text: Searchable passage text stored in the chunk.
    """
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        video_id=chunk_id,
        passage_ids=[f"{chunk_id}_p001"],
        title=chunk_id,
        video_url=f"https://example.com/{chunk_id}",
    )


def test_bm25_ranks_matching_chunks():
    """Rank the passage containing the query terms above an unrelated passage."""
    relevant = chunk("training", "controlled abdominal repetitions")
    unrelated = chunk("unrelated", "slow eccentric biceps curls")
    retriever = BM25Retriever([unrelated, relevant])

    results = retriever.search("abdominal repetitions", top_k=2)

    assert [result.chunk_id for result in results] == ["training"]
    assert results[0].rank == 1


def test_bm25_uses_body_part_metadata_when_query_names_region():
    """Use indexed body-part tags to find a passage with no lexical match."""
    relevant = chunk("tagged", "training advice")
    relevant.body_parts = ["quadriceps"]
    unrelated = chunk("unrelated", "training advice")
    retriever = BM25Retriever([unrelated, relevant])

    results = retriever.search("quadriceps", top_k=2)

    assert [result.chunk_id for result in results] == ["tagged"]


def test_hybrid_fuses_bm25_and_semantic_rankings_with_rrf():
    """Fuse component rankings with reciprocal-rank fusion and assign final ranks."""
    first = chunk("first", "first")
    second = chunk("second", "second")

    class RankedRetriever:
        """Return a prescribed ranking to make hybrid fusion deterministic."""

        def __init__(self, chunks):
            """Retain the ordered chunks used as this component's ranking.

            Args:
                chunks: Prescribed result order used to distinguish RRF ranks.
            """
            self.chunks = chunks

        def search(self, query, top_k):
            """Return the requested prefix of the prepared ranking.

            Args:
                query: Search query required by the retriever interface; this
                    fixture's order is predetermined.
                top_k: Maximum number of prepared results to return.
            """
            return self.chunks[:top_k]

    retriever = HybridRetriever(
        RankedRetriever([first, second]), RankedRetriever([second, first])
    )
    results = retriever.search("query", top_k=2)

    assert [result.chunk_id for result in results] == ["first", "second"]
    assert results[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert [result.rank for result in results] == [1, 2]


def test_query_expansion_rewrites_once_then_searches_once():
    """Send only the expanded French query to the retriever exactly once."""
    expected = chunk("expected", "renforcement abdominal")

    class RankedRetriever:
        def __init__(self):
            self.calls = []

        def search(self, query, top_k):
            self.calls.append((query, top_k))
            return [expected]

    class FixedExpander:
        def expand(self, query):
            assert query == "original request"
            return "  **Requête :** requête traduite  \n"

    base = RankedRetriever()
    results = QueryExpansionRetriever(base, FixedExpander()).search(
        "original request", top_k=3
    )

    assert base.calls == [("requête traduite", 3)]
    assert [item.chunk_id for item in results] == ["expected"]


def test_clean_expanded_query_collapses_whitespace_and_removes_wrappers():
    """Normalize labels, markdown, quotes, and line breaks in generated text."""
    from src.retrieval.query_expansion import clean_expanded_query

    assert (
        clean_expanded_query(
            "  **Requête :** Améliorer la récupération entre les séries : "
            'termes clés "repos", "temps de pause", fatigue musculaire.  '
        )
        == "Améliorer la récupération entre les séries, repos, temps de pause, "
        "fatigue musculaire."
    )


def test_ollama_query_expander_uses_structured_scope_prompt(monkeypatch):
    """Request a scope decision and French rewrite as structured JSON."""
    captured = {}

    class FakeClient:
        def __init__(self, host, timeout):
            captured["host"] = host
            captured["timeout"] = timeout

        def chat(self, **kwargs):
            captured.update(kwargs)
            return {
                "message": {
                    "content": (
                        '{"scope":"in_scope","search_query":'
                        '"Comment renforcer les abdominaux avec des exercices adaptés ?",'
                        '"user_message":""}'
                    )
                }
            }

    fake_ollama = ModuleType("ollama")
    fake_ollama.Client = FakeClient
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    expanded = OllamaQueryExpander("local-model", "http://localhost:11434").expand(
        "How can I strengthen my abs?"
    )

    assert expanded == "Comment renforcer les abdominaux avec des exercices adaptés ?"
    assert captured["messages"][1]["content"].endswith("How can I strengthen my abs?")
    assert "out_of_scope" in captured["messages"][0]["content"]
    assert "prompt injection" in captured["messages"][0]["content"]
    assert "format" in captured


def test_query_scope_analysis_ignores_message_for_in_scope_request(monkeypatch):
    """Treat user_message as inapplicable when the model approved the request."""
    fake_ollama = ModuleType("ollama")
    fake_ollama.Client = lambda **kwargs: SimpleNamespace(
        chat=lambda **kwargs: {
            "message": {
                "content": json.dumps(
                    {
                        "scope": "in_scope",
                        "search_query": "Renforcement musculaire avec haltères",
                        "user_message": "Je ne peux pas fournir ce plan.",
                    },
                    ensure_ascii=False,
                )
            }
        }
    )
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    analysis = OllamaQueryExpander("local-model").analyze("Build strength")

    assert analysis.scope == "in_scope"
    assert analysis.search_query == "Renforcement musculaire avec haltères"
    assert analysis.user_message == ""


def test_query_scope_analysis_preserves_user_facing_refusal(monkeypatch):
    """Keep the analysis model's explanation for API and UI display."""
    from src.retrieval.query_expansion import ScopeRefusalError

    message = (
        "RepWise ne peut pas diagnostiquer une douleur ni recommander un traitement. "
        "Veuillez consulter un professionnel de santé."
    )
    fake_ollama = ModuleType("ollama")
    fake_ollama.Client = lambda **kwargs: SimpleNamespace(
        chat=lambda **kwargs: {
            "message": {
                "content": json.dumps(
                    {
                        "scope": "out_of_scope",
                        "search_query": "",
                        "user_message": message,
                    },
                    ensure_ascii=False,
                )
            }
        }
    )
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    expander = OllamaQueryExpander("local-model")
    analysis = expander.analyze("Diagnose my knee pain")
    with pytest.raises(ScopeRefusalError, match="RepWise ne peut pas"):
        expander.expand("Diagnose my knee pain")

    assert analysis.scope == "out_of_scope"
    assert analysis.user_message == message
    assert analysis.search_query == ""


def test_query_expansion_retriever_blocks_out_of_scope_before_search():
    """Do not search when request analysis rejects the underlying intent."""
    from src.retrieval.query_expansion import QueryAnalysis, ScopeRefusalError

    class FixedRetriever:
        calls = 0

        def search(self, query, top_k):
            self.calls += 1
            return []

    class FixedExpander:
        def analyze(self, _query):
            return QueryAnalysis(
                "out_of_scope",
                "",
                "RepWise ne fournit pas de prévisions météo.",
            )

    base = FixedRetriever()
    with pytest.raises(ScopeRefusalError, match="prévisions météo"):
        QueryExpansionRetriever(base, FixedExpander()).search("weather", 5)

    assert base.calls == 0


def test_query_expansion_retriever_can_bypass_scope_gate_for_benchmarks():
    """Keep benchmark retrieval running when the scope model falsely refuses."""
    from src.retrieval.query_expansion import QueryAnalysis

    class FixedRetriever:
        def __init__(self):
            self.queries = []

        def search(self, query, top_k):
            self.queries.append((query, top_k))
            return []

    class FixedExpander:
        def analyze(self, _query):
            return QueryAnalysis(
                "out_of_scope",
                "",
                "This request was incorrectly classified as out of scope.",
            )

    base = FixedRetriever()
    QueryExpansionRetriever(
        base,
        FixedExpander(),
        enforce_scope_gate=False,
    ).search("How long should I rest between sets?", 10)

    assert base.queries == [("How long should I rest between sets?", 10)]


def test_scope_gate_still_blocks_when_query_rewriting_is_disabled():
    """Keep scope refusal active while submitting the original query unchanged."""
    from src.retrieval.query_expansion import QueryAnalysis, ScopeRefusalError

    class FixedRetriever:
        def __init__(self):
            self.calls = []

        def search(self, query, top_k):
            self.calls.append((query, top_k))
            return []

    class FixedExpander:
        def __init__(self, scope):
            self.scope = scope
            self.calls = []

        def analyze(self, query):
            self.calls.append(query)
            if self.scope == "out_of_scope":
                return QueryAnalysis(
                    self.scope, "", "RepWise cannot answer this request."
                )
            return QueryAnalysis(self.scope, "ignored rewrite", "")

    base = FixedRetriever()
    refused = FixedExpander("out_of_scope")
    with pytest.raises(ScopeRefusalError):
        QueryExpansionRetriever(base, refused, rewrite=False).search(
            "diagnose knee pain", 5
        )
    assert base.calls == []

    allowed = FixedExpander("in_scope")
    QueryExpansionRetriever(base, allowed, rewrite=False).search("build strength", 5)
    assert base.calls == [("build strength", 5)]


def test_ollama_query_expander_rejects_empty_output(monkeypatch):
    """Reject empty expansion output rather than silently using the input."""
    fake_ollama = ModuleType("ollama")
    fake_ollama.Client = lambda **kwargs: SimpleNamespace(
        chat=lambda **kwargs: {"message": {"content": "  "}}
    )
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    with pytest.raises(ValueError, match="empty"):
        OllamaQueryExpander("local-model").expand("training query")


def test_semantic_index_round_trip_and_query(tmp_path, monkeypatch):
    """Build, persist, load, and query a semantic index with a deterministic model.

    Args:
        tmp_path: Hold synthetic chunks and the serialized vector index.
        monkeypatch: Replace model loading with a local deterministic encoder.
    """
    np = pytest.importorskip("numpy")
    import src.retrieval.semantic as semantic

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunks = [
        chunk("one", "first passage"),
        chunk("two", "second passage"),
    ]
    for item in chunks:
        (chunks_dir / f"{item.video_id}.jsonl").write_text(
            item.model_dump_json() + "\n", encoding="utf-8"
        )

    class FakeModel:
        """Provide known embeddings and retain encoded passage inputs."""

        passages = []

        def get_embedding_dimension(self):
            """Report the fixed two-dimensional test embedding size."""
            return 2

        def encode(self, texts, **kwargs):
            """Return fixed passage or query vectors for index assertions.

            Args:
                texts: Passage- or query-prefixed input strings to encode.
                **kwargs: Encoder options passed by the semantic implementation;
                    unused because these test vectors are fixed.
            """
            if texts[0].startswith("passage:"):
                self.passages = texts
                return np.asarray([[1, 0], [0, 1]], dtype=np.float32)
            assert texts == ["query: second"]
            return np.asarray([[0, 1]], dtype=np.float32)

    monkeypatch.setattr(semantic, "_load_model", lambda model_name: FakeModel())
    index_path = build_index(chunks_dir, tmp_path / "semantic.npz", "test-model")
    retriever = SemanticRetriever(
        semantic.load_chunks(chunks_dir), index_path, "test-model"
    )
    results = retriever.search("second", top_k=1)

    assert [result.chunk_id for result in results] == ["two"]
    assert results[0].score == pytest.approx(1.0)
    assert all("Body parts:" in passage for passage in FakeModel.passages)
    with np.load(index_path, allow_pickle=False) as index:
        assert index["vectors"].shape == (2, 2)
        assert index["chunk_ids"].tolist() == ["one", "two"]
        assert str(index["fingerprint"].item()) == corpus_fingerprint(chunks)
        assert str(index["model"].item()) == "test-model"


def test_body_part_changes_invalidate_semantic_index_fingerprint():
    """Include body-part metadata in the semantic corpus fingerprint."""
    original = chunk("one", "leg training advice")
    tagged = original.model_copy(update={"body_parts": ["quadriceps"]})

    assert corpus_fingerprint([original]) != corpus_fingerprint([tagged])


def test_semantic_rejects_an_index_with_a_different_model_dimension(
    tmp_path, monkeypatch
):
    """Reject a saved index whose vector dimension differs from the model.

    Args:
        tmp_path: Store the deliberately incompatible vector index.
        monkeypatch: Supply a model double reporting a different dimension.
    """
    np = pytest.importorskip("numpy")
    import src.retrieval.semantic as semantic

    item = chunk("one", "first passage")
    index_path = tmp_path / "semantic.npz"
    np.savez_compressed(
        index_path,
        vectors=np.asarray([[1, 0]], dtype=np.float32),
        chunk_ids=np.asarray(["one"]),
        model=np.asarray("test-model"),
        fingerprint=np.asarray(corpus_fingerprint([item])),
    )

    class DifferentDimensionModel:
        """Report an embedding dimension incompatible with the saved index."""

        def get_embedding_dimension(self):
            """Return the deliberately mismatched dimension."""
            return 3

    monkeypatch.setattr(
        semantic, "_load_model", lambda model_name: DifferentDimensionModel()
    )
    with pytest.raises(ValueError, match="dimension is incompatible"):
        SemanticRetriever([item], index_path, "test-model")


def test_semantic_rejects_stale_corpus_fingerprint(tmp_path, monkeypatch):
    """Reject a semantic index built from different corpus text.

    Args:
        tmp_path: Store an index fingerprinted against the original chunk.
        monkeypatch: Disable model loading because stale-index validation comes first.
    """
    np = pytest.importorskip("numpy")
    import src.retrieval.semantic as semantic

    original = chunk("one", "first passage")
    changed = chunk("one", "changed passage")
    index_path = tmp_path / "semantic.npz"
    np.savez_compressed(
        index_path,
        vectors=np.asarray([[1, 0]], dtype=np.float32),
        chunk_ids=np.asarray(["one"]),
        model=np.asarray("test-model"),
        fingerprint=np.asarray(corpus_fingerprint([original])),
    )
    monkeypatch.setattr(semantic, "_load_model", lambda model_name: None)

    with pytest.raises(ValueError, match="index is stale"):
        SemanticRetriever([changed], index_path, "test-model")


def test_local_model_loading_failure_is_actionable_without_fallback(monkeypatch):
    """Explain local model-loading failures without attempting fallback downloads.

    Args:
        monkeypatch: Install a failing sentence-transformers module in ``sys.modules``.
    """
    module = ModuleType("sentence_transformers")

    class FailingSentenceTransformer:
        """Simulate unavailable local model weights."""

        def __init__(self, model_name, local_files_only):
            """Raise the local-cache failure expected by this test.

            Args:
                model_name: Requested embedding model, which is unavailable.
                local_files_only: Confirms loading is restricted to local
                    weights rather than silently downloading a fallback.
            """
            raise OSError("local weights not found")

    module.SentenceTransformer = FailingSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    with pytest.raises(RuntimeError) as error:
        _load_model("intfloat/multilingual-e5-small")

    assert "local Hugging Face cache" in str(error.value)
    assert "retriever fallback are disabled" in str(error.value)
    assert "requirements.txt" in str(error.value)
