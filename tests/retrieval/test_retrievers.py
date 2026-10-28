import sys
from types import ModuleType

import pytest

from config.config import AppConfig
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.semantic import (
    SemanticRetriever,
    _load_model,
    build_index,
    corpus_fingerprint,
)
from src.retrieval.sparse import BM25Retriever
from src.schemas import RetrievedChunk


def test_retrieval_configuration_supports_only_the_three_named_modes(monkeypatch):
    assert AppConfig().retrieval_strategy == "hybrid"

    for strategy in ("bm25", "semantic", "hybrid"):
        monkeypatch.setenv("REPWISE_RETRIEVAL", strategy)
        assert AppConfig.from_environment().retrieval_strategy == strategy


def chunk(chunk_id: str, text: str, flags: list[str] | None = None):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        video_id=chunk_id,
        passage_ids=[f"{chunk_id}_p001"],
        title=chunk_id,
        video_url=f"https://example.com/{chunk_id}",
        flags=flags or [],
    )


def test_bm25_ranks_matching_chunks_and_excludes_commercial_content():
    relevant = chunk("training", "controlled abdominal repetitions")
    commercial = chunk("promo", "controlled abdominal repetitions", ["sponsor"])
    retriever = BM25Retriever([commercial, relevant])

    results = retriever.search("abdominal repetitions", top_k=2)

    assert [result.chunk_id for result in results] == ["training"]
    assert results[0].rank == 1


def test_hybrid_fuses_bm25_and_semantic_rankings_with_rrf():
    first = chunk("first", "first")
    second = chunk("second", "second")

    class RankedRetriever:
        def __init__(self, chunks):
            self.chunks = chunks

        def search(self, query, top_k):
            return self.chunks[:top_k]

    retriever = HybridRetriever(
        RankedRetriever([first, second]), RankedRetriever([second, first])
    )
    results = retriever.search("query", top_k=2)

    assert [result.chunk_id for result in results] == ["first", "second"]
    assert results[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert [result.rank for result in results] == [1, 2]


def test_semantic_index_round_trip_and_query(tmp_path, monkeypatch):
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
        def get_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            if texts[0].startswith("passage:"):
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
    with np.load(index_path, allow_pickle=False) as index:
        assert index["vectors"].shape == (2, 2)
        assert index["chunk_ids"].tolist() == ["one", "two"]
        assert str(index["fingerprint"].item()) == corpus_fingerprint(chunks)
        assert str(index["model"].item()) == "test-model"


def test_semantic_rejects_an_index_with_a_different_model_dimension(
    tmp_path, monkeypatch
):
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
        def get_embedding_dimension(self):
            return 3

    monkeypatch.setattr(
        semantic, "_load_model", lambda model_name: DifferentDimensionModel()
    )
    with pytest.raises(ValueError, match="dimension is incompatible"):
        SemanticRetriever([item], index_path, "test-model")


def test_semantic_rejects_stale_corpus_fingerprint(tmp_path, monkeypatch):
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
    module = ModuleType("sentence_transformers")

    class FailingSentenceTransformer:
        def __init__(self, model_name, local_files_only):
            raise OSError("local weights not found")

    module.SentenceTransformer = FailingSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    with pytest.raises(RuntimeError) as error:
        _load_model("intfloat/multilingual-e5-small")

    assert "local Hugging Face cache" in str(error.value)
    assert "retriever fallback are disabled" in str(error.value)
    assert "requirements.txt" in str(error.value)
