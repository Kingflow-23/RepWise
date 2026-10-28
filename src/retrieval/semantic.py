"""Local semantic retrieval backed by a small, on-disk embedding matrix.

The index stores vectors and stable chunk IDs only. Source text and metadata are
loaded from the JSONL corpus so an index cannot silently replace provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..schemas.models import RetrievedChunk

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


def load_chunks(chunk_dir: str | Path) -> list[RetrievedChunk]:
    """Read all per-video JSONL files in a deterministic order."""
    chunks = []
    for path in sorted(Path(chunk_dir).glob("*.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            chunk = RetrievedChunk.model_validate(json.loads(line))
            if not chunk.chunk_id:
                raise ValueError(
                    f"{path.name}:{line_number} has no chunk_id. Rebuild the corpus with the current ingestion code."
                )
            chunks.append(chunk)
    if not chunks:
        raise FileNotFoundError(
            f"No JSONL chunks found in {chunk_dir}. Build the corpus first."
        )
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Chunk IDs must be unique before an embedding index can be built."
        )
    return chunks


def corpus_fingerprint(chunks: list[RetrievedChunk]) -> str:
    """Hash the text and IDs that determine the vectors."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.title.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_model(model_name: str):
    """Import the optional embedding package only when retrieval is requested."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Semantic retrieval dependencies are missing. Install them with: "
            "python -m pip install -r requirements.txt"
        ) from error
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as error:
        raise RuntimeError(
            f"Could not load embedding model {model_name!r} from the local Hugging Face "
            "cache (network access and retriever fallback are disabled). Install it with "
            "`python -m pip install -r requirements.txt`, then download it "
            "once using the command in docs/retrieval/hybrid_retrieval.md. "
            f"Underlying error: {type(error).__name__}: {error}"
        ) from error


def build_index(
    chunk_dir: str | Path,
    index_path: str | Path,
    model_name: str = DEFAULT_MODEL,
) -> Path:
    """Embed corpus chunks locally and save a reproducible NumPy index."""
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Index creation needs NumPy. Install requirements.txt."
        ) from error
    chunks = [
        chunk
        for chunk in load_chunks(chunk_dir)
        if not {"sponsor", "cta"} & set(chunk.flags)
    ]
    if not chunks:
        raise ValueError(
            "No eligible chunks remain after excluding sponsor and CTA content."
        )
    model = _load_model(model_name)
    # E5 uses task prefixes to distinguish searches from stored passages.
    passages = [f"passage: {chunk.title}. {chunk.text}" for chunk in chunks]
    vectors = np.asarray(
        model.encode(passages, normalize_embeddings=True, show_progress_bar=True),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] == 0:
        raise ValueError(
            "Embedding model returned an invalid index matrix. "
            f"Expected ({len(chunks)}, embedding_dimension), got {vectors.shape}."
        )
    model_dimension = model.get_sentence_embedding_dimension()
    if model_dimension is not None and vectors.shape[1] != model_dimension:
        raise ValueError(
            "Embedding model returned vectors with an unexpected dimension: "
            f"expected {model_dimension}, got {vectors.shape[1]}."
        )
    target = Path(index_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as file:
        np.savez_compressed(
            file,
            vectors=vectors,
            chunk_ids=np.asarray([chunk.chunk_id for chunk in chunks]),
            model=np.asarray(model_name),
            fingerprint=np.asarray(corpus_fingerprint(chunks)),
        )
    return target


class SemanticRetriever:
    """Rank local corpus chunks by cosine similarity to an E5 query vector."""

    def __init__(
        self,
        chunks: list[RetrievedChunk],
        index_path: str | Path,
        model_name: str = DEFAULT_MODEL,
    ):
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                "Semantic search needs NumPy. Install requirements.txt."
            ) from error
        self.np = np
        self.chunks = [
            chunk for chunk in chunks if not {"sponsor", "cta"} & set(chunk.flags)
        ]
        ids = [chunk.chunk_id for chunk in self.chunks]
        if not ids or any(not chunk_id for chunk_id in ids):
            raise ValueError(
                "Semantic retrieval requires eligible chunks with non-empty chunk IDs."
            )
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Chunk IDs must be unique before an embedding index can be loaded."
            )
        try:
            with np.load(index_path, allow_pickle=False) as index:
                self.vectors = np.asarray(index["vectors"], dtype=np.float32)
                indexed_ids = index["chunk_ids"].tolist()
                indexed_model = str(index["model"].item())
                fingerprint = str(index["fingerprint"].item())
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Embedding index not found: {index_path}. Build it with `python -m src.retrieval.index --build`."
            ) from error
        except (KeyError, OSError, ValueError) as error:
            raise ValueError(
                f"Embedding index {index_path} is unreadable or incomplete. "
                "Rebuild it with `python -m src.retrieval.index --build`."
            ) from error
        if indexed_model != model_name:
            raise ValueError(
                f"Index uses {indexed_model}, but configuration requests {model_name}."
            )
        if (
            self.vectors.ndim != 2
            or self.vectors.shape[0] != len(indexed_ids)
            or self.vectors.shape[1] == 0
        ):
            raise ValueError(
                "Embedding index has an invalid vector matrix or row count. "
                "Rebuild it with `python -m src.retrieval.index --build`."
            )
        if indexed_ids != ids or fingerprint != corpus_fingerprint(self.chunks):
            raise ValueError(
                "Embedding index is stale. Rebuild it with `python -m src.retrieval.index --build`."
            )
        self.model = _load_model(model_name)
        model_dimension = self.model.get_sentence_embedding_dimension()
        if model_dimension is not None and self.vectors.shape[1] != model_dimension:
            raise ValueError(
                "Embedding index dimension is incompatible with the configured model: "
                f"index has {self.vectors.shape[1]}, model expects {model_dimension}. "
                "Rebuild the index with `python -m src.retrieval.index --build`."
            )

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the highest cosine-similarity chunks with stable ranks."""
        query_vectors = self.np.asarray(
            self.model.encode([f"query: {query}"], normalize_embeddings=True),
            dtype=self.np.float32,
        )
        if query_vectors.shape != (1, self.vectors.shape[1]):
            raise ValueError(
                "Embedding model returned a query vector with an incompatible shape: "
                f"expected (1, {self.vectors.shape[1]}), got {query_vectors.shape}."
            )
        query_vector = query_vectors[0]
        scores = self.vectors @ query_vector
        # Stable sort makes ties reproducible in repeated index runs.
        order = self.np.argsort(-scores, kind="stable")[:top_k]
        return [
            self.chunks[int(position)].model_copy(
                update={"rank": rank, "score": float(scores[position])}
            )
            for rank, position in enumerate(order, start=1)
        ]
