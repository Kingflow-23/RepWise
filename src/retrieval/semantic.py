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
            "Semantic retrieval needs the optional dependencies. Install them with: "
            "python -m pip install -r requirements-embeddings.txt"
        ) from error
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as error:
        raise RuntimeError(
            f"Embedding model {model_name!r} is not installed locally. "
            "Download it once with the command in docs/retrieval/hybrid_retrieval.md."
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
            "Index creation needs NumPy. Install requirements-embeddings.txt."
        ) from error
    chunks = [
        chunk
        for chunk in load_chunks(chunk_dir)
        if not {"sponsor", "cta"} & set(chunk.flags)
    ]
    model = _load_model(model_name)
    # E5 uses task prefixes to distinguish searches from stored passages.
    passages = [f"passage: {chunk.title}. {chunk.text}" for chunk in chunks]
    vectors = model.encode(passages, normalize_embeddings=True, show_progress_bar=True)
    target = Path(index_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as file:
        np.savez_compressed(
            file,
            vectors=np.asarray(vectors, dtype=np.float32),
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
                "Semantic search needs NumPy. Install requirements-embeddings.txt."
            ) from error
        self.np = np
        self.chunks = [
            chunk for chunk in chunks if not {"sponsor", "cta"} & set(chunk.flags)
        ]
        ids = [chunk.chunk_id for chunk in self.chunks]
        try:
            with np.load(index_path, allow_pickle=False) as index:
                self.vectors = index["vectors"]
                indexed_ids = index["chunk_ids"].tolist()
                indexed_model = str(index["model"].item())
                fingerprint = str(index["fingerprint"].item())
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Embedding index not found: {index_path}. Build it with `python -m src.retrieval.index --build`."
            ) from error
        if indexed_model != model_name:
            raise ValueError(
                f"Index uses {indexed_model}, but configuration requests {model_name}."
            )
        if indexed_ids != ids or fingerprint != corpus_fingerprint(self.chunks):
            raise ValueError(
                "Embedding index is stale. Rebuild it with `python -m src.retrieval.index --build`."
            )
        self.model = _load_model(model_name)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the highest cosine-similarity chunks with stable ranks."""
        query_vector = self.model.encode(
            [f"query: {query}"], normalize_embeddings=True
        )[0]
        scores = self.vectors @ query_vector
        # Stable sort makes ties reproducible in repeated index runs.
        order = self.np.argsort(-scores, kind="stable")[:top_k]
        return [
            self.chunks[int(position)].model_copy(
                update={"rank": rank, "score": float(scores[position])}
            )
            for rank, position in enumerate(order, start=1)
        ]
