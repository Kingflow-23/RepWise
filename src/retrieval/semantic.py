"""Local semantic retrieval backed by a small, on-disk embedding matrix.

The index stores vectors and stable chunk IDs only. Source text and metadata are
loaded from the JSONL corpus so an index cannot silently replace provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter

from ..schemas.models import RetrievedChunk

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
logger = logging.getLogger(__name__)


def load_chunks(chunk_dir: str | Path) -> list[RetrievedChunk]:
    """Read and validate per-video JSONL chunks in sorted path order.

    Args:
        chunk_dir: Directory containing one or more per-video ``.jsonl`` files.

    Returns:
        Validated chunks in filename and line order.

    Raises:
        FileNotFoundError: If no JSONL chunks are found.
        ValueError: If a record is invalid, lacks a chunk ID, or IDs are
            duplicated.
        OSError: If a chunk file cannot be read.
        json.JSONDecodeError: If a non-empty line is not valid JSON.
    """
    started = perf_counter()
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
    logger.info(
        "Corpus chunks loaded directory=%s chunks=%s elapsed_ms=%.1f",
        chunk_dir,
        len(chunks),
        (perf_counter() - started) * 1000,
    )
    return chunks


def corpus_fingerprint(chunks: list[RetrievedChunk]) -> str:
    """Hash fields that determine the stored semantic vectors.

    Args:
        chunks: Ordered chunks used to build or validate an index.

    Returns:
        SHA-256 hex digest over each chunk's ID, title, text, and body parts in
        iteration order. Other metadata is intentionally excluded because it
        does not contribute to passage embeddings.
    """
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.title.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(" ".join(chunk.body_parts).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_model(model_name: str):
    """Load a locally cached Sentence Transformers model on demand.

    Args:
        model_name: Hugging Face model identifier to load from the local cache.

    Returns:
        A ``SentenceTransformer`` configured for local-files-only loading.

    Raises:
        RuntimeError: If the optional package is missing or the model cannot be
            loaded from the local cache. The original error is chained.
    """
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
            "once using the command in docs/setup.md. "
            f"Underlying error: {type(error).__name__}: {error}"
        ) from error


def build_index(
    chunk_dir: str | Path,
    index_path: str | Path,
    model_name: str = DEFAULT_MODEL,
) -> Path:
    """Embed corpus chunks locally and save vectors with corpus identity.

    E5 passage prefixes and normalized embeddings are used, while the index
    stores vectors, ordered chunk IDs, model name, and a corpus fingerprint.

    Args:
        chunk_dir: Directory containing per-video corpus chunk JSONL files.
        index_path: Destination ``.npz`` path; parent directories are created.
        model_name: Locally cached embedding model identifier.

    Returns:
        Destination path of the newly written compressed NumPy index.

    Raises:
        RuntimeError: If NumPy or the Sentence Transformers dependency/model is
            unavailable.
        FileNotFoundError: If no chunks are present.
        ValueError: If the model returns vectors of an invalid shape or
            dimension.
        OSError: If the destination index cannot be created or written.
    """
    started = perf_counter()
    logger.info("Embedding index build started model=%s", model_name)
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Index creation needs NumPy. Install requirements.txt."
        ) from error
    chunks = load_chunks(chunk_dir)
    model = _load_model(model_name)
    # E5 uses task prefixes to distinguish searches from stored passages.
    passages = [
        f"passage: {chunk.title}. Body parts: {', '.join(chunk.body_parts)}. "
        f"{chunk.text}"
        for chunk in chunks
    ]
    vectors = np.asarray(
        model.encode(passages, normalize_embeddings=True, show_progress_bar=True),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] == 0:
        raise ValueError(
            "Embedding model returned an invalid index matrix. "
            f"Expected ({len(chunks)}, embedding_dimension), got {vectors.shape}."
        )
    model_dimension = model.get_embedding_dimension()
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
    logger.info(
        "Embedding index build completed eligible_chunks=%s dimensions=%s elapsed_ms=%.1f",
        len(chunks),
        vectors.shape[1],
        (perf_counter() - started) * 1000,
    )
    return target


class SemanticRetriever:
    """Rank local corpus chunks by cosine similarity to an E5 query vector.

    The saved index is checked against ordered chunk IDs, a fingerprint, and
    the configured model before it is used. Text and metadata remain sourced
    from the caller-provided chunks rather than serialized into the index.
    """

    def __init__(
        self,
        chunks: list[RetrievedChunk],
        index_path: str | Path,
        model_name: str = DEFAULT_MODEL,
    ):
        """Load and validate vectors for the supplied ordered chunks.

        Args:
            chunks: Corpus chunks corresponding exactly to the saved index's
                ordered chunk IDs.
            index_path: Compressed NumPy index created by :func:`build_index`.
            model_name: Model identifier required by both the index and local
                model cache.

        Raises:
            RuntimeError: If NumPy or the embedding model dependency is missing.
            FileNotFoundError: If the index file does not exist.
            ValueError: If chunks or the index are invalid, stale, use another
                model, or have an incompatible vector dimension.
        """
        started = perf_counter()
        logger.info(
            "Semantic retriever initialization started model=%s chunks=%s",
            model_name,
            len(chunks),
        )
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                "Semantic search needs NumPy. Install requirements.txt."
            ) from error
        self.np = np
        self.chunks = chunks
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
            logger.error(
                "Semantic index stale indexed_chunks=%s current_chunks=%s",
                len(indexed_ids),
                len(ids),
            )
            raise ValueError(
                "Embedding index is stale. Rebuild it with `python -m src.retrieval.index --build`."
            )
        self.model = _load_model(model_name)
        model_dimension = self.model.get_embedding_dimension()
        if model_dimension is not None and self.vectors.shape[1] != model_dimension:
            raise ValueError(
                "Embedding index dimension is incompatible with the configured model: "
                f"index has {self.vectors.shape[1]}, model expects {model_dimension}. "
                "Rebuild the index with `python -m src.retrieval.index --build`."
            )
        logger.info(
            "Semantic retriever ready chunks=%s dimensions=%s elapsed_ms=%.1f",
            len(self.chunks),
            self.vectors.shape[1],
            (perf_counter() - started) * 1000,
        )

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the highest cosine-similarity chunks with stable ranks.

        Args:
            query: User query encoded with the E5 ``query:`` prefix.
            top_k: Maximum number of corpus chunks to return.

        Returns:
            Chunks ordered by descending cosine similarity, with one-based
            ranks and floating-point similarity scores attached. A stable sort
            preserves corpus order for ties.

        Raises:
            ValueError: If the model returns a query vector with an incompatible
                shape.
        """
        started = perf_counter()
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
        results = [
            self.chunks[int(position)].model_copy(
                update={"rank": rank, "score": float(scores[position])}
            )
            for rank, position in enumerate(order, start=1)
        ]
        logger.debug(
            "Semantic search completed top_k=%s returned=%s elapsed_ms=%.1f",
            top_k,
            len(results),
            (perf_counter() - started) * 1000,
        )
        return results
