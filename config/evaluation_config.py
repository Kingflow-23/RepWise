"""Settings for the unified evaluation runner."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Systems under test
RETRIEVER = "bm25"  # Compare "bm25", "semantic" and "hybrid" on identical labels.
PLANNER = "mock-abstain"  # deterministic placeholder; use --llm ollama for application generation

# Data
CHUNKS_DIR = (
    REPO / "data" / "corpus" / "chunks"
)  # chunks built by the ingestion pipeline

# Retrieval scoring
K_VALUES = (1, 3, 5, 10)  # Recall@k, Precision@k are computed at these k
MAIN_K = 5  # a retrieval case passes when all its passages are in the top 5
RETRIEVAL_DEPTH = 50  # how many chunks we ask the retriever for
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_INDEX = REPO / "data" / "corpus" / "index" / "semantic.npz"

# Output
OUT_DIR = REPO / "outputs" / "eval_runs"
FAIL_ON_GATES = (
    False  # True: the command returns an error code when a release gate fails
)
