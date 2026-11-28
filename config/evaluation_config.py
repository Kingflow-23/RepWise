"""Settings shared by the retrieval, generation and red-team evaluation suites.

Models, Ollama URL, default retriever, embedding settings and ``top_k`` come
from ``config.config.AppConfig`` so evaluation runs the application's setup.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CHUNKS_DIR = REPO / "data" / "corpus" / "chunks"  # built by the ingestion pipeline
OUT_DIR = REPO / "outputs" / "eval_runs"  # one sub-folder per suite run

# Retrieval
RETRIEVERS = ("bm25", "semantic", "hybrid")  # strategies compared side by side
K_VALUES = (1, 3, 5, 10)  # Recall@k and Precision@k cutoffs
MAIN_K = 10  # retrievers are ranked by Recall@MAIN_K, then MRR, then latency
RETRIEVAL_SUCCESS_K = 5  # a question succeeds when evidence is in the top 5
RETRIEVAL_DEPTH = 50  # how many chunks each retriever returns

# Generation and red team
# The datasets only hold questions, so every request is answered for this
# fixed student profile (the red-team cases refer to dumbbells and 45 minutes).
EVAL_PROFILE = {
    "experience": "beginner",
    "goal": "Build general strength and muscle safely",
    "available_days": ["Monday", "Wednesday", "Friday"],
    "session_minutes": 45,
    "equipment": ["dumbbells"],
}
