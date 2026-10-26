"""Settings of an evaluation run.

Change the values here instead of passing options on the command line.
The benchmark script reuses the same settings.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Systems under test
RETRIEVER = "bm25"        # "bm25" baseline, or "module.path:function" once the retrieval pipeline exists
PLANNER = "mock-naive"    # "mock-naive", "mock-abstain", or "module.path:function" for the real generator

# Data
CHUNKS_DIR = REPO / "data" / "corpus" / "chunks"   # chunks built by the ingestion pipeline
SPLIT = "all"             # "dev" while tuning, "test" for the results we report, "all" for both

# Retrieval scoring
K_VALUES = (1, 3, 5, 10)  # Recall@k, Precision@k are computed at these k
MAIN_K = 5                # a retrieval case passes when all its passages are in the top 5
RETRIEVAL_DEPTH = 50      # how many chunks we ask the retriever for

# Output
OUT_DIR = REPO / "outputs" / "eval_runs"
FAIL_ON_GATES = False     # True: the command returns an error code when a release gate fails

# Benchmark (python -m src.evaluation.benchmark)
# Each entry is one system: a retriever and a planner, written like RETRIEVER and PLANNER above.
# Add one line per model to compare once the retrieval and generation pipelines exist.
BENCHMARK = [
    {"name": "bm25 + mock-naive", "retriever": "bm25", "planner": "mock-naive"},
    {"name": "bm25 + mock-abstain", "retriever": "bm25", "planner": "mock-abstain"},
]
BENCHMARK_SUITE = "all"   # which suites each system is evaluated on
# Ranking order: the table is sorted on the first metric, ties are broken by the next ones.
# Each entry is (suite, metric); every metric listed here is "higher is better".
BENCHMARK_RANK_BY = [("retrieval", "recall@5"), ("generation", "faithfulness"), ("generation", "pass_rate")]
BENCHMARK_OUT_DIR = REPO / "outputs" / "benchmark_runs"
