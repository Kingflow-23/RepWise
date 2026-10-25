"""Tests of the benchmark script: it runs every listed system and writes a ranking."""

import json

from src.evaluation import benchmark
from config import evaluation_config as config

CHUNKS = [
    {"video_id": "v1", "start": 0, "end": 60, "passage_ids": ["v1_p0001", "v1_p0004"], "flags": [],
     "text": "entre deux et trois minutes de récupération entre chaque série pour un débutant"},
    {"video_id": "v2", "start": 0, "end": 60, "passage_ids": ["v2_p0001", "v2_p0003"], "flags": [],
     "text": "la semaine de deload on réduit le volume de 50 à 70 pour cent en gardant les charges"},
]


def write_chunks(folder):
    folder.mkdir()
    for c in CHUNKS:
        (folder / f"{c['video_id']}.jsonl").write_text(json.dumps(c, ensure_ascii=False) + "\n", encoding="utf-8")
    return folder


def test_rank_puts_best_first_and_missing_values_last(monkeypatch):
    """Systems are sorted on the first ranking metric; a system without that metric (N/A) goes last."""
    monkeypatch.setattr(config, "BENCHMARK_RANK_BY", [("retrieval", "recall@5")])
    rows = [{"name": "a", ("retrieval", "recall@5"): 0.4},
            {"name": "b", ("retrieval", "recall@5"): None},
            {"name": "c", ("retrieval", "recall@5"): 0.9}]
    assert [r["name"] for r in benchmark.rank(rows)] == ["c", "a", "b"]


def test_ties_are_broken_by_the_next_metric(monkeypatch):
    """When two systems have the same first metric, the second metric decides."""
    monkeypatch.setattr(config, "BENCHMARK_RANK_BY", [("retrieval", "recall@5"), ("generation", "pass_rate")])
    rows = [{"name": "a", ("retrieval", "recall@5"): 0.5, ("generation", "pass_rate"): 0.2},
            {"name": "b", ("retrieval", "recall@5"): 0.5, ("generation", "pass_rate"): 0.8}]
    assert [r["name"] for r in benchmark.rank(rows)] == ["b", "a"]


def test_benchmark_runs_every_system_and_writes_the_ranking(tmp_path, monkeypatch):
    """Each system in BENCHMARK gets its own evaluation run, and ranking.md / ranking.json list all of them."""
    monkeypatch.setattr(config, "CHUNKS_DIR", write_chunks(tmp_path / "chunks"))
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "BENCHMARK_OUT_DIR", tmp_path / "bench")
    monkeypatch.setattr(config, "BENCHMARK", [
        {"name": "naive", "retriever": "bm25", "planner": "mock-naive"},
        {"name": "abstain", "retriever": "bm25", "planner": "mock-abstain"},
    ])
    assert benchmark.main() == 0

    assert len(list((tmp_path / "runs").iterdir())) == 2  # one evaluation run per system
    out = next((tmp_path / "bench").iterdir())
    ranking = json.loads((out / "ranking.json").read_text(encoding="utf-8"))
    assert {r["name"] for r in ranking} == {"naive", "abstain"}
    assert "| 1 |" in (out / "ranking.md").read_text(encoding="utf-8")
