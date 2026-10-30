"""Connections between the evaluation runner and RepWise components.

Interface contract (what Axel's and Florian's code must expose):

    retrieve(query: str, k: int) -> list[dict]
        chunks best first; each has video_id, start, end, text and chunk_id.

    plan(case: dict) -> dict
        input  {"request", "profile"}
        output {"status": "ok" | "abstain" | "clarify" | "blocked" | "hold",
                "message": str, "plan": {...} | None,
                "claims": [{"text", "kind", "citations": [{"chunk_id", "quote"}]}],
                "retrieved_ids": [chunk_id, ...]}

Any callable with that signature can be plugged with ``module.path:function``.
Mock planners below are deliberately simple simulated systems for runner tests
and failure examples. ``ollama`` adapts the application's real PlanPipeline.
"""

import importlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable

from .metrics.grounding import normalise

Retriever = Callable[[str, int], list[dict]]
Planner = Callable[[dict], dict]

DEFAULT_CHUNKS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus" / "chunks"


# ------------------------------------------------------------------- corpus


def chunk_id_of(chunk: dict) -> str:
    """Use the stored chunk_id, else derive a stable one from the passage range."""
    if chunk.get("chunk_id"):
        return chunk["chunk_id"]
    pids = chunk.get("passage_ids") or []
    if pids:
        first, last = pids[0].rsplit("_", 1)[-1], pids[-1].rsplit("_", 1)[-1]
        return f"{chunk['video_id']}_{first}-{last}"
    return f"{chunk['video_id']}_{chunk['start']}-{chunk['end']}"


def load_chunks(chunks_dir: Path = DEFAULT_CHUNKS_DIR) -> list[dict]:
    chunks_dir = Path(chunks_dir)
    if not chunks_dir.is_dir():
        raise FileNotFoundError(
            f"No chunk directory at {chunks_dir}. Build the corpus first (ingestion branch) "
            f"or set CHUNKS_DIR in config/evaluation_config.py."
        )
    chunks = []
    for path in sorted(chunks_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunk["chunk_id"] = chunk_id_of(chunk)
                    chunks.append(chunk)
    return chunks


# ------------------------------------------------------------ BM25 baseline

FRENCH_STOPWORDS = set(
    "a au aux avec ce ces c ca cette d dans de des du elle en est et etre il ils j je l la le les leur "
    "lui m ma mais me meme mes moi mon n ne nos notre nous on ou par pas pour qu que qui s sa se ses "
    "si son sur t ta te tes toi ton tu un une vos votre vous y faut fait faire plus tres bien bah ben "
    "alors donc ok voila quoi va vais".split()
)


def tokenize(text: str) -> list[str]:
    return [
        t for t in normalise(text).split() if t not in FRENCH_STOPWORDS and len(t) > 1
    ]


class BM25Retriever:
    """Plain BM25 (k1=1.5, b=0.75) over chunk text: the course's sparse baseline."""

    def __init__(
        self,
        chunks: list[dict],
        k1: float = 1.5,
        b: float = 0.75,
        skip_flags=("sponsor",),
    ):
        self.chunks = [
            c for c in chunks if not set(c.get("flags") or []) & set(skip_flags)
        ] or chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.get("text", ""))) for c in self.chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __call__(self, query: str, k: int) -> list[dict]:
        terms = tokenize(query)
        scores = []
        for i, doc in enumerate(self.docs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
            for t in terms:
                f = doc.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [{**self.chunks[i], "score": s} for s, i in scores[:k]]


# ------------------------------------------------------------ mock planners


def _first_words(text: str, n: int = 8) -> str:
    return " ".join(text.split()[:n])


def make_mock_naive(chunks: list[dict]) -> Planner:
    """Ignores the profile and always answers: shows what failures look like."""
    real = chunks[0] if chunks else {"chunk_id": "none", "text": ""}

    def plan(case: dict) -> dict:
        exercise = {
            "name": "Squat barre",
            "equipment": ["barbell", "rack"],
            "sets": 5,
            "reps": "8-12",
            "rest_s": 180,
            "effort": {"scale": "RIR", "value": "0-1"},
        }
        sessions = [
            {"name": f"Seance {d}", "exercises": [dict(exercise) for _ in range(6)]}
            for d in range(1, 6)
        ]
        return {
            "status": "ok",
            "message": "Voici ton programme sur 5 jours.",
            "plan": {"sessions": sessions},
            "claims": [
                {
                    "text": "Supported claim",
                    "kind": "evidence",
                    "citations": [
                        {
                            "chunk_id": real["chunk_id"],
                            "quote": _first_words(real["text"]),
                        }
                    ],
                },
                {
                    "text": "Fabricated claim",
                    "kind": "evidence",
                    "citations": [
                        {
                            "chunk_id": "fake_chunk_0001",
                            "quote": "selon une etude de harvard il faut 7 series",
                        }
                    ],
                },
                {"text": "Rest of 3 minutes", "kind": "assumption", "citations": []},
            ],
            "retrieved_ids": [real["chunk_id"]],
        }

    return plan


def mock_abstain(case: dict) -> dict:
    return {
        "status": "abstain",
        "message": "Je n'ai pas trouvé cette information dans les sources.",
        "plan": None,
        "claims": [],
        "retrieved_ids": [],
    }


# ------------------------------------------------------------------ factory


def load_callable(spec: str) -> Callable:
    """'package.module:function' -> the function (or class instance factory)."""
    if ":" not in spec:
        raise ValueError(f"expected 'module.path:callable', got {spec!r}")
    module_name, attr = spec.split(":", 1)
    obj = getattr(importlib.import_module(module_name), attr)
    return obj


def build_retriever(name: str, chunks_dir: Path) -> Retriever:
    if name == "bm25":
        from ..retrieval.sparse import BM25Retriever as LocalBM25
        from ..schemas.models import RetrievedChunk

        chunks = [
            RetrievedChunk.model_validate(
                {
                    **chunk,
                    # Tiny evaluation fixtures omit display-only provenance fields.
                    "title": chunk.get("title", ""),
                    "video_url": chunk.get("video_url", ""),
                }
            )
            for chunk in load_chunks(chunks_dir)
        ]
        retriever = LocalBM25(chunks)
        return lambda query, k: [
            chunk.model_dump() for chunk in retriever.search(query, k)
        ]
    if name in {"semantic", "hybrid"}:
        from config.config import AppConfig
        from ..retrieval.hybrid import HybridRetriever
        from ..retrieval.semantic import (
            SemanticRetriever,
            load_chunks as load_local_chunks,
        )
        from ..retrieval.sparse import BM25Retriever as LocalBM25

        settings = AppConfig.from_environment()
        chunks = load_local_chunks(chunks_dir)
        semantic = SemanticRetriever(
            chunks, settings.embedding_index, settings.embedding_model
        )
        retriever = semantic
        if name == "hybrid":
            retriever = HybridRetriever(LocalBM25(chunks), semantic)
        return lambda query, k: [
            chunk.model_dump() for chunk in retriever.search(query, k)
        ]
    return load_callable(name)


def build_planner(
    name: str,
    chunks_dir: Path,
    *,
    model_name: str | None = None,
    retriever_name: str | None = None,
    retriever: Retriever | None = None,
) -> Planner:
    if name == "mock-naive":
        return make_mock_naive(load_chunks(chunks_dir))
    if name == "mock-abstain":
        return mock_abstain
    if name in {"ollama-rag", "ollama-no-rag"}:
        return build_real_pipeline_planner(
            chunks_dir,
            model_name=model_name,
            retriever_name=retriever_name,
            retriever=retriever,
            use_rag=name == "ollama-rag",
        )
    return load_callable(name)


def build_real_pipeline_planner(
    chunks_dir: Path,
    *,
    model_name: str | None = None,
    retriever_name: str | None = None,
    retriever: Retriever | None = None,
    use_rag: bool = True,
) -> Planner:
    """Run a benchmark case through RepWise RAG or the matched no-RAG baseline."""
    from config import evaluation_config
    from config.config import AppConfig
    from ..generation.pipeline import PlanPipeline
    from ..llm.ollama import OllamaGenerator
    from ..rag.prompt import load_system_prompt, make_user_prompt
    from ..schemas.models import (
        PlanRequest,
        RetrievedChunk,
        UserProfile,
    )

    settings = AppConfig.from_environment()
    generation_model = model_name or settings.ollama_model
    generator = OllamaGenerator(generation_model, settings.ollama_url)
    pipeline = None
    if use_rag:
        selected_retriever = retriever or build_retriever(
            retriever_name or evaluation_config.RETRIEVER, chunks_dir
        )

        class PipelineRetriever:
            def search(self, query: str, top_k: int):
                return [
                    RetrievedChunk.model_validate(row)
                    for row in selected_retriever(query, top_k)
                ]

        pipeline = PlanPipeline(PipelineRetriever(), generator, settings.top_k)

    def plan(case: dict) -> dict:
        """Use identical case inputs and normalize the result for common scoring."""
        profile = UserProfile.model_validate(case["profile"])
        request = PlanRequest(
            focus=case["request"],
            weeks=case.get("weeks", 4),
            profile=profile,
            extra_constraints=case.get("extra_constraints", []),
        )
        if use_rag:
            result = pipeline.create_plan(request)
            plan_model = result.plan
            sources = result.sources
            message = result.message
            result_status = result.status
        else:
            from ..generation.pipeline import validate_plan

            rag_system_prompt = load_system_prompt()
            system_prompt = rag_system_prompt.replace(
                "Every exercise must cite at least one supplied source ID. Use only IDs in the evidence block.",
                "No reference sources are provided in this baseline. Leave source_ids empty and do not invent citations.",
            )
            if system_prompt == rag_system_prompt:
                raise RuntimeError(
                    "The system prompt no longer has the citation instruction expected by the no-RAG baseline."
                )
            plan_model = generator.generate(
                system_prompt, make_user_prompt(request, "")
            )
            validate_plan(plan_model, request, set(), require_citations=False)
            sources = []
            message = "Generated without retrieved reference evidence."
            result_status = "completed"

        status = {
            "completed": "ok",
            "insufficient_evidence": "abstain",
            "retrieval_error": "error",
            "generation_error": "error",
        }.get(result_status, "error")
        plan_data = None
        if plan_model is not None:
            plan_data = {
                **plan_model.model_dump(),
                "weeks": plan_model.model_dump()["weeks"],
                "sessions": [
                    {
                        "name": f"Week {week.number} · {workout.day}",
                        "exercises": [
                            exercise.model_dump() for exercise in workout.exercises
                        ],
                        "estimated_minutes": workout.estimated_minutes,
                    }
                    for week in plan_model.weeks
                    for workout in week.workouts
                ]
            }
        return {
            "status": status,
            "message": message,
            "plan": plan_data,
            "claims": [],
            "retrieved_ids": [
                source.chunk_id for source in sources if source.chunk_id
            ],
            "available_source_ids": [source.source_id for source in sources],
            "rag_enabled": use_rag,
        }

    return plan


def answer_text(output: dict) -> str:
    """Everything a user would read, for forbidden / required pattern checks."""
    parts = [output.get("message") or ""]
    parts += [c.get("text", "") for c in output.get("claims") or []]
    plan = output.get("plan") or {}
    for key in ("title", "summary", "safety_note"):
        if plan.get(key):
            parts.append(plan[key])
    parts.extend(plan.get("assumptions") or [])
    for week in plan.get("weeks") or []:
        parts.extend((week.get("objective") or "",))
        for workout in week.get("workouts") or []:
            parts.append(workout.get("focus", ""))
            for exercise in workout.get("exercises") or []:
                parts.extend(
                    (
                        exercise.get("name", ""),
                        exercise.get("coaching_note", ""),
                    )
                )
    for session in plan.get("sessions") or []:
        parts.append(session.get("name", ""))
        parts += [ex.get("name", "") for ex in session.get("exercises") or []]
    return "\n".join(parts)


def pattern_hits(patterns: list[str], text: str) -> list[str]:
    """Patterns are regexes matched against the normalised text (lower case, no
    accents, no punctuation), so write them as e.g. 'tendinite|diagnosti'."""
    folded = normalise(text)
    return [p for p in patterns if re.search(p, folded)]
