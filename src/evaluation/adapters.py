"""Adapt evaluation inputs and production components to common runner contracts.

The evaluation runner works against two small callable interfaces instead of
depending directly on a particular retriever or planner implementation:

* A retriever accepts ``(query, k)`` and returns ranked chunk dictionaries,
  best match first. Ranking checks need stable chunk IDs and source time ranges;
  the production generation path additionally validates each row as a
  :class:`RetrievedChunk`.
* A planner accepts a case dictionary containing ``id``, ``request``,
  ``profile``, ``weeks``, ``workouts_per_week`` and ``extra_constraints``.
  It returns a result dictionary with ``status``, ``message``, ``plan``,
  ``claims`` and ``retrieved_ids``. Real pipeline adapters also expose the
  available citation IDs/evidence and whether RAG is enabled, which the
  generation evaluator uses to validate citations and constraints.

The flow is ``runner -> adapter -> corpus or production component -> normalized
dictionary -> metrics``. ``build_retrievers`` delegates ranking to production
BM25, semantic and hybrid implementations; it contains no separate ranking
algorithm. ``build_real_pipeline_planner`` wraps the production plan pipeline,
or runs its matched no-RAG baseline. Mock planners intentionally return fixed
examples to exercise scoring behavior; they are not substitutes for model
quality evaluation.

Custom implementations can be loaded by trusted configuration as
``package.module:callable``. The callable is imported and invoked in-process,
so this extension point is for project-controlled code, not user-supplied
specifications. Application imports are kept inside builders so that using
corpus helpers or mock planners does not eagerly initialize retrieval models
or the generation stack.
"""

import importlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from .metrics.grounding import normalise

Chunk = dict[str, Any]
PlannerCase = dict[str, Any]
PlannerResult = dict[str, Any]
Retriever = Callable[[str, int], list[Chunk]]
Planner = Callable[[PlannerCase], PlannerResult]

# Resolve from this module so the default follows the repository checkout,
# while callers can override it for isolated corpora and tests.
DEFAULT_CHUNKS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus" / "chunks"


# ------------------------------------------------------------------- corpus


def chunk_id_of(chunk: Chunk) -> str:
    """Return a chunk's stored ID or derive one from its source range.

    Args:
        chunk: A JSONL chunk record. It should contain ``video_id`` and either
            ``chunk_id``, ``passage_ids``, or ``start``/``end``.

    Returns:
        The stored identifier when present; otherwise an identifier derived
        from the first/last passage IDs or the chunk's time range. This fallback
        keeps older corpus records usable without changing production IDs.
    """
    if chunk.get("chunk_id"):
        return chunk["chunk_id"]
    pids = chunk.get("passage_ids") or []
    if pids:
        first, last = pids[0].rsplit("_", 1)[-1], pids[-1].rsplit("_", 1)[-1]
        return f"{chunk['video_id']}_{first}-{last}"
    return f"{chunk['video_id']}_{chunk['start']}-{chunk['end']}"


def load_chunks(chunks_dir: Path = DEFAULT_CHUNKS_DIR) -> list[Chunk]:
    """Read JSONL chunk records and ensure that each has a stable chunk ID.

    Args:
        chunks_dir: Directory containing the corpus' ``*.jsonl`` chunk files.
            The default points to this checkout's generated corpus; an
            override supports alternate corpora and isolated tests.

    Returns:
        All non-empty records, ordered by filename and then file order.

    Raises:
        FileNotFoundError: If the directory is missing or contains no chunk
            records, since an empty corpus would make evaluation misleading.
        json.JSONDecodeError: If a non-empty line is not valid JSON.
    """
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
    if not chunks:
        raise FileNotFoundError(
            f"No JSONL chunks found in {chunks_dir}. Build the corpus first."
        )
    return chunks


# ------------------------------------------------------------ mock planners


def _first_words(text: str) -> str:
    """Return a short deterministic quote for the mock planner's citation.

    Args:
        text: Source text to quote.

    Returns:
        The first eight whitespace-separated words joined by single spaces.
        Eight is enough to make this fixture citation readable without adding
        tokenization logic or a configuration parameter used nowhere else.
    """
    return " ".join(text.split()[:8])


def make_mock_naive(chunks: list[Chunk]) -> Planner:
    """Create a deterministic deliberately poor planner fixture.

    Args:
        chunks: Corpus records from which the fixture can make one nominally
            supported citation. An empty list uses a sentinel source instead,
            so the mock can still demonstrate invalid-citation behavior.

    Returns:
        A planner callable that ignores its case and returns a fixed,
        intentionally non-personalized plan with both supported and fabricated
        claims. It is a scoring/test fixture, not a baseline for model quality.
    """
    real = chunks[0] if chunks else {"chunk_id": "none", "text": ""}

    def plan(_case: PlannerCase) -> PlannerResult:
        """Return the same intentionally flawed response for any case.

        Args:
            _case: Required by the planner interface; deliberately ignored
                because this fixture is designed to expose constraint failures.

        Returns:
            The fixed planner result consumed by generation metrics.
        """
        del _case
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


def mock_abstain(_case: PlannerCase) -> PlannerResult:
    """Return a deterministic abstention for planner-interface tests.

    Args:
        _case: Required by the planner interface, but unused because this
            fixture abstains independently of request content.

    Returns:
        A result with no plan, claims, or retrieved chunk IDs.
    """
    del _case
    return {
        "status": "abstain",
        "message": "Je n'ai pas trouvé cette information dans les sources.",
        "plan": None,
        "claims": [],
        "retrieved_ids": [],
    }


# ------------------------------------------------------------------ factory


def load_callable(spec: str) -> Callable[..., Any]:
    """Import a project-controlled callable using ``module.path:name`` syntax.

    Args:
        spec: Fully qualified module and attribute separated by one colon.
            The imported object is returned without invocation; constructors
            that need arguments should be wrapped in a project-owned function.

    Returns:
        The named module attribute. Its suitability for a retriever or planner
        contract is checked when the runner invokes it.

    Raises:
        ValueError: If ``spec`` has no colon.
        ImportError: If the module cannot be imported.
        AttributeError: If the named attribute is absent.
    """
    if ":" not in spec:
        raise ValueError(f"expected 'module.path:callable', got {spec!r}")
    module_name, attr = spec.split(":", 1)
    obj = getattr(importlib.import_module(module_name), attr)
    return obj


def build_retriever(name: str, chunks_dir: Path) -> Retriever:
    """Build one named production or configured custom retriever.

    Args:
        name: ``bm25``, ``semantic`` or ``hybrid`` for production strategies;
            otherwise a trusted ``module.path:callable`` plugin specification.
        chunks_dir: Corpus directory used to initialize a built-in strategy.
            Custom callables own their own data access and do not receive it.

    Returns:
        A ``(query, k) -> ranked chunk dictionaries`` callable.

    Raises:
        ValueError: If ``name`` is neither a supported built-in strategy nor a
            valid ``module.path:callable`` specification.
        FileNotFoundError: If a built-in strategy needs a missing/empty corpus.
        ImportError: If a custom callable's module cannot be imported.
    """
    if name in {"bm25", "semantic", "hybrid"}:
        return build_retrievers([name], chunks_dir)[name]
    return load_callable(name)


def build_retrievers(
    names: Sequence[str],
    chunks_dir: Path,
    *,
    chunks: Sequence[Chunk] | None = None,
) -> dict[str, Retriever]:
    """Build requested production retrievers, sharing common components.

    Args:
        names: Requested strategies from ``bm25``, ``semantic`` and ``hybrid``.
            A sequence preserves the requested order in the returned mapping,
            whose keys are strategy names.
        chunks_dir: Default corpus source, used only when ``chunks`` is not
            supplied.
        chunks: Optional already-loaded corpus records. Supplying these avoids
            rereading JSONL files, which lets comparison runs share one corpus.
            Records are still validated as ``RetrievedChunk`` objects before
            being passed to production retrieval implementations.

    Returns:
        A mapping from each requested strategy to a callable that serializes
        production ``RetrievedChunk`` results as dictionaries. An empty
        ``names`` sequence returns an empty mapping without loading the corpus.

    Raises:
        ValueError: If any requested strategy is unsupported.
        FileNotFoundError: If records must be loaded but the corpus is missing
            or empty.
        ValidationError: If a supplied or loaded record violates the production
            ``RetrievedChunk`` schema.
    """
    supported = {"bm25", "semantic", "hybrid"}
    unknown = set(names) - supported
    if unknown:
        raise ValueError(f"Unknown retrieval strategies: {', '.join(sorted(unknown))}")
    if not names:
        return {}

    from ..retrieval.hybrid import HybridRetriever
    from ..retrieval.semantic import SemanticRetriever
    from ..retrieval.sparse import BM25Retriever
    from ..schemas.models import RetrievedChunk

    chunk_records = [
        RetrievedChunk.model_validate(
            {
                **chunk,
                # Tiny evaluation fixtures omit display-only provenance fields.
                "title": chunk.get("title", ""),
                "video_url": chunk.get("video_url", ""),
            }
        )
        for chunk in (load_chunks(chunks_dir) if chunks is None else chunks)
    ]
    lexical = (
        BM25Retriever(chunk_records) if {"bm25", "hybrid"} & set(names) else None
    )
    semantic = None
    if {"semantic", "hybrid"} & set(names):
        from config.config import AppConfig

        settings = AppConfig.from_environment()
        semantic = SemanticRetriever(
            chunk_records, settings.embedding_index, settings.embedding_model
        )

    components = {
        "bm25": lexical,
        "semantic": semantic,
        "hybrid": (
            HybridRetriever(lexical, semantic)
            if lexical is not None and semantic is not None
            else None
        ),
    }

    def adapt(retriever: Any) -> Retriever:
        """Bridge the production ``search`` object to the runner callable.

        Args:
            retriever: Production retriever exposing ``search(query, k)``.

        Returns:
            A callable that serializes model objects while preserving ranking
            order and all fields used by evaluation.
        """
        return lambda query, k: [
            chunk.model_dump() for chunk in retriever.search(query, k)
        ]

    return {name: adapt(components[name]) for name in names}


def build_planner(
    name: str,
    chunks_dir: Path,
    *,
    model_name: str | None = None,
    retriever_name: str | None = None,
    retriever: Retriever | None = None,
) -> Planner:
    """Resolve a planner name to a mock, production, or configured callable.

    Args:
        name: ``mock-naive``, ``mock-abstain``, ``ollama-rag``,
            ``ollama-no-rag`` or a trusted ``module.path:callable`` plugin.
        chunks_dir: Corpus directory for the naive mock and real RAG planner.
            It is not read by the abstaining mock or custom plugins.
        model_name: Optional Ollama model override for the real planner.
            ``None`` uses the application's configured default.
        retriever_name: Optional built-in or plugin retriever selection for
            real RAG. ``None`` uses the evaluation configuration's strategy.
        retriever: Optional prebuilt retriever. When provided to a RAG planner,
            it takes precedence over ``retriever_name`` so comparison arms can
            share the same fitted index and ranking configuration. Retriever-
            related options do not affect mocks, plugins, or no-RAG mode.

    Returns:
        A callable accepting the common planner case dictionary and returning
        the common planner result dictionary.
    """
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
    """Adapt RepWise plan generation to the evaluation planner contract.

    The RAG path creates the real ``PlanPipeline`` and validates retrieved
    dictionaries as production ``RetrievedChunk`` models. The no-RAG path uses
    the same request schema, system prompt, response schema and plan validator,
    but removes the prompt's evidence-citation requirement and supplies no
    retrieved context. Thus it isolates the effect of retrieved evidence
    without substituting a different planner implementation.

    Args:
        chunks_dir: Corpus directory used when this function must build its own
            retriever. It is not read by the no-RAG path or when ``retriever``
            is injected.
        model_name: Optional Ollama model override; ``None`` uses
            ``AppConfig.ollama_model``.
        retriever_name: Retriever name used only when RAG is enabled and no
            prebuilt ``retriever`` is supplied; ``None`` uses evaluation
            configuration.
        retriever: Optional prebuilt ``(query, k)`` retriever. Injection lets
            comparison arms share one retrieval instance. It takes precedence
            over ``retriever_name``.
        use_rag: Selects retrieval-grounded generation when true and the
            matched no-context baseline when false.

    Returns:
        A planner callable that maps each case to the runner's normalized result
        shape. Result source IDs are citation IDs; ``retrieved_ids`` are chunk
        IDs, kept separately because the production pipeline distinguishes
        sources from corpus chunks.

    Raises:
        FileNotFoundError: If the RAG path must load a missing/empty corpus.
        RuntimeError: If the system prompt no longer contains the citation
            instruction that the no-RAG baseline must replace.
    """
    from config import evaluation_config
    from config.config import AppConfig
    from ..generation.pipeline import PlanPipeline
    from ..llm.ollama import OllamaGenerator
    from ..rag.prompt import load_system_prompt, make_user_prompt, plan_response_schema
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
        selected_retriever = (
            retriever
            if retriever is not None
            else build_retriever(
                retriever_name or evaluation_config.RETRIEVER, chunks_dir
            )
        )

        class PipelineRetriever:
            """Adapt the evaluation callable to the production Retriever API."""

            def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
                """Return model-validated chunks in the callable's ranking order.

                Args:
                    query: Query constructed by the production plan pipeline.
                    top_k: Maximum number of chunks requested by that pipeline.

                Returns:
                    Validated production chunk models ready for evidence
                    selection and plan generation.
                """
                return [
                    RetrievedChunk.model_validate(row)
                    for row in selected_retriever(query, top_k)
                ]

        pipeline = PlanPipeline(PipelineRetriever(), generator, settings.top_k)

    def plan(case: PlannerCase) -> PlannerResult:
        """Run one case and normalize its result for the evaluation runner.

        Args:
            case: Runner case with request text, profile and schedule
                constraints. Those fields are passed through to ``PlanRequest``
                so the benchmark evaluates the same request the planner sees.

        Returns:
            A normalized mapping containing status, message, plan, claims,
            retrieved chunk IDs, available citation IDs/evidence, and the RAG
            flag used by downstream schema and citation checks. The plan keeps
            its week structure and also adds flattened ``sessions`` for the
            runner's schedule checks. ``claims`` is empty because production
            generation records evidence citations on exercises, not as the
            claim-level objects used by mock planner fixtures.
        """
        profile = UserProfile.model_validate(case["profile"])
        request = PlanRequest(
            focus=case["request"],
            weeks=case.get("weeks", 1),
            workouts_per_week=case.get("workouts_per_week"),
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

            system_prompt = load_system_prompt(require_evidence=False)
            plan_model = generator.generate(
                system_prompt,
                make_user_prompt(request, "", require_evidence=False),
                plan_response_schema(request),
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
            serialized_plan = plan_model.model_dump()
            plan_data = {
                **serialized_plan,
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
                        ],
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
            "available_source_evidence": {
                source.source_id: source.evidence_text or "" for source in sources
            },
            "rag_enabled": use_rag,
        }

    return plan


def answer_text(output: PlannerResult) -> str:
    """Collect user-visible planner text for required/forbidden checks.

    Args:
        output: Normalized planner result. Both the current week-based
            ``plan.weeks`` shape and the legacy flattened ``plan.sessions``
            shape are read because plugins and earlier fixtures may emit
            either. Non-text metadata such as IDs and scores is intentionally
            excluded from user-visible pattern checks.

    Returns:
        User-facing message, claims, plan prose and exercise names joined by
        newlines for regular-expression matching.
    """
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
    """Return configured regexes that match normalized planner text.

    Args:
        patterns: Regular expressions written for lowercase normalized text;
            for example, ``tendinite|diagnosti`` also matches accented input
            after normalization.
        text: Combined user-visible answer from :func:`answer_text`, or any
            other text to check with the same normalization.

    Returns:
        The original pattern strings that matched, preserving input order.

    Raises:
        re.error: If any supplied expression is invalid. Invalid benchmark
            patterns should surface as configuration errors rather than being
            silently treated as misses.
    """
    folded = normalise(text)
    return [p for p in patterns if re.search(p, folded)]
