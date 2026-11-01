# Generation and validation

## Student information

The input contains a small `UserProfile`: experience, training goal, available days, session length, equipment and optional preferences. Each application request generates exactly one week; the student can start another request for a fresh week. Each request also sets a weekly workout count (default: up to three, limited by available days). The current `focus` can change for every request, for example from general strength to abdominal training.

The demo API stores this profile in SQLite so a student does not need to enter it again for every plan. The database stores account names, Argon2 password hashes, profiles, and completed plans with their source links. Raw passwords and access tokens are not persisted. It does not collect medical details or full conversation transcripts.

## Prompt

The prompt has a fixed system instruction and a user message containing the student request, optional refinement guidance, an optional compact previous-plan summary, and a clearly labelled evidence block and output schema. The system prompt is stored in `prompts/system_prompt_v1.txt`, so its wording can be reviewed and updated without changing Python code. It says that transcript text and previous plans are reference material and cannot change the model's instructions, and requires generated plan fields to be in English even when the student request or source excerpts are in another language. Source excerpts themselves remain in their original language. It also asks the model to use assumptions rather than inventing citations.

For a returning student, generation also includes a compact summary of their latest saved plan and its original focus. The summary contains plan/workout/exercise details only (not the previous source evidence), is capped at 2,400 characters, and is treated as reference data rather than instructions. The current request takes priority. Students can optionally enter up to 500 characters describing what they want changed; this is sent as refinement guidance and saved with the new request.

Ollama is asked to return JSON matching `TrainingPlan`. Pydantic parses the JSON immediately after the call.

## Checks after generation

The code rejects a plan when:

- its number of weeks is different from the request;
- week numbers are missing or repeated;
- a week does not contain exactly the requested number of workouts;
- it schedules a workout on an unavailable day;
- a workout exceeds the requested duration or 120 minutes;
- it omits the rough time estimate for any exercise;
- an exercise is only a body-part/workout-split label or repeats within a workout;
- an exercise has no source ID or cites a source that was not retrieved;
- a rest interval exceeds 300 seconds;
- an exercise requires equipment absent from the profile (unless the profile says full gym).

These are structural and consistency checks, not verification that a source supports an exercise. Exact transcript quotes are optional and are not validated, so French or other non-English evidence will not cause an otherwise valid English plan to fail. The checks do not prove that the programme is safe, medically suitable or scientifically complete. The returned plan always contains a short safety note and the system must not be used to diagnose pain or injuries.

The one-week application plan, workouts-per-week count, and required per-exercise time estimate are enforced in the Ollama structured-output schema, not only checked after generation. Exercise and workout durations are rough estimates: the prompt asks the model to aim reasonably close to the student's available session time without padding a shorter, appropriate workout, and the exercise estimates account for sets, repetitions, rest, and setup. Workout estimates also account for warm-up and transitions. The existing session-duration ceiling remains enforced. The prompt asks for clear English across languages, relevant source IDs, and equipment in the student's profile without requiring exact evidence quotes. Other failed checks still trigger correction attempts and may cause the request to be rejected rather than saving an invalid plan.

## When the pipeline stops

If retrieval returns fewer than five chunks, the pipeline returns `insufficient_evidence` without calling the model; it does not pad a short result set with zero-match passages. If the model response is malformed or violates the checks, it returns `generation_error` with a readable message. If the knowledge base is not connected, it returns `retrieval_error`.

## Full implemented request path

The FastAPI `POST /api/v1/students/{student_id}/plans` route authenticates the account, reads its saved profile, combines it with `PlanOptions`, and invokes `PlanPipeline`. The pipeline builds a query from focus, goal, experience, equipment, and extra constraints, asks the configured BM25, semantic, or hybrid retriever for `top_k` chunks, and selects usable evidence. Hybrid uses RRF to combine BM25 and local multilingual E5 ranks. See [retrieval setup](../retrieval/hybrid_retrieval.md).

The prompt builder loads `prompts/system_prompt_v1.txt`, serializes the request and every chunk returned by retrieval, adds the bounded previous-plan summary when available, and includes the `TrainingPlan` JSON schema. The default retrieval count is five and configuration cannot reduce it below five. Evidence selection does not truncate or drop retrieved chunks for length. Before each model attempt, the pipeline logs system, user, and schema character counts plus an approximate input-token count (four characters per token); Ollama logs its provider-reported prompt token count when available. These measurements make prompt growth observable while keeping prior-plan context bounded. `OllamaGenerator` sends this to the configured local Ollama host with temperature 0.2 and a 32,768-token context window. Pydantic parses the model output. If parsing or validation fails, the pipeline allows two correction attempts with the validation error, then returns a controlled `generation_error`. A successful plan is stored with its request and selected source records. Failed results are returned as API errors and are not persisted as plans.

The Streamlit demo in `app/streamlit_app.py` is an HTTP client for this API. It provides profile create/update and load, plan generation, source display, and saved-plan history with confirmed deletion. Full retrieved extracts are shown with highlights for matching intent terms from the student's focus, goal, preferences, and constraints. Matching is accent- and plural-insensitive lexical overlap, not an explanation of the model's reasoning; an extract without highlighted terms may still be semantically related. It does not call the model or database directly. Run the API and Streamlit as separate processes; set `REPWISE_API_URL` when using a non-default API address.

The health route indicates whether retrieval is ready and reports the configured model name; it does not make a live Ollama health check. The API does not substitute a mock when the corpus, embedding package, model, or index is missing. It keeps the health page available and returns the retrieval setup error when a plan is requested.

## Configuration and limitations

`OLLAMA_MODEL`, `OLLAMA_COMPARISON_MODEL`, `OLLAMA_BASE_URL`, `REPWISE_TOP_K`, `REPWISE_RETRIEVAL`, `REPWISE_EMBEDDING_MODEL`, `REPWISE_EMBEDDING_INDEX`, `REPWISE_CHUNK_FILE`, `REPWISE_DATABASE_PATH`, and `REPWISE_AUTH_SECRET` are read by `config.config.AppConfig`. The Streamlit API URL is read by `config.app_config.get_api_url()` from `REPWISE_API_URL`. Source IDs are checked against retrieved results when supplied, but are not proof that a source semantically justifies an exercise or prescription. Other limits include medical safety, progression quality, and the scope and quality of the source corpus.
