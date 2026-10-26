# Generation and validation

## Student information

The input contains a small `UserProfile`: experience, training goal, available days, session length, equipment and optional preferences. The student provides this information deliberately. The current `focus` can change for every request, for example from general strength to abdominal training.

The demo API stores this profile in SQLite so a student does not need to enter it again for every plan. The database stores only the profile and completed plans with their source links. It does not collect medical details, credentials or full conversation transcripts.

## Prompt

The prompt has three visible parts: a fixed system instruction, the student request as JSON, and a clearly labelled evidence block. The system prompt is stored in `prompts/system_prompt_v1.txt`, so its wording can be reviewed and updated without changing Python code. It says that transcript text is reference material and cannot change the model's instructions. It also asks the model to use assumptions rather than inventing citations.

Ollama is asked to return JSON matching `TrainingPlan`. Pydantic parses the JSON immediately after the call.

## Checks after generation

The code rejects a plan when:

- its number of weeks is different from the request;
- week numbers are missing or repeated;
- it uses more workouts than the student has available days;
- it schedules a workout on an unavailable day;
- a workout exceeds the requested duration;
- an exercise has no source ID or refers to a source that was not retrieved;
- an exercise requires equipment absent from the profile (unless the profile says full gym).

These are simple consistency checks. They do not prove that the programme is safe, medically suitable or scientifically complete. The returned plan always contains a short safety note and the system must not be used to diagnose pain or injuries.

## When the pipeline stops

If retrieval returns no usable evidence, the pipeline returns `insufficient_evidence` and does not ask the model to invent a plan. If the model response is malformed or violates the checks, it returns `generation_error` with a readable message. If the knowledge base is not connected, it returns `retrieval_error`.

## Full implemented request path

The FastAPI `POST /api/v1/students/{student_id}/plans` route authenticates the account, reads its saved profile, combines it with `PlanOptions`, and invokes `PlanPipeline`. The pipeline builds a query from focus, goal, experience, equipment, and extra constraints, asks the configured BM25, semantic, or hybrid retriever for `top_k` chunks, and selects usable evidence. Hybrid uses RRF to combine BM25 and local multilingual E5 ranks. See [retrieval setup](../retrieval/hybrid_retrieval.md).

The prompt builder loads `prompts/system_prompt_v1.txt`, serializes the request and selected evidence, and includes the `TrainingPlan` JSON schema. `OllamaGenerator` sends this to the configured local Ollama host with temperature 0.2. Pydantic parses the model output. If parsing or validation fails, the pipeline makes one correction attempt with the validation error, then returns a controlled `generation_error`. A successful plan is stored with its request and selected source records. Failed results are returned as API errors and are not persisted as plans.

The Streamlit demo in `app/streamlit_app.py` is an HTTP client for this API. It provides profile create/update and load, plan generation, source display, and saved-plan history. It does not call the model or database directly. Run the API and Streamlit as separate processes; set `REPWISE_API_URL` when using a non-default API address.

The health route indicates whether retrieval is ready and reports the configured model name; it does not make a live Ollama health check. The API does not substitute a mock when the corpus, embedding package, model, or index is missing. It keeps the health page available and returns the retrieval setup error when a plan is requested.

## Configuration and limitations

`OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `REPWISE_TOP_K`, `REPWISE_RETRIEVAL`, `REPWISE_EMBEDDING_MODEL`, `REPWISE_EMBEDDING_INDEX`, `REPWISE_CHUNK_FILE`, `REPWISE_DATABASE_PATH`, and `REPWISE_AUTH_SECRET` are read by `config.config.AppConfig`. The Streamlit API URL is read by `config.app_config.get_api_url()` from `REPWISE_API_URL`. Citation-ID validation proves that an exercise points to evidence supplied to the model; it does not prove that the passage semantically supports the recommendation. Other limits include medical safety, progression quality, and the scope and quality of the source corpus.
