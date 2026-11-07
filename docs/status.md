# Implementation status

What works now, what depends on outside services or credentials, and what is
left for the web frontend (`app/`). Last verified on 2026-11-07.

## Implemented and verified

| Area | Status | Verified by |
|---|---|---|
| Six agents (orchestrator + five specialists) | Done | `tests/agents/test_orchestrator.py`, `test_specialists.py`; live `scripts.run_agent` |
| Plan-and-Execute: plan, delegate, check, one replan, bounded steps and time | Done | `tests/agents/test_orchestrator.py` |
| 19 tools, one catalog (`config/tools.json`), consistency check | Done | `tests/tools/test_catalog.py`, MCP startup check |
| Approval gate with pending actions (owner, expiry, hash, single use) | Done | `tests/agents/test_approvals.py`, `tests/app/test_agent_api.py` |
| Draft plans and approval-gated activation | Done | `tests/agents/test_approvals.py`, `tests/tools/test_generate_workout_plan.py` |
| Memory: working, semantic, episodic (90-day retention), no health details | Done | `tests/agents/test_health_and_memory.py` |
| Health check (LLM, English and French, fail closed) replacing the keyword lists | Done | Unit tests with a fake client; live check on `qwen3.5:4b` |
| Shared feedback-adaptation rules for the API and the agent | Done | `tests/app/test_api.py` (existing contract), `tests/tools/test_training_tools.py` |
| MCP server: one entry point, token-bound account, every catalog tool | Done | `tests/tools/test_mcp_server.py`, `python -m scripts.mcp_smoke` |
| Agent, approval and saved-location API routes | Done | `tests/app/test_agent_api.py`, API smoke test with uvicorn |
| Location tools with Photon (geocoding, nearby search, distances), limited to metropolitan France, unrelated places filtered by code | Done | Mocked tests; live checks of French towns, vague names and places abroad |
| Catch-up after missed workouts decided by code, exact moves shown in the approval | Done | `tests/tools/test_training_tools.py`, `tests/agents/test_approvals.py`, live multi-agent run |
| Key facts (available days vs workout days) and places list written by code | Done | `tests/agents/test_orchestrator.py`, live runs |
| Calendar tools (Google adapter, internal booking record, slots with time zones) | Done | Mocked tests, including daylight saving |
| Existing RAG, generation and evaluation behaviour | Preserved | Full suite |
| Streamlit test console (`genai_app/`): profile, plans, feedback, assistant chat with plan timeline, approvals inbox, per-request metrics (phases, model calls, tokens, tool latency) and session export, saved location, calendar, tool catalog, MCP host config | Done | `tests/app/test_streamlit_app.py`, `tests/app/test_streamlit_agents.py`; driven against a real API |

## Depends on external services or credentials

| Feature | Needs | Without it |
|---|---|---|
| Agents, plan generation, health check | Ollama with `REPWISE_AGENT_MODEL` and `REPWISE_HEALTH_CHECK_MODEL` | The agent says the model is unreachable and changes nothing. The health check fails closed, so nothing is adapted automatically. |
| Plan generation, `search_documents` | The prepared corpus (`REPWISE_CHUNK_FILE`) and, for hybrid retrieval, the semantic index | `retrieval_error` / `search_unavailable` |
| Real calendar slots and bookings | A Google OAuth client and `python -m scripts.connect_google_calendar connect` per account | `calendar_not_connected`. Moving workouts within the plan still works. |
| Nearby places | Internet access to Photon (`REPWISE_PHOTON_URL`). The public server rate-limits bursts of requests. | `places_unavailable` |
| MCP server | A login token (`REPWISE_MCP_ACCESS_TOKEN`) and the API's secret | The server refuses to start. |

## Deferred to the web frontend phase

The web frontend (`app/`) was deliberately left unchanged. Everything below already
works in the Streamlit app, and the backend contracts are ready for `app/`:

- a chat view calling `POST /agent/requests`;
- approve and decline buttons for `pending_actions`;
- a list of pending actions after a reload;
- a location-sharing permission prompt, and the opt-in saved location (`/location`);
- showing draft plans (`status: "draft"` in the plan list) apart from the active one.

See [the approval contract](agents.md#contract-for-the-frontend).

## Known limitations

- **Model quality.** Routing, task wording and answer prose come from a local 9B
  model. Code guards facts, dates, approvals and safety, but wording can still be
  imperfect. Rerun `python -m scripts.run_agent` after changing prompts or models.
- **Conversation history lasts only while the process runs.** It is per
  `conversation_id`. Long-term context comes from semantic and episodic memory.
- **Agent requests are synchronous.** A request that generates a plan can take one
  to two minutes on a local model.
- **The health check flags; it doesn't diagnose.** An unclear comment can be
  flagged, and then the student approves the action or adapts manually.
- **Map data** (OpenStreetMap) can be incomplete or out of date, and never shows
  stock or opening hours. Unrelated shops are filtered by name, so a shop with a
  neutral name ("Sport System") can still be a running or team-sport shop.
- **Only metropolitan France is served.** Overseas departments and other
  countries are refused; widen `service_area` in `config/places.json` to change
  that.
- **Security** suits a local course project; see
  [security limits](api.md#security-limits). The MCP server is for a trusted,
  local stdio host only.
