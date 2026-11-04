# RepWise MCP lab

The root notebook `Agentic_04Nov_S3_Lab_Build_and_connect_an_MCP_server.ipynb`
adapts the campus lab to a real RepWise workout request. It runs locally with
Python 3.11 or later and Ollama. No hosted-model key or Colab setup is needed.

```powershell
python -m pip install -r requirements-notebook.txt
ollama pull qwen3.5:4b
```

Open the notebook with this Python environment as its kernel, then run all
cells. The prepared chunk files must be available through `REPWISE_CHUNK_FILE`.
Alternatively, execute and save all notebook outputs from the project root:

```powershell
python scripts/execute_mcp_lab.py
```

The runner uses the current Python interpreter as a temporary notebook kernel,
so it does not require installing a named kernel in your Jupyter settings.
The default lab uses the installed `qwen3.5:4b` for both the agent and backend
generation, BM25 retrieval and the regular scope gate. Query rewriting is
disabled for this bounded exercise. Set `REPWISE_AGENT_MODEL` to another
installed tool-capable Ollama model to experiment.

The notebook starts a temporary API on a free localhost port, registers a
temporary account and saves its profile. It passes the access token privately
to a stdio MCP server. The model sees the tool schema and training request,
never the token or account configuration. The MCP server forwards the model's
validated arguments to the existing authenticated API. The API performs the
real retrieval, generation, validation and persistence.

The agent discovers the schema through MCP, chooses tool arguments, calls the
tool once, reads its structured result and summarizes it. Assertions check the
returned Pydantic JSON, source references and the saved plan through the API.
No mock model or prewritten plan is substituted. At most five agent rounds
and 240 seconds are allowed; repeated generation calls and false completions
are rejected. A failed run raises visibly. The temporary API, account and
database are cleaned up when the request cell finishes, including on failure.
Invalid model arguments are rejected locally and returned to the model for
correction within the same step budget, before executing a write tool.
The normal `.env`, running API and user database are not modified.

## Connecting to an existing API

`python -m src.mcp_server.server` starts the standalone stdio server. Its host
must provide these environment variables privately after authenticating:

- `REPWISE_API_URL`: the trusted RepWise backend URL.
- `REPWISE_MCP_USER_ID`: the user ID returned by login.
- `REPWISE_MCP_ACCESS_TOKEN`: the bearer token returned by login.

Only `generate_workout_plan` is exposed. The tool cannot choose an account ID
or API URL. The backend checks the token and account ownership for every call.
This is a trusted local stdio connection, not a public MCP HTTP deployment.

The protocol uses the official MCP Python SDK v1, pinned below v2 because its
client and FastMCP API differ. Reference: https://py.sdk.modelcontextprotocol.io/v1/
