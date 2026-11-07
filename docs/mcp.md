# The MCP server

RepWise runs **two kinds of server**. They share the same tools, database and
catalog, and have different jobs:

| | FastAPI backend | MCP server |
|---|---|---|
| Start | `python -m uvicorn src.api.main:app` | `python -m src.mcp_server.server` |
| Used by | The web app (`app/`), Streamlit (`genai_app/`), any HTTP client | An MCP host (Claude Desktop, an IDE, the course notebook) |
| Transport | HTTP on a port | stdio: the host starts it as a subprocess |
| Accounts | Every account; a bearer token per request | **One** account, bound at startup by its token |
| Agents | `POST .../agent/requests` runs the orchestrator | The host's own model calls the tools directly |
| Approvals | `POST .../agent/actions/{id}/approve` | Writes needing approval return `pending_approval`; the student approves them through the API |

Use FastAPI for the RepWise app. Use the MCP server to connect RepWise's tools to
an MCP host.

## How it starts

`python -m src.mcp_server.server` runs `main()` in
[`src/mcp_server/server.py`](../src/mcp_server/server.py), which does this:

1. Configure logging to stderr and `logs/`. stdout carries the MCP protocol.
2. Load settings from the environment and `.env`.
3. **Authenticate.** Verify `REPWISE_MCP_ACCESS_TOKEN` with `REPWISE_AUTH_SECRET`,
   and check that the account exists in `REPWISE_DATABASE_PATH`. If anything is
   wrong, print the reason to stderr and exit with code 1. A host never gets a
   server with no account bound.
4. Check that `config/tools.json` matches the implementations, or refuse to start.
5. Register one MCP tool per catalog entry, with the catalog's description and the
   Pydantic arguments schema.
6. Serve over stdio.

Each call goes through the same approval gate as the agents
([`run_tool_with_approval`](../src/agents/approvals.py)), with `proposed_by="mcp"`.
The corpus and models load only when a tool first needs them.

## Why tools run in-process

The server verifies the token once, then calls the tool functions directly on the
database. That is the fastest option: a call costs one SQLite query, with no extra
HTTP hop and no duplicate API routes. The trade-off is that the server must run on
the machine that holds the database, which suits a local stdio host. It isn't a
public network deployment: there is no HTTP listener and no multi-user login.

**No tool accepts an account ID.** A model or client can't act for another
account, because the account comes only from the verified token. An expired token
stops the server from starting; sign in again to get a new one.

## Connecting a host

Sign in through the API, `POST /api/v1/auth/login`, and keep the `access_token`.
Then configure the host, for example:

```json
{
  "mcpServers": {
    "repwise": {
      "command": "C:/path/to/RepWise/.venv/Scripts/python.exe",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "C:/path/to/RepWise",
      "env": {
        "REPWISE_DATABASE_PATH": "data/repwise.sqlite3",
        "REPWISE_AUTH_SECRET": "<the API's secret>",
        "REPWISE_MCP_ACCESS_TOKEN": "<access_token from login>",
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

This JSON is the host's *launch configuration*. It is separate from the tool
catalog (`config/tools.json`). Keep it private: it holds a token and a secret.

## Checking it

```powershell
python -m scripts.mcp_smoke
```

This starts `server.py` over stdio on a temporary database and checks four things:

- every catalog tool is listed, with no `student_id`;
- a read works;
- a write is held for approval;
- an invalid call returns a structured error.

It also confirms that a wrong token stops the server from starting.
`tests/tools/test_mcp_server.py` runs the same checks, plus authentication edge
cases, in the test suite.

The course notebook [`S3_RepWise_MCP_server.ipynb`](../notebooks/S3_RepWise_MCP_server.ipynb)
walks through discovery and calls the same way, on a temporary account.
