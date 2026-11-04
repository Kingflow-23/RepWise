"""Run RepWise's MCP tools over stdio.

The tool implementations live in :mod:`agentic.tools`; this entry
point makes the server easy to launch from the Session 3 notebook and from MCP
hosts that expect a root-level ``server.py`` file.

Run from the repository root with ``python server.py``.
"""

from agentic.mcp_server import mcp


if __name__ == "__main__":
    mcp.run(show_banner=False)
