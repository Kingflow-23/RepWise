"""Refresh the generated parts of ``config/tools.json`` from the Python code.

Descriptions, owners, access and approval rules are written by hand in the JSON
file and are never touched here. Only ``input_schema`` and ``output_models``
are regenerated from ``src/tools/registry.py``.

From the repository root:

    python -m scripts.sync_tool_catalog          # rewrite the generated parts
    python -m scripts.sync_tool_catalog --check  # only report drift (exit 1)
"""

from __future__ import annotations

import argparse
import json
import sys

from config.config import TOOL_CATALOG_PATH
from src.tools.catalog import load_catalog
from src.tools.registry import (
    IMPLEMENTATIONS,
    catalog_problems,
    generated_catalog_parts,
)


def sync() -> list[str]:
    """Rewrite ``input_schema`` and ``output_models`` for every implemented tool.

    Returns:
        Problems that a rewrite cannot fix (a tool missing on one side).
    """
    data = json.loads(TOOL_CATALOG_PATH.read_text(encoding="utf-8"))
    for tool in data["tools"]:
        if tool["name"] in IMPLEMENTATIONS:
            tool.update(generated_catalog_parts(tool["name"]))
    TOOL_CATALOG_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    load_catalog.cache_clear()
    return catalog_problems()


def main() -> int:
    """Run the command line.

    Returns:
        0 when the catalog matches the code, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Only report drift.")
    arguments = parser.parse_args()
    problems = catalog_problems() if arguments.check else sync()
    for problem in problems:
        print(f"- {problem}")
    if problems:
        return 1
    print("config/tools.json matches the tool implementations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
