"""Load and check ``config/tools.json``, the single catalog of agent tools.

The catalog is read at runtime: tool descriptions shown to the model, which
agents may call a tool, whether it writes, and when it needs approval all come
from this file. Edit the file to change them; there is no second list in code.

``input_schema`` and ``output_models`` in the file are generated from the
Pydantic models in ``src/tools/registry.py``; ``python -m
scripts.sync_tool_catalog`` refreshes them and ``--check`` reports drift.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator

from config.config import TOOL_CATALOG_PATH
from src.schemas.models import Model

AgentName = Literal[
    "orchestrator", "planning", "performance", "schedule", "location", "knowledge"
]


class ToolSpec(Model):
    """Everything the catalog says about one tool.

    Attributes:
        name: Unique tool name used by agents and MCP clients.
        description: What the tool does, shown to the language model.
        agents: Agents allowed to call it; the first one owns it.
        access: ``read`` tools change nothing; ``write`` tools save data.
        approval: ``never``, ``always``, or ``conditional`` (decided per call).
        approval_reason: Text shown to the student when approval is needed.
        external_services: Services the tool contacts, such as ``ollama``.
        failure_codes: ``ToolFailure.code`` values the tool can return.
        input_schema: JSON schema of the arguments (generated).
        output_models: Names of the success result models (generated).
    """

    name: str = Field(pattern=r"^[a-z][a-z_]*$")
    description: str = Field(min_length=10)
    agents: list[AgentName] = Field(min_length=1)
    access: Literal["read", "write"]
    approval: Literal["never", "always", "conditional"]
    approval_reason: str | None = None
    external_services: list[str] = Field(default_factory=list)
    failure_codes: list[str] = Field(default_factory=list)
    input_schema: dict
    output_models: list[str] = Field(min_length=1)

    @property
    def owner(self) -> AgentName:
        """Return the agent responsible for this tool."""
        return self.agents[0]

    @model_validator(mode="after")
    def approval_is_explained(self) -> "ToolSpec":
        """Require a reason for approval-gated tools, and no approval on reads.

        Returns:
            The validated spec.

        Raises:
            ValueError: If the approval settings are inconsistent.
        """
        if self.approval != "never" and not self.approval_reason:
            raise ValueError(
                f"{self.name}: an approval-gated tool needs approval_reason"
            )
        if self.access == "read" and self.approval != "never":
            raise ValueError(f"{self.name}: a read tool cannot require approval")
        return self


class ToolCatalog(Model):
    """The whole catalog file."""

    description: str
    version: int = Field(ge=1)
    tools: list[ToolSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def names_are_unique(self) -> "ToolCatalog":
        """Reject two tools with the same name.

        Returns:
            The validated catalog.

        Raises:
            ValueError: If a name is duplicated.
        """
        names = [tool.name for tool in self.tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate tool names: {', '.join(duplicates)}")
        return self


@lru_cache(maxsize=1)
def load_catalog() -> ToolCatalog:
    """Read and validate ``config/tools.json`` once per process.

    Returns:
        The validated catalog.

    Raises:
        OSError: If the file cannot be read.
        pydantic.ValidationError: If the file does not match :class:`ToolCatalog`.
    """
    return ToolCatalog.model_validate_json(
        TOOL_CATALOG_PATH.read_text(encoding="utf-8")
    )


def get_tool_spec(name: str) -> ToolSpec:
    """Return one tool's catalog entry.

    Args:
        name: Tool name.

    Returns:
        Its spec.

    Raises:
        KeyError: If the catalog has no such tool.
    """
    for tool in load_catalog().tools:
        if tool.name == name:
            return tool
    raise KeyError(f"Tool {name!r} is not in config/tools.json.")


def tools_for_agent(agent: AgentName) -> list[ToolSpec]:
    """List the tools an agent may call, in catalog order.

    Args:
        agent: Agent name.

    Returns:
        The tools whose ``agents`` include ``agent``.
    """
    return [tool for tool in load_catalog().tools if agent in tool.agents]


def generated_parts(input_schema: dict, output_models: list[str]) -> dict:
    """Return the catalog fields that are generated from code.

    Args:
        input_schema: JSON schema of the arguments model.
        output_models: Names of the success result models.

    Returns:
        ``{"input_schema": ..., "output_models": ...}``, normalized through
        JSON so it compares equal to what is read from the file.
    """
    return json.loads(
        json.dumps({"input_schema": input_schema, "output_models": output_models})
    )
