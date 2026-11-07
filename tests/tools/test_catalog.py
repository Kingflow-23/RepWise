"""The single tool catalog (config/tools.json) must match the code and the agent sheet."""

import json

import pytest

from config.config import TOOL_CATALOG_PATH
from src.agents.specialists import SPECIALISTS, tool_schemas
from src.tools.catalog import ToolCatalog, load_catalog, tools_for_agent
from src.tools.registry import IMPLEMENTATIONS, catalog_problems

# The tools the completed agent sheet lists for each agent.
SHEET_TOOLS = {
    "orchestrator": {
        "get_user_profile",
        "get_current_plan",
        "get_timetable",
        "get_workout",
    },
    "planning": {"get_workout_history", "generate_workout_plan", "save_recommendation"},
    "performance": {
        "save_workout_feedback",
        "update_upcoming_workout",
        "save_recommendation",
    },
    "schedule": {
        "get_calendar_busy",
        "propose_training_slots",
        "organize_schedule",
        "add_workouts_to_calendar",
    },
    "location": {"get_location"},
    "knowledge": {"search_documents"},
}


def test_catalog_matches_the_implementations():
    assert catalog_problems() == []


def test_every_tool_of_the_agent_sheet_is_available_to_its_agent():
    for agent, names in SHEET_TOOLS.items():
        available = {tool.name for tool in tools_for_agent(agent)}
        assert names <= available, f"{agent} is missing {names - available}"


def test_every_tool_has_a_unique_name_and_an_implementation():
    names = [tool.name for tool in load_catalog().tools]
    assert len(names) == len(set(names))
    assert set(names) == set(IMPLEMENTATIONS)


def test_duplicate_names_are_rejected():
    data = json.loads(TOOL_CATALOG_PATH.read_text(encoding="utf-8"))
    data["tools"].append(data["tools"][0])

    with pytest.raises(ValueError, match="Duplicate tool names"):
        ToolCatalog.model_validate(data)


def test_a_read_tool_cannot_require_approval():
    data = json.loads(TOOL_CATALOG_PATH.read_text(encoding="utf-8"))
    tool = next(item for item in data["tools"] if item["name"] == "get_timetable")
    tool.update(approval="always", approval_reason="x")

    with pytest.raises(ValueError, match="read tool cannot require approval"):
        ToolCatalog.model_validate(data)


def test_consequential_writes_always_need_approval():
    approval = {tool.name: tool.approval for tool in load_catalog().tools}
    for name in (
        "activate_plan",
        "update_upcoming_workout",
        "organize_schedule",
        "add_workouts_to_calendar",
    ):
        assert approval[name] == "always"
    assert approval["save_workout_feedback"] == "conditional"


def test_no_tool_lets_the_model_choose_the_account():
    for tool in load_catalog().tools:
        schema = json.dumps(tool.input_schema)
        for forbidden in (
            "student_id",
            "user_id",
            "access_token",
            "api_url",
            "latitude",
        ):
            assert forbidden not in schema, f"{tool.name} accepts {forbidden}"


def test_each_specialist_is_offered_only_its_catalog_tools():
    for name in SPECIALISTS:
        offered = {schema["function"]["name"] for schema in tool_schemas(name)}
        assert offered == {tool.name for tool in tools_for_agent(name)}


def test_the_orchestrator_has_no_training_or_calendar_writes():
    writes = {
        tool.name for tool in tools_for_agent("orchestrator") if tool.access == "write"
    }
    assert writes == {"remember_fact"}
