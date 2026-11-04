"""A stdio MCP server forwarding plan requests to the authenticated API."""

from __future__ import annotations

import os
from typing import Annotated

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ..schemas.models import PlanOptions
from ..tools.generate_workout_plan import GenerateWorkoutPlanResult


def create_server(
    api_url: str, user_id: str, access_token: str,
    *, transport: httpx.AsyncBaseTransport | None = None,
) -> FastMCP:
    """Bind one authenticated account outside model-controlled arguments.

    This server is launched by a trusted local host over stdio. It does not
    expose a public HTTP listener or implement a separate login mechanism.
    The backend validates the token and account ownership on every request.
    """
    if not api_url or not user_id or not access_token:
        raise ValueError("Set the API URL, MCP user ID and MCP access token before starting")
    mcp = FastMCP("RepWise")

    @mcp.tool()
    async def generate_workout_plan(
        focus: Annotated[str, Field(min_length=3, max_length=300)],
        workouts_per_week: Annotated[int | None, Field(ge=1, le=7)] = None,
        extra_constraints: Annotated[list[str] | None, Field(max_length=10)] = None,
        refinement_instructions: Annotated[str | None, Field(max_length=500)] = None,
    ) -> GenerateWorkoutPlanResult:
        """Generate and save a one-week workout plan for the signed-in user.

        Use the saved profile's experience, equipment and available days.
        Provide a training focus and optional weekly workout count. Returns
        the validated plan, evidence sources and saved ID, or an explicit
        failure status. This tool saves a new plan: call once per requested plan.
        """
        options = PlanOptions(
            focus=focus, workouts_per_week=workouts_per_week,
            extra_constraints=extra_constraints or [],
            refinement_instructions=refinement_instructions,
        )
        async with httpx.AsyncClient(
            base_url=api_url.rstrip("/"), timeout=180, transport=transport,
            headers={"Authorization": f"Bearer {access_token}"},
        ) as client:
            response = await client.post(
                f"/api/v1/students/{user_id}/plans", json=options.model_dump()
            )
        if response.is_error:
            # Keep authentication/transport errors as MCP errors, not fake plans.
            detail = response.json().get("detail")
            if isinstance(detail, dict) and "status" in detail:
                return GenerateWorkoutPlanResult.model_validate(detail)
            if response.status_code == 404:
                return GenerateWorkoutPlanResult(status="missing_profile", message=str(detail))
            raise RuntimeError(f"RepWise API rejected the request (HTTP {response.status_code})")
        return GenerateWorkoutPlanResult.model_validate(response.json())

    return mcp


def main() -> None:
    server = create_server(
        os.environ.get("REPWISE_API_URL", "http://127.0.0.1:8000"),
        os.environ.get("REPWISE_MCP_USER_ID", ""),
        os.environ.get("REPWISE_MCP_ACCESS_TOKEN", ""),
    )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
