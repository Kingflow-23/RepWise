"""Reuse the application's plan contracts for the generation tool."""

from pydantic import model_validator

from ...schemas.models import PlanResult


class GenerateWorkoutPlanResult(PlanResult):
    """A completed tool call includes the ID of the saved, validated plan."""

    plan_id: str | None = None

    @model_validator(mode="after")
    def saved_plan_matches_status(self) -> "GenerateWorkoutPlanResult":
        if self.status == "completed" and not self.plan_id:
            raise ValueError("A completed generation must include its saved plan ID")
        if self.status != "completed" and self.plan_id is not None:
            raise ValueError("An unsuccessful generation cannot have a saved plan ID")
        return self
