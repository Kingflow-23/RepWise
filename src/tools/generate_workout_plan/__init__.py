"""Generate and save a validated one-week training plan."""

from .schemas import GenerateWorkoutPlanResult
from .tool import generate_workout_plan

__all__ = ["GenerateWorkoutPlanResult", "generate_workout_plan"]
