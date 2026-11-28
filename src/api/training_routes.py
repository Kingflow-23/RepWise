"""HTTP routes for workout performance (sets, reps, weights) and history.

These routes call the same tool functions as the Performance and Planning
agents (``log_workout_performance``, ``get_workout``, ``get_workout_history``),
so the web app and the agents apply exactly the same rules: exercise names
are checked against the workout, a wrong name saves nothing, and saving an
exercise again replaces its record. Logging performance needs no approval,
as in ``config/tools.json``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import Path as APIPath
from pydantic import Field

from src.schemas.models import Model
from src.tools.context import ToolContext
from src.tools.profile import PerformanceRecord, get_workout
from src.tools.results import ToolFailure
from src.tools.training import (
    HistoryResult,
    PerformanceResult,
    get_workout_history,
    log_workout_performance,
)

logger = logging.getLogger(__name__)

#: HTTP status for each tool failure code these routes can return.
TOOL_FAILURE_STATUS = {
    "student_not_found": 404,
    "plan_not_found": 404,
    "workout_not_found": 404,
    "exercise_not_found": 422,
    "invalid_request": 422,
    "storage_error": 503,
}

WORKOUT_PATH = "/plans/{plan_id}/weeks/{week_number}/workouts/{workout_index}/performance"


class PerformanceRequest(Model):
    """What the student did in one workout, per exercise."""

    exercises: list[PerformanceRecord] = Field(min_length=1, max_length=30)


def raise_for_failure(result: object) -> None:
    """Turn a tool failure into an HTTP error with its code and message.

    Args:
        result: A tool's return value.

    Raises:
        HTTPException: When ``result`` is a :class:`ToolFailure`.
    """
    if isinstance(result, ToolFailure):
        status = TOOL_FAILURE_STATUS.get(result.code, 500)
        message = (
            "The database could not be used. Please try again."
            if result.code == "storage_error"
            else result.error
        )
        raise HTTPException(status, {"code": result.code, "message": message})


def build_training_router(
    *,
    require_account: Callable[..., str],
    check_owner: Callable[[str, str], None],
    make_context: Callable[..., ToolContext],
) -> APIRouter:
    """Create the performance and history routes.

    Args:
        require_account: FastAPI dependency returning the authenticated ID.
        check_owner: Raises 403 when the path account is not the caller.
        make_context: Builds a trusted ToolContext for an account.

    Returns:
        A router to include in the FastAPI app.
    """
    router = APIRouter(prefix="/api/v1/students/{student_id}")

    @router.get(WORKOUT_PATH, response_model=list[PerformanceRecord])
    def read_performance(
        student_id: str,
        plan_id: str,
        week_number: int = APIPath(ge=1),
        workout_index: int = APIPath(ge=0),
        user_id: str = Depends(require_account),
    ) -> list[PerformanceRecord]:
        """Return what was logged for one workout (empty when nothing was).

        Raises:
            HTTPException: 403 for another account, 404 for an unknown plan
                or workout.
        """
        check_owner(student_id, user_id)
        result = get_workout(
            make_context(student_id), week_number, workout_index, plan_id=plan_id
        )
        raise_for_failure(result)
        return result.performance

    @router.post(WORKOUT_PATH, response_model=PerformanceResult)
    def save_performance(
        student_id: str,
        plan_id: str,
        body: PerformanceRequest,
        week_number: int = APIPath(ge=1),
        workout_index: int = APIPath(ge=0),
        user_id: str = Depends(require_account),
    ) -> PerformanceResult:
        """Save sets, reps and weights for one workout of an owned plan.

        Raises:
            HTTPException: 403 for another account, 404 for an unknown plan or
                workout, 422 for an exercise that is not in the workout (then
                nothing is saved), 503 when the database fails.
        """
        check_owner(student_id, user_id)
        result = log_workout_performance(
            make_context(student_id),
            week_number,
            workout_index,
            body.exercises,
            plan_id=plan_id,
        )
        raise_for_failure(result)
        logger.info(
            "Workout performance saved week=%s workout=%s exercises=%s",
            week_number,
            workout_index,
            len(result.records),
        )
        return result

    @router.get("/history", response_model=HistoryResult)
    def read_history(
        student_id: str,
        max_plans: int = Query(5, ge=1, le=20),
        user_id: str = Depends(require_account),
    ) -> HistoryResult:
        """Return logged workouts and performance of the recent plans.

        Raises:
            HTTPException: 403 for another account, 404 without a profile.
        """
        check_owner(student_id, user_id)
        result = get_workout_history(make_context(student_id), max_plans=max_plans)
        raise_for_failure(result)
        return result

    return router
