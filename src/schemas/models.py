"""Clear data shapes used by RepWise V1."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    """Reject unknown fields so input errors are visible early."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class UserProfile(Model):
    """Small set of student-provided details useful for a workout plan."""

    experience: str = Field(
        description="For example: beginner, intermediate or advanced"
    )
    goal: str = Field(min_length=3, max_length=200)
    available_days: list[str] = Field(min_length=1, max_length=7)
    session_minutes: int = Field(ge=15, le=180)
    equipment: list[str] = Field(default_factory=list)
    preferences: str | None = Field(default=None, max_length=500)


class PlanOptions(Model):
    """The current request sent alongside a saved student profile."""

    focus: str = Field(min_length=3, max_length=300)
    weeks: int = Field(default=4, ge=1, le=12)
    extra_constraints: list[str] = Field(default_factory=list, max_length=10)


class PlanRequest(PlanOptions):
    """Complete input used inside the generation pipeline."""

    profile: UserProfile


class RetrievedChunk(Model):
    """A chunk returned by the retrieval layer owned by the ingestion team."""

    text: str = Field(min_length=1)
    video_id: str
    passage_ids: list[str] = Field(min_length=1)
    title: str
    video_url: str
    start_url: str | None = None
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    speaker: str | None = None
    topics: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    score: float | None = None
    rank: int | None = Field(default=None, ge=1)


class Source(Model):
    """A short link back to evidence used by the model."""

    source_id: str
    title: str
    url: str
    excerpt: str


class Exercise(Model):
    name: str
    sets: int = Field(ge=1, le=10)
    repetitions: str
    rest_seconds: int = Field(ge=0, le=600)
    coaching_note: str
    source_ids: list[str] = Field(default_factory=list)


class Workout(Model):
    day: str
    focus: str
    estimated_minutes: int = Field(ge=1, le=180)
    exercises: list[Exercise] = Field(min_length=1, max_length=10)


class Week(Model):
    number: int = Field(ge=1)
    objective: str
    workouts: list[Workout] = Field(min_length=1, max_length=7)


class TrainingPlan(Model):
    title: str
    summary: str
    weeks: list[Week] = Field(min_length=1, max_length=12)
    assumptions: list[str] = Field(default_factory=list)
    safety_note: str


class PlanResult(Model):
    """Result returned by the pipeline and sent back by the API."""

    status: str
    message: str
    plan: TrainingPlan | None = None
    sources: list[Source] = Field(default_factory=list)

    @model_validator(mode="after")
    def completed_result_has_plan(self) -> "PlanResult":
        if self.status == "completed" and self.plan is None:
            raise ValueError("a completed result needs a plan")
        return self
