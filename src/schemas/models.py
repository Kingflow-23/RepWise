"""Clear data shapes used by RepWise V1."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class Model(BaseModel):
    """Reject unknown fields so input errors are visible early."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class UserProfile(Model):
    """Student-provided constraints used to personalize a workout plan.

    ``available_days`` and ``session_minutes`` constrain the schedule;
    ``equipment`` constrains exercise selection. ``experience``, ``goal``, and
    optional ``preferences`` provide context for query construction and
    generation, but are not independently verified facts.
    """

    experience: str = Field(
        description="For example: beginner, intermediate or advanced"
    )
    goal: str = Field(min_length=3, max_length=200)
    available_days: list[str] = Field(min_length=1, max_length=7)
    session_minutes: int = Field(ge=15, le=180)
    equipment: list[str] = Field(default_factory=list)
    preferences: str | None = Field(default=None, max_length=500)


class AccountCredentials(Model):
    """Credentials accepted by local account registration and login routes.

    ``full_name`` is the user-facing account identifier and must be unique in
    the repository. ``password`` is accepted only at the API boundary and is
    stored as a password hash, never as plaintext.
    """

    full_name: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8, max_length=200)


class AccountIdentity(Model):
    """Public account identity, without password hashes or bearer credentials."""

    user_id: str
    full_name: str


class AccessToken(Model):
    """Bearer-token response returned after successful account authentication.

    ``user_id`` is the internal account ID used to scope later API operations;
    ``token_type`` identifies how clients present ``access_token``.
    """

    access_token: str
    user_id: str
    full_name: str
    token_type: str = "bearer"


class PlanOptions(Model):
    """Request-specific focus and schedule, separate from a saved profile.

    ``workouts_per_week`` may be omitted to use the profile-based default.
    ``extra_constraints`` carries additional natural-language requirements
    without expanding the fixed profile schema for each new request.
    ``refinement_instructions`` tells the generator how to adapt the latest
    saved plan, when one exists.
    """

    focus: str = Field(min_length=3, max_length=300)
    workouts_per_week: int | None = Field(default=None, ge=1, le=7)
    extra_constraints: list[str] = Field(default_factory=list, max_length=10)
    refinement_instructions: str | None = Field(default=None, max_length=500)


class PlanRequest(PlanOptions):
    """Complete input to generation, combining request options and a saved profile.

    If no weekly workout count is supplied, the pipeline targets up to three
    workouts, limited by the profile's available days.
    """

    weeks: int = Field(default=1, ge=1, le=12)
    profile: UserProfile

    @model_validator(mode="after")
    def weekly_workouts_fit_available_days(self) -> "PlanRequest":
        """Reject an explicit schedule that exceeds the profile's available days.

        Args:
            self: Validated request whose requested workout count and profile days
                are being compared.

        Returns:
            This request when its weekly workout count fits the available days.

        Raises:
            ValueError: If the explicit count exceeds the number of available days.
        """
        if self.workouts_per_week is not None and self.workouts_per_week > len(
            self.profile.available_days
        ):
            raise ValueError("workouts_per_week cannot exceed available_days")
        return self

    @property
    def target_workouts_per_week(self) -> int:
        """Resolve the explicit weekly count or its profile-based default.

        Args:
            self: Request whose optional count and profile availability determine
                the target.

        Returns:
            The requested count, or the smaller of three and the available-day
            count when no explicit count was supplied.
        """
        return self.workouts_per_week or min(3, len(self.profile.available_days))


class RetrievedChunk(Model):
    """Indexable transcript chunk and provenance returned by a retriever.

    ``text`` is the complete chunk used as evidence, not a display excerpt.
    ``chunk_id`` is the unique identifier in the corpus.
    ``passage_ids`` connects the chunk to multiple underlying passage records;
    video IDs, timestamps, title, and URL support source display. Topics and
    body-part tags are retrieval metadata, while score and rank describe a
    particular search result rather than permanent corpus content.
    """

    text: str = Field(min_length=1)
    chunk_id: str
    video_id: str
    passage_ids: list[str] = Field(min_length=1)
    title: str
    video_url: str
    start_url: str | None = None
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    speaker: str | None = None
    topics: list[str] = Field(default_factory=list)
    topic_counts: dict[str, int] = Field(default_factory=dict)
    body_parts: list[str] = Field(default_factory=list)
    scientific_references: list[str] = Field(default_factory=list)
    reference_review_status: str | None = None
    word_count: int | None = Field(default=None, ge=0)
    score: float | None = None
    rank: int | None = Field(default=None, ge=1)


class Source(Model):
    """A prompt-scoped citation pointing back to retrieved corpus evidence.

    ``source_id`` is the citation key supplied to the model. ``chunk_id`` is
    the underlying corpus identity and may differ. ``excerpt`` is suitable for
    concise display; ``evidence_text`` retains the full text for user review.
    ``matched_terms`` are review highlights, not an explanation of model
    reasoning.
    """

    source_id: str
    title: str
    url: str
    excerpt: str
    chunk_id: str | None = None
    passage_ids: list[str] = Field(default_factory=list)
    video_id: str | None = None
    start: int | None = None
    end: int | None = None
    video_url: str | None = None
    start_url: str | None = None
    evidence_text: str | None = None
    matched_terms: list[str] = Field(default_factory=list)


class EvidenceQuote(Model):
    """A verbatim passage excerpt tied to one retrieved source."""

    source_id: str
    quote: str = Field(min_length=12, max_length=500)


class Exercise(Model):
    """One prescribed movement with dosage, coaching, equipment, and citations.

    ``source_ids`` identify supplied evidence; ``supporting_evidence`` can hold
    optional quotes from those sources. Source IDs are checked against retrieved
    evidence, but quote text is not checked or required.
    """

    name: str = Field(min_length=3, max_length=100)
    sets: int = Field(ge=1, le=10)
    repetitions: str
    rest_seconds: int = Field(ge=0, le=300)
    estimated_minutes: int = Field(ge=1, le=120)
    coaching_note: str
    source_ids: list[str] = Field(default_factory=list)
    supporting_evidence: list[EvidenceQuote] = Field(default_factory=list)
    equipment: list[str] = Field(default_factory=list)


class Workout(Model):
    """A single scheduled workout within a plan week.

    ``estimated_minutes`` is checked against the student's requested session
    duration. ``focus`` describes this workout and is distinct from the
    request-wide focus.
    """

    day: str
    focus: str
    estimated_minutes: int = Field(ge=1, le=120)
    exercises: list[Exercise] = Field(min_length=1, max_length=10)


CompletionStatus = Literal["completed", "partially_completed", "not_completed"]
WorkoutDifficulty = Literal[
    "much_too_easy",
    "slightly_too_easy",
    "just_right",
    "slightly_too_hard",
    "much_too_hard",
]
# Plan names use English weekdays. Kept here, not strftime("%A"), which follows
# the computer's language.
WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
# "draft": generated but not approved yet; "active": the plan the student
# follows; "archived": replaced by a newer active plan.
PlanStatus = Literal["draft", "active", "archived"]


def weekday_number(day: str) -> int | None:
    """Return 0 for Monday through 6 for Sunday.

    Args:
        day: An English weekday name, in any letter case.

    Returns:
        The weekday number, or ``None`` when the name is not a weekday.
    """
    names = [name.casefold() for name in WEEKDAYS]
    cleaned = day.strip().casefold()
    return names.index(cleaned) if cleaned in names else None


class WorkoutFeedbackRequest(Model):
    """A student's concise report on one scheduled workout."""

    completion_status: CompletionStatus
    difficulty: WorkoutDifficulty
    comment: str | None = Field(default=None, max_length=500)


class WorkoutFeedback(Model):
    """Persisted workout feedback and the workout it adapted, if any.

    ``health_concern`` is decided once, when the feedback is saved, by the
    health check in ``src/agents/health.py``: ``True`` when the comment may
    describe pain or injury, ``False`` when it does not, and ``None`` when
    there was no comment to check.
    """

    id: str
    plan_id: str
    week_number: int = Field(ge=1)
    workout_index: int = Field(ge=0)
    completion_status: CompletionStatus
    difficulty: WorkoutDifficulty
    comment: str | None = None
    created_at: str
    adapted_week_number: int | None = Field(default=None, ge=1)
    adapted_workout_index: int | None = Field(default=None, ge=0)
    health_concern: bool | None = None


class Week(Model):
    """A numbered plan week with an objective and scheduled workouts."""

    number: int = Field(ge=1)
    objective: str
    workouts: list[Workout] = Field(min_length=1, max_length=7)


class TrainingPlan(Model):
    """Structured plan payload returned by successful generation.

    The application currently requests one week per interaction, while the
    shared schema can represent multiple weeks for validation and evaluation.
    ``assumptions`` expose model assumptions and ``safety_note`` is always part
    of the user-facing plan; neither makes the plan medically authoritative.
    """

    title: str
    summary: str
    weeks: list[Week] = Field(min_length=1, max_length=12)
    assumptions: list[str] = Field(default_factory=list)
    safety_note: str


class PlanResult(Model):
    """Pipeline outcome shared with the API, including optional source records.

    ``status`` distinguishes completion from retrieval, evidence, and
    generation failures. A completed result must include a plan; the current
    pipeline omits it on failures. Sources can still be returned when generation
    was attempted but validation failed.
    """

    status: str
    message: str
    plan: TrainingPlan | None = None
    sources: list[Source] = Field(default_factory=list)
    _evaluation_diagnostics: dict[str, object] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def completed_result_has_plan(self) -> "PlanResult":
        """Require a plan payload when the result reports successful completion.

        Args:
            self: Pipeline result whose status and optional plan are checked.

        Returns:
            This result when its status and plan payload are consistent.

        Raises:
            ValueError: If the status is ``completed`` but no plan is present.
        """
        if self.status == "completed" and self.plan is None:
            raise ValueError("a completed result needs a plan")
        return self
