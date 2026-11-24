"""The only module that knows how to call Ollama."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Protocol

from src.agents.metrics import record_external_call
from src.langfuse_tracing import finish_generation, generation_observation

from ..schemas.models import TrainingPlan, Workout
from .errors import classify_provider_error

logger = logging.getLogger(__name__)

#: Time budget for one structured Ollama call, in seconds.
GENERATION_TIMEOUT_SECONDS = 300


class PlanGenerator(Protocol):
    """Interface for providers that return a validated structured plan."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> TrainingPlan:
        """Generate a plan from prompts, optionally using a provider schema.

        Args:
            system_prompt: Instructions defining the model's role and constraints.
            user_prompt: Request and retrieved evidence supplied for this generation.
            response_schema: Optional JSON schema constraining the provider response.

        Returns:
            The parsed plan.

        Raises:
            Exception: If the provider call or response parsing fails.
        """


class OllamaGenerator:
    """Ask a local Ollama model for structured training-plan output."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        """Configure the local Ollama model endpoint.

        Args:
            model: Installed Ollama model name to use for generation.
            base_url: Base URL of the Ollama server.
        """
        self.model = model
        self.base_url = base_url
        self._last_raw_output: str | None = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> TrainingPlan:
        """Call Ollama and validate its JSON response as a training plan.

        Args:
            self: Generator holding the Ollama model name and server URL.
            system_prompt: Instructions defining the model's role and constraints.
            user_prompt: Request and retrieved evidence supplied for this generation.
            response_schema: Optional JSON schema passed to Ollama; defaults to the
                schema generated from :class:`TrainingPlan`.

        Returns:
            The validated plan.

        Raises:
            GenerationFailure: A ``RuntimeError`` subclass naming the cause:
                Ollama unreachable, model not installed, timeout (none of
                them worth a retry), or an unreadable answer (retryable).
        """
        started = perf_counter()
        logger.info("Ollama generation started model=%s", self.model)
        self._last_raw_output = None
        try:
            with generation_observation(
                "training plan generation", self.model
            ) as observation:
                try:
                    content, prompt_eval_tokens, eval_tokens = self._generate_content(
                        system_prompt,
                        user_prompt,
                        response_schema or TrainingPlan.model_json_schema(),
                    )
                except Exception as error:
                    finish_generation(observation, started, error=error)
                    record_external_call(
                        "training plan generation", started, None, None
                    )
                    raise
                finish_generation(observation, started, prompt_eval_tokens, eval_tokens)
                record_external_call(
                    "training plan generation",
                    started,
                    prompt_eval_tokens,
                    eval_tokens,
                )
            self._last_raw_output = content
            if isinstance(prompt_eval_tokens, bool) or not isinstance(
                prompt_eval_tokens, int
            ):
                prompt_eval_tokens = None
            plan = TrainingPlan.model_validate_json(content)
            logger.info(
                "Ollama generation returned valid JSON model=%s weeks=%s "
                "prompt_eval_tokens=%s elapsed_ms=%.1f",
                self.model,
                len(plan.weeks),
                prompt_eval_tokens if prompt_eval_tokens is not None else "unavailable",
                (perf_counter() - started) * 1000,
            )
            return plan
        except Exception as error:
            failure = classify_provider_error(
                error, model=self.model, timeout_seconds=GENERATION_TIMEOUT_SECONDS
            )
            logger.error(
                "Ollama generation failed model=%s status=%s error_type=%s "
                "elapsed_ms=%.1f",
                self.model,
                failure.status,
                type(error).__name__,
                (perf_counter() - started) * 1000,
            )
            raise failure from error

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's raw benchmark answer without schema validation."""
        started = perf_counter()
        logger.info("Ollama raw benchmark generation started model=%s", self.model)
        with generation_observation("benchmark generation", self.model) as observation:
            try:
                from ollama import Client

                client = Client(host=self.base_url, timeout=300)
                response = client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": 0.2, "num_ctx": 32_768},
                    think=False,
                )
                content = (
                    response["message"]["content"]
                    if isinstance(response, dict)
                    else response.message.content
                )
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Ollama returned an empty answer.")
                finish_generation(
                    observation,
                    started,
                    _response_count(response, "prompt_eval_count"),
                    _response_count(response, "eval_count"),
                )
                record_external_call(
                    "benchmark generation",
                    started,
                    _response_count(response, "prompt_eval_count"),
                    _response_count(response, "eval_count"),
                )
            except Exception as error:
                finish_generation(observation, started, error=error)
                record_external_call("benchmark generation", started, None, None)
                logger.error(
                    "Ollama raw benchmark generation failed model=%s "
                    "error_type=%s elapsed_ms=%.1f",
                    self.model,
                    type(error).__name__,
                    (perf_counter() - started) * 1000,
                )
                raise RuntimeError(
                    f"Ollama could not generate a benchmark answer with {self.model!r}: "
                    f"{error}"
                ) from error
        logger.info(
            "Ollama raw benchmark answer returned model=%s chars=%s elapsed_ms=%.1f",
            self.model,
            len(content),
            (perf_counter() - started) * 1000,
        )
        return content

    def generate_workout(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> Workout:
        """Call Ollama and validate its JSON response as one workout."""
        started = perf_counter()
        logger.info("Ollama workout adaptation started model=%s", self.model)
        try:
            with generation_observation(
                "workout adaptation", self.model
            ) as observation:
                try:
                    content, prompt_tokens, output_tokens = self._generate_content(
                        system_prompt,
                        user_prompt,
                        response_schema or Workout.model_json_schema(),
                    )
                except Exception as error:
                    finish_generation(observation, started, error=error)
                    record_external_call("workout adaptation", started, None, None)
                    raise
                finish_generation(observation, started, prompt_tokens, output_tokens)
                record_external_call(
                    "workout adaptation", started, prompt_tokens, output_tokens
                )
            workout = Workout.model_validate_json(content)
            logger.info(
                "Ollama workout adaptation returned valid JSON model=%s elapsed_ms=%.1f",
                self.model,
                (perf_counter() - started) * 1000,
            )
            return workout
        except Exception as error:
            logger.error(
                "Ollama workout adaptation failed model=%s error_type=%s elapsed_ms=%.1f",
                self.model,
                type(error).__name__,
                (perf_counter() - started) * 1000,
            )
            raise RuntimeError(
                "Ollama could not generate a valid workout adjustment. "
                f"Details: {error}"
            ) from error

    def _generate_content(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> tuple[str, int | None, int | None]:
        """Request structured JSON while retaining provider token diagnostics."""
        from ollama import Client

        client = Client(host=self.base_url, timeout=GENERATION_TIMEOUT_SECONDS)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format=response_schema,
            options={"temperature": 0.2, "num_ctx": 32_768},
            think=False,
        )
        prompt_eval_tokens = _response_count(response, "prompt_eval_count")
        eval_tokens = _response_count(response, "eval_count")
        content = (
            response["message"]["content"]
            if isinstance(response, dict)
            else response.message.content
        )
        return content, prompt_eval_tokens, eval_tokens


def _response_count(response: Any, name: str) -> int | None:
    """Return a valid Ollama token count, excluding booleans and other values."""
    value = (
        response.get(name)
        if isinstance(response, dict)
        else getattr(response, name, None)
    )
    return value if isinstance(value, int) and not isinstance(value, bool) else None
