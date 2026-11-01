"""The only module that knows how to call Ollama."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Protocol

from ..schemas.models import TrainingPlan, Workout

logger = logging.getLogger(__name__)


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
            RuntimeError: If the Ollama call or response validation fails.
        """
        started = perf_counter()
        logger.info("Ollama generation started model=%s", self.model)
        self._last_raw_output = None
        try:
            content, prompt_eval_tokens = self._generate_content(
                system_prompt,
                user_prompt,
                response_schema or TrainingPlan.model_json_schema(),
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
            logger.error(
                "Ollama generation failed model=%s error_type=%s elapsed_ms=%.1f",
                self.model,
                type(error).__name__,
                (perf_counter() - started) * 1000,
            )
            raise RuntimeError(
                "Ollama could not generate a valid plan. Confirm the local server is "
                f"running and install the configured model with `ollama pull {self.model}`. "
                f"Details: {error}"
            ) from error

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's raw benchmark answer without schema validation."""
        started = perf_counter()
        logger.info("Ollama raw benchmark generation started model=%s", self.model)
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
        except Exception as error:
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
            content, _ = self._generate_content(
                system_prompt,
                user_prompt,
                response_schema or Workout.model_json_schema(),
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
    ) -> tuple[str, int | None]:
        """Request structured JSON while retaining provider token diagnostics."""
        from ollama import Client

        client = Client(host=self.base_url, timeout=300)
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
        prompt_eval_tokens = (
            response.get("prompt_eval_count")
            if isinstance(response, dict)
            else getattr(response, "prompt_eval_count", None)
        )
        content = (
            response["message"]["content"]
            if isinstance(response, dict)
            else response.message.content
        )
        return content, prompt_eval_tokens
