"""The only module that knows how to call Ollama."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Protocol

from ..schemas.models import TrainingPlan

logger = logging.getLogger(__name__)


class PlanGenerator(Protocol):
    """Generate one validated structured plan."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> TrainingPlan:
        """Return a plan or raise a provider/parsing error."""


class OllamaGenerator:
    """Ask a local Ollama model for JSON matching ``TrainingPlan``."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> TrainingPlan:
        """Call Ollama and parse its response with the shared schema."""
        from ollama import Client

        started = perf_counter()
        logger.info("Ollama generation started model=%s", self.model)
        try:
            client = Client(host=self.base_url, timeout=120)
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=response_schema or TrainingPlan.model_json_schema(),
                options={"temperature": 0.2},
            )
            content = (
                response["message"]["content"]
                if isinstance(response, dict)
                else response.message.content
            )
            plan = TrainingPlan.model_validate_json(content)
            logger.info(
                "Ollama generation returned valid JSON model=%s weeks=%s elapsed_ms=%.1f",
                self.model,
                len(plan.weeks),
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
