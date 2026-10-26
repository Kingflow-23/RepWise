"""The only module that knows how to call Ollama."""

from __future__ import annotations

from typing import Protocol

from ..schemas.models import TrainingPlan


class PlanGenerator(Protocol):
    """Generate one validated structured plan."""

    def generate(self, system_prompt: str, user_prompt: str) -> TrainingPlan:
        """Return a plan or raise a provider/parsing error."""


class OllamaGenerator:
    """Ask a local Ollama model for JSON matching ``TrainingPlan``."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, system_prompt: str, user_prompt: str) -> TrainingPlan:
        """Call Ollama and parse its response with the shared schema."""
        from ollama import Client

        try:
            client = Client(host=self.base_url, timeout=120)
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=TrainingPlan.model_json_schema(),
                options={"temperature": 0.2},
            )
            content = (
                response["message"]["content"]
                if isinstance(response, dict)
                else response.message.content
            )
            return TrainingPlan.model_validate_json(content)
        except Exception as error:
            raise RuntimeError("Ollama could not generate a valid plan.") from error
