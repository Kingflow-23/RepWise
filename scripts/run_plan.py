"""Run the RepWise V1 pipeline with a JSONL chunk file and Ollama."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.config import AppConfig
from src.generation import PlanPipeline
from src.llm import OllamaGenerator
from src.retrieval import JsonlRetriever
from src.schemas import PlanRequest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a workout plan from retrieved transcript chunks."
    )
    parser.add_argument(
        "--request", required=True, help="Path to a PlanRequest JSON file"
    )
    parser.add_argument(
        "--chunks", required=True, help="Path to Axel's chunk JSONL file"
    )
    arguments = parser.parse_args()

    config = AppConfig.from_environment()
    request = PlanRequest.model_validate_json(
        Path(arguments.request).read_text(encoding="utf-8")
    )
    pipeline = PlanPipeline(
        retriever=JsonlRetriever(arguments.chunks),
        generator=OllamaGenerator(config.ollama_model, config.ollama_url),
        top_k=config.top_k,
    )
    print(pipeline.create_plan(request).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
