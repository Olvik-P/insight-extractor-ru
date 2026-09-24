"""CLI entrypoint: analyze an already-anonymized transcript file."""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from insight_extractor_ru.core.models import Transcript
from insight_extractor_ru.core.orchestrator import run_pipeline
from insight_extractor_ru.integrations.llm.base import LLMAdapter
from insight_extractor_ru.integrations.llm.deepseek import DeepSeekAdapter
from insight_extractor_ru.integrations.llm.gigachat import GigaChatAdapter
from insight_extractor_ru.integrations.llm.yandexgpt import YandexGPTAdapter

_PROVIDER_ENV_VARS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "yandexgpt": "YANDEXGPT_API_KEY",
    "gigachat": "GIGACHAT_ACCESS_TOKEN",
}


def _build_adapter(provider: str) -> LLMAdapter:
    env_var = _PROVIDER_ENV_VARS[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(
            f"missing environment variable {env_var} for provider {provider!r}"
        )

    if provider == "deepseek":
        return DeepSeekAdapter(api_key=api_key)
    if provider == "yandexgpt":
        return YandexGPTAdapter(api_key=api_key)
    return GigaChatAdapter(access_token=api_key)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="insight-extractor-ru",
        description="Analyze an already-anonymized B2B call transcript.",
    )
    parser.add_argument(
        "transcript", help="Path to an anonymized transcript JSON file"
    )
    parser.add_argument(
        "--provider", choices=sorted(_PROVIDER_ENV_VARS), default="deepseek"
    )
    parser.add_argument("--output", help="Write result JSON here instead of stdout")
    args = parser.parse_args(argv)

    with open(args.transcript, encoding="utf-8") as transcript_file:
        transcript = Transcript.model_validate(json.load(transcript_file))

    adapter = _build_adapter(args.provider)
    result = run_pipeline(transcript, adapter)

    output_json = result.model_dump_json(indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(output_json)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
