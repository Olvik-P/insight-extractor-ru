"""YandexGPT adapter: AI Studio's OpenAI-compatible endpoint with JSON Schema."""

from __future__ import annotations

import httpx

from insight_extractor_ru.integrations.llm._http import (
    DEFAULT_TIMEOUT_SECONDS,
    post_chat_completion,
)
from insight_extractor_ru.integrations.llm.base import LLMAdapter, LLMAdapterError

_DEFAULT_BASE_URL = "https://ai.api.cloud.yandex.net/v1"


class YandexGPTAdapter(LLMAdapter):
    """Targets Yandex AI Studio's OpenAI-compatible chat completions API.

    Uses ``response_format: {"type": "json_schema", ...}`` as the
    structured-output path, and ``response_format: {"type": "json_object"}``
    as the fallback.
    """

    supports_structured_output = True

    def __init__(
        self,
        api_key: str,
        model: str = "yandexgpt/latest",
        base_url: str = _DEFAULT_BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request_structured(self, prompt: str, schema: dict) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "chunk_analysis",
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        body = post_chat_completion(
            self.client, f"{self.base_url}/chat/completions", self._headers(), payload
        )
        return self._extract_content(body)

    def _request_json_mode(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        body = post_chat_completion(
            self.client, f"{self.base_url}/chat/completions", self._headers(), payload
        )
        return self._extract_content(body)

    @staticmethod
    def _extract_content(body: dict) -> str:
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMAdapterError(
                f"unexpected YandexGPT response shape: {body!r}"
            ) from exc
