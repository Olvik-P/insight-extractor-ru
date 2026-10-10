"""YandexGPT adapter: AI Studio's OpenAI-compatible endpoint with JSON Schema."""

from __future__ import annotations

import httpx

from insight_extractor_ru.integrations.llm.openai_compatible import (
    ChatCompletionsAdapter,
)

_DEFAULT_BASE_URL = "https://ai.api.cloud.yandex.net/v1"


class YandexGPTAdapter(ChatCompletionsAdapter):
    """Targets Yandex AI Studio's OpenAI-compatible chat completions API.

    Uses ``response_format: {"type": "json_schema", ...}`` as the
    structured-output path, and ``response_format: {"type": "json_object"}``
    as the fallback.
    """

    supports_structured_output = True
    provider_name = "YandexGPT"

    def __init__(
        self,
        api_key: str,
        model: str = "yandexgpt/latest",
        base_url: str = _DEFAULT_BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(model=model, base_url=base_url, client=client)
        self.api_key = api_key

    def _auth_header(self) -> str:
        return f"Api-Key {self.api_key}"

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
        return self._complete(payload, self._message_content)

    def _request_json_mode(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        return self._complete(payload, self._message_content)
