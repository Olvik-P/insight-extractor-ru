"""GigaChat adapter: function_calling and json_mode structured output.

GigaChat authenticates with a short-lived OAuth Bearer access token obtained
via a separate client-credentials exchange against Sber's OAuth endpoint
(using a long-lived "Authorization key"). Obtaining/refreshing that token is
out of scope for this adapter — callers pass an already-obtained
``access_token`` in, keeping this adapter's own contract (and its tests)
independent of the OAuth flow and its expiry handling.
"""

from __future__ import annotations

import httpx

from insight_extractor_ru.integrations.llm._http import (
    DEFAULT_TIMEOUT_SECONDS,
    post_chat_completion,
)
from insight_extractor_ru.integrations.llm.base import LLMAdapter, LLMAdapterError

_DEFAULT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
_FUNCTION_NAME = "submit_chunk_analysis"


class GigaChatAdapter(LLMAdapter):
    """Targets GigaChat's chat completions API.

    Uses ``function_calling`` (GigaChat's default structured-output
    mechanism) as the structured-output path, and
    ``response_format: {"type": "json_object"}`` as the fallback. GigaChat
    does not support parallel tool calls, which is not a constraint here
    since only one function is ever requested per call.
    """

    supports_structured_output = True

    def __init__(
        self,
        access_token: str,
        model: str = "GigaChat",
        base_url: str = _DEFAULT_BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        self.access_token = access_token
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    def _request_structured(self, prompt: str, schema: dict) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "functions": [
                {
                    "name": _FUNCTION_NAME,
                    "description": "Submit the chunk analysis result.",
                    "parameters": schema,
                }
            ],
            "function_call": {"name": _FUNCTION_NAME},
        }
        body = post_chat_completion(
            self.client, f"{self.base_url}/chat/completions", self._headers(), payload
        )
        try:
            return body["choices"][0]["message"]["function_call"]["arguments"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMAdapterError(
                f"unexpected GigaChat response shape: {body!r}"
            ) from exc

    def _request_json_mode(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        body = post_chat_completion(
            self.client, f"{self.base_url}/chat/completions", self._headers(), payload
        )
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMAdapterError(
                f"unexpected GigaChat response shape: {body!r}"
            ) from exc
