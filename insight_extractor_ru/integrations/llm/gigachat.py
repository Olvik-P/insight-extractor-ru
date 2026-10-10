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

from insight_extractor_ru.integrations.llm.openai_compatible import (
    ChatCompletionsAdapter,
)

_DEFAULT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
_FUNCTION_NAME = "submit_chunk_analysis"


class GigaChatAdapter(ChatCompletionsAdapter):
    """Targets GigaChat's chat completions API.

    Uses ``function_calling`` (GigaChat's default structured-output
    mechanism) as the structured-output path, and
    ``response_format: {"type": "json_object"}`` as the fallback. GigaChat
    does not support parallel tool calls, which is not a constraint here
    since only one function is ever requested per call.
    """

    supports_structured_output = True
    provider_name = "GigaChat"

    def __init__(
        self,
        access_token: str,
        model: str = "GigaChat",
        base_url: str = _DEFAULT_BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(model=model, base_url=base_url, client=client)
        self.access_token = access_token

    def _auth_header(self) -> str:
        return f"Bearer {self.access_token}"

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
        return self._complete(payload, self._function_call_arguments)

    @staticmethod
    def _function_call_arguments(body: dict) -> str:
        return body["choices"][0]["message"]["function_call"]["arguments"]

    def _request_json_mode(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        return self._complete(payload, self._message_content)
