"""DeepSeek adapter: OpenAI-compatible chat completions with strict function calling."""

from __future__ import annotations

import httpx

from insight_extractor_ru.integrations.llm._http import (
    DEFAULT_TIMEOUT_SECONDS,
    post_chat_completion,
)
from insight_extractor_ru.integrations.llm.base import LLMAdapter, LLMAdapterError

_DEFAULT_BASE_URL = "https://api.deepseek.com"
_TOOL_NAME = "submit_chunk_analysis"


class DeepSeekAdapter(LLMAdapter):
    """Targets DeepSeek's OpenAI-compatible API.

    Uses function calling with ``strict: true`` (server-side JSON-Schema
    validation of the tool arguments) as the structured-output path, and
    ``response_format: {"type": "json_object"}`` as the fallback.

    DeepSeek's V4 models default to "thinking" (reasoning) mode, which
    rejects a forced ``tool_choice`` naming a specific function with a 400
    error ("Thinking mode does not support this tool_choice") — confirmed
    empirically against the live API. Both request paths here explicitly
    send ``"thinking": {"type": "disabled"}``: required for the structured
    path to work at all, and applied to the json_object path too since
    reasoning traces add cost/latency we don't need for direct extraction.

    ``deepseek-chat``/``deepseek-reasoner`` are legacy model ids, retired by
    DeepSeek on 2026-07-24 — current ids are ``deepseek-flash`` (default
    here, DeepSeek-V4.1-Flash) and ``deepseek-v4-pro`` for higher-quality
    reasoning. Pass ``model="deepseek-v4-pro"`` explicitly if flash's
    judgment quality proves insufficient for scoring nuanced criteria.
    """

    supports_structured_output = True

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-flash",
        base_url: str = _DEFAULT_BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request_structured(self, prompt: str, schema: dict) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": _TOOL_NAME,
                        "description": "Submit the chunk analysis result.",
                        "parameters": schema,
                        "strict": True,
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": _TOOL_NAME},
            },
        }
        body = post_chat_completion(
            self.client, f"{self.base_url}/chat/completions", self._headers(), payload
        )
        try:
            tool_calls = body["choices"][0]["message"]["tool_calls"]
            return tool_calls[0]["function"]["arguments"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMAdapterError(
                f"unexpected DeepSeek response shape: {body!r}"
            ) from exc

    def _request_json_mode(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }
        body = post_chat_completion(
            self.client, f"{self.base_url}/chat/completions", self._headers(), payload
        )
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMAdapterError(
                f"unexpected DeepSeek response shape: {body!r}"
            ) from exc
