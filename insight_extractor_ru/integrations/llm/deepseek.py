"""DeepSeek adapter: OpenAI-compatible chat completions with strict function calling."""

from __future__ import annotations

import httpx

from insight_extractor_ru.integrations.llm.openai_compatible import (
    ChatCompletionsAdapter,
)

_DEFAULT_BASE_URL = "https://api.deepseek.com"
_TOOL_NAME = "submit_chunk_analysis"


class DeepSeekAdapter(ChatCompletionsAdapter):
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
    provider_name = "DeepSeek"

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-flash",
        base_url: str = _DEFAULT_BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(model=model, base_url=base_url, client=client)
        self.api_key = api_key

    def _auth_header(self) -> str:
        return f"Bearer {self.api_key}"

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
        return self._complete(payload, self._tool_call_arguments)

    @staticmethod
    def _tool_call_arguments(body: dict) -> str:
        return body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]

    def _request_json_mode(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }
        return self._complete(payload, self._message_content)
