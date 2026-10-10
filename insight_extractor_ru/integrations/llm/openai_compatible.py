"""Shared plumbing for the OpenAI-compatible chat-completions adapters."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import httpx

from insight_extractor_ru.integrations.llm._http import (
    DEFAULT_TIMEOUT_SECONDS,
    post_chat_completion,
)
from insight_extractor_ru.integrations.llm.base import LLMAdapter, LLMAdapterError


class ChatCompletionsAdapter(LLMAdapter):
    """Base for providers exposing ``POST {base_url}/chat/completions``.

    Subclasses supply the provider name (for error messages), the auth
    header, and the two ``_request_*`` payloads; transport, URL building
    and response-shape error handling live here once.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name, used in error messages."""

    def __init__(
        self, model: str, base_url: str, client: httpx.Client | None = None
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)

    @abstractmethod
    def _auth_header(self) -> str:
        """Value of the ``Authorization`` header."""

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": self._auth_header(),
            "Content-Type": "application/json",
        }

    def _complete(
        self, payload: dict, extract: Callable[[dict], str]
    ) -> str:
        """POST ``payload`` and pull the result text out of the response body."""
        body = post_chat_completion(
            self.client, f"{self.base_url}/chat/completions", self._headers(), payload
        )
        try:
            return extract(body)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMAdapterError(
                f"unexpected {self.provider_name} response shape: {body!r}"
            ) from exc

    @staticmethod
    def _message_content(body: dict) -> str:
        return body["choices"][0]["message"]["content"]
