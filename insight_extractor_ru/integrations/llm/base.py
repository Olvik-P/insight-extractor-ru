"""Provider-agnostic LLM adapter interface.

Implements the contract from
``openspec/changes/add-transcript-analysis-pipeline/specs/chunk-analysis``:
try the provider's native structured-output/function-calling mechanism
first, and fall back to ``json_object`` mode plus Pydantic validation and a
bounded retry (with the validation error fed back into the prompt) when
structured output is unavailable or the response fails validation.

Concrete adapters (DeepSeek/YandexGPT/GigaChat) only need to implement the
two provider-specific request methods below; the try-structured-then-fall-
back-then-retry flow lives here once, shared by all of them.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

T = TypeVar("T")


class LLMAdapterError(Exception):
    """Base class for adapter-level failures."""


class LLMResponseInvalid(LLMAdapterError):
    """Raised when a provider's response never validates, even after retry."""


class LLMAdapter(ABC):
    """Common interface used by ``analysis/`` regardless of LLM provider."""

    #: Whether this provider/model supports native structured output or
    #: function calling with schema enforcement. Subclasses may flip this
    #: per-instance if a specific model lacks the feature.
    supports_structured_output: bool = True

    #: Number of retries after the first attempt in the json_object fallback
    #: path, per the "retries once" requirement in chunk-analysis's spec.
    max_fallback_retries: int = 1

    @abstractmethod
    def _request_structured(self, prompt: str, schema: dict) -> str:
        """Call the provider's native structured-output/function-calling API.

        Must return the raw JSON *text* the provider produced (already
        extracted from whatever envelope the provider's API uses), which
        the base class parses and validates. Should raise
        :class:`LLMAdapterError` (or a subclass) if the call itself fails.
        """

    @abstractmethod
    def _request_json_mode(self, prompt: str) -> str:
        """Call the provider's ``json_object``-mode API.

        Must return the raw JSON text content of the response. Should
        raise :class:`LLMAdapterError` (or a subclass) if the call itself
        fails.
        """

    def analyze_chunk(
        self, prompt: str, schema: dict, validate: Callable[[dict], T]
    ) -> T:
        """Get a schema-shaped, validated result for one rendered chunk.

        ``validate`` receives the parsed JSON dict and must return the
        validated model (e.g. ``RawChunkResult.model_validate``) or raise
        on invalid data (e.g. ``pydantic.ValidationError``).
        """
        if self.supports_structured_output:
            try:
                raw_text = self._request_structured(prompt, schema)
                return validate(json.loads(raw_text))
            except Exception:
                # Fall through to the json_object + retry fallback path.
                pass

        return self._request_with_fallback(prompt, validate)

    def _request_with_fallback(
        self, prompt: str, validate: Callable[[dict], T]
    ) -> T:
        current_prompt = prompt
        last_error: Exception | None = None

        for _ in range(self.max_fallback_retries + 1):
            raw_text = self._request_json_mode(current_prompt)
            try:
                data = json.loads(raw_text)
                return validate(data)
            except Exception as exc:  # json.JSONDecodeError or ValidationError
                last_error = exc
                current_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response failed validation: {exc}\n"
                    "Return corrected JSON only, matching the requested schema."
                )

        raise LLMResponseInvalid(
            f"response failed validation after "
            f"{self.max_fallback_retries + 1} attempt(s): {last_error}"
        ) from last_error
