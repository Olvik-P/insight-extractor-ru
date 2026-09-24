"""Small shared HTTP helper for the OpenAI-compatible chat-completions adapters."""

from __future__ import annotations

import httpx

from insight_extractor_ru.integrations.llm.base import LLMAdapterError

#: httpx.Client() defaults to a 5-second timeout for every phase (connect,
#: read, write, pool) — too short for a real LLM completion, which can
#: legitimately take well over 5 seconds to generate a full JSON response.
#: Live testing against DeepSeek showed intermittent, silently-swallowed
#: chunk failures traced back to exactly this: a handful of calls hit the
#: default 5s read timeout while most completed faster and succeeded,
#: producing failures with no discernible pattern in the request content.
DEFAULT_TIMEOUT_SECONDS = 60.0


def post_chat_completion(
    client: httpx.Client, url: str, headers: dict[str, str], payload: dict
) -> dict:
    """POST a chat-completions request and return the parsed JSON body.

    Raises :class:`LLMAdapterError` on a transport failure or a non-2xx
    response, so adapter code can treat any failure uniformly.
    """
    try:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMAdapterError(f"request to {url} failed: {exc}") from exc
    return response.json()
