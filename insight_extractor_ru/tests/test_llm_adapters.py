from __future__ import annotations

import json

import httpx
import pytest

from insight_extractor_ru.integrations.llm.base import (
    LLMAdapter,
    LLMAdapterError,
    LLMResponseInvalid,
)
from insight_extractor_ru.integrations.llm.deepseek import DeepSeekAdapter
from insight_extractor_ru.integrations.llm.gigachat import GigaChatAdapter
from insight_extractor_ru.integrations.llm.yandexgpt import YandexGPTAdapter

_SCHEMA = {"type": "object", "properties": {"score": {"type": "integer"}}}


def _validate_score(data: dict) -> dict:
    if "score" not in data:
        raise ValueError("missing 'score'")
    return data


class _FakeAdapter(LLMAdapter):
    """Adapter with scripted responses, for testing the base retry/fallback flow."""

    def __init__(
        self,
        structured_result: str | Exception | None = None,
        json_mode_results: list[str] | None = None,
    ) -> None:
        self._structured_result = structured_result
        self._json_mode_results = list(json_mode_results or [])
        self.json_mode_prompts: list[str] = []

    def _request_structured(self, prompt: str, schema: dict) -> str:
        if isinstance(self._structured_result, Exception):
            raise self._structured_result
        assert self._structured_result is not None
        return self._structured_result

    def _request_json_mode(self, prompt: str) -> str:
        self.json_mode_prompts.append(prompt)
        return self._json_mode_results.pop(0)


class TestLLMAdapterBaseFlow:
    def test_uses_structured_result_when_supported_and_valid(self) -> None:
        adapter = _FakeAdapter(structured_result=json.dumps({"score": 2}))

        result = adapter.analyze_chunk("prompt", _SCHEMA, _validate_score)

        assert result == {"score": 2}

    def test_falls_back_to_json_mode_when_structured_call_fails(self) -> None:
        adapter = _FakeAdapter(
            structured_result=LLMAdapterError("not supported"),
            json_mode_results=[json.dumps({"score": 3})],
        )

        result = adapter.analyze_chunk("prompt", _SCHEMA, _validate_score)

        assert result == {"score": 3}

    def test_retries_once_with_validation_error_fed_back_into_prompt(self) -> None:
        adapter = _FakeAdapter(
            structured_result=LLMAdapterError("not supported"),
            json_mode_results=[
                json.dumps({"oops": 1}),  # fails validation
                json.dumps({"score": 5}),  # succeeds on retry
            ],
        )

        result = adapter.analyze_chunk("prompt", _SCHEMA, _validate_score)

        assert result == {"score": 5}
        assert len(adapter.json_mode_prompts) == 2
        assert "failed validation" in adapter.json_mode_prompts[1]

    def test_raises_after_exhausting_retries(self) -> None:
        adapter = _FakeAdapter(
            structured_result=LLMAdapterError("not supported"),
            json_mode_results=[
                json.dumps({"oops": 1}),
                json.dumps({"oops": 2}),
            ],
        )

        with pytest.raises(LLMResponseInvalid):
            adapter.analyze_chunk("prompt", _SCHEMA, _validate_score)

    def test_skips_structured_call_when_provider_does_not_support_it(self) -> None:
        adapter = _FakeAdapter(json_mode_results=[json.dumps({"score": 1})])
        adapter.supports_structured_output = False

        result = adapter.analyze_chunk("prompt", _SCHEMA, _validate_score)

        assert result == {"score": 1}


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestDeepSeekAdapter:
    def test_structured_request_extracts_tool_call_arguments(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert payload["tools"][0]["function"]["strict"] is True
            # DeepSeek's V4 models reject a forced tool_choice while in
            # "thinking" mode (400: "Thinking mode does not support this
            # tool_choice") — confirmed against the live API — so thinking
            # must be explicitly disabled for structured output to work.
            assert payload["thinking"] == {"type": "disabled"}
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "arguments": json.dumps({"score": 2})
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            )

        adapter = DeepSeekAdapter(
            api_key="test", client=_mock_client(handler)
        )

        result = adapter.analyze_chunk("prompt", _SCHEMA, _validate_score)

        assert result == {"score": 2}

    def test_json_mode_fallback_extracts_message_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            if "tools" in payload:
                return httpx.Response(500, json={"error": "unsupported"})
            assert payload["thinking"] == {"type": "disabled"}
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps({"score": 4})}}]},
            )

        adapter = DeepSeekAdapter(api_key="test", client=_mock_client(handler))

        result = adapter.analyze_chunk("prompt", _SCHEMA, _validate_score)

        assert result == {"score": 4}


class TestYandexGPTAdapter:
    def test_structured_request_uses_json_schema_response_format(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert payload["response_format"]["type"] == "json_schema"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps({"score": 1})}}]},
            )

        adapter = YandexGPTAdapter(api_key="test", client=_mock_client(handler))

        result = adapter.analyze_chunk("prompt", _SCHEMA, _validate_score)

        assert result == {"score": 1}


class TestGigaChatAdapter:
    def test_structured_request_extracts_function_call_arguments(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            assert payload["function_call"]["name"] == "submit_chunk_analysis"
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "function_call": {
                                    "arguments": json.dumps({"score": 3})
                                }
                            }
                        }
                    ]
                },
            )

        adapter = GigaChatAdapter(
            access_token="test", client=_mock_client(handler)
        )

        result = adapter.analyze_chunk("prompt", _SCHEMA, _validate_score)

        assert result == {"score": 3}

    def test_malformed_response_shape_raises_adapter_error_and_falls_back(
        self,
    ) -> None:
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            calls.append(payload)
            if "function_call" in payload:
                return httpx.Response(200, json={"choices": [{"message": {}}]})
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps({"score": 9})}}]},
            )

        adapter = GigaChatAdapter(
            access_token="test", client=_mock_client(handler)
        )

        result = adapter.analyze_chunk("prompt", _SCHEMA, _validate_score)

        assert result == {"score": 9}
        assert len(calls) == 2
