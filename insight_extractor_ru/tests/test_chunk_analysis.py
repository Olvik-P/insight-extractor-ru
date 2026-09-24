from __future__ import annotations

import json

import pytest

from insight_extractor_ru.analysis.chunk_analysis import analyze_chunk
from insight_extractor_ru.analysis.prompts import build_chunk_prompt
from insight_extractor_ru.core.models import Chunk, Role, Turn
from insight_extractor_ru.integrations.llm.base import LLMAdapter, LLMResponseInvalid

_CHUNK = Chunk(
    chunk_index=0,
    turns=[
        Turn(turn_index=1, role=Role.MANAGER, text="Какой у вас бюджет?"),
        Turn(turn_index=2, role=Role.CLIENT, text="Пока не определились."),
    ],
)

_VALID_RESPONSE = {
    "criteria": [
        {
            "id": "budget_qualification",
            "score": 1,
            "reason": "Бюджет не определён",
            "evidence": [{"turn_index": 2}],
            "confidence": 0.7,
            "flags": [],
        }
    ],
    "summary_fragment": {
        "client_pain": None,
        "objections": [],
        "action_items": [],
    },
}


class _StubAdapter(LLMAdapter):
    def __init__(self, structured_json: str | None = None) -> None:
        self.structured_json = structured_json

    def _request_structured(self, prompt: str, schema: dict) -> str:
        if self.structured_json is None:
            raise RuntimeError("structured not configured")
        return self.structured_json

    def _request_json_mode(self, prompt: str) -> str:
        raise RuntimeError("fallback not configured for this test")


class TestBuildChunkPrompt:
    def test_prompt_contains_turn_labeled_text_and_criteria(self) -> None:
        prompt = build_chunk_prompt(_CHUNK)

        assert "[1] МЕНЕДЖЕР: Какой у вас бюджет?" in prompt
        assert "[2] КЛИЕНТ: Пока не определились." in prompt
        assert "budget_qualification" in prompt
        assert "talk_ratio" not in prompt  # deterministic, not LLM-scored


class TestAnalyzeChunk:
    def test_schema_valid_response_becomes_raw_chunk_result(self) -> None:
        adapter = _StubAdapter(structured_json=json.dumps(_VALID_RESPONSE))

        result = analyze_chunk(adapter, _CHUNK)

        assert result.chunk_index == 0
        assert result.criteria[0].id == "budget_qualification"
        assert result.criteria[0].evidence[0].turn_index == 2

    def test_schema_invalid_response_raises_after_fallback_exhausted(self) -> None:
        adapter = _StubAdapter(structured_json=None)

        class _FailingFallback(_StubAdapter):
            def _request_json_mode(self, prompt: str) -> str:
                return json.dumps({"criteria": [{"id": "not_a_real_criterion"}]})

        with pytest.raises(LLMResponseInvalid):
            analyze_chunk(_FailingFallback(), _CHUNK)
