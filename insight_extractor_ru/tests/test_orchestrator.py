from __future__ import annotations

import json

from insight_extractor_ru.core.models import Chunk, Role, Transcript, Turn
from insight_extractor_ru.core.orchestrator import run_pipeline
from insight_extractor_ru.integrations.llm.base import LLMAdapter, LLMAdapterError

_TRANSCRIPT = Transcript(
    call_id="call-e2e-1",
    turns=[
        Turn(turn_index=1, role=Role.MANAGER, text="Здравствуйте! Какая у вас задача?"),
        Turn(
            turn_index=2,
            role=Role.CLIENT,
            text="Нужно автоматизировать обработку заявок, сейчас очень дорого.",
        ),
        Turn(turn_index=3, role=Role.MANAGER, text="Понял, а какой у вас бюджет?"),
        Turn(turn_index=4, role=Role.CLIENT, text="Пока не определились с бюджетом."),
        Turn(turn_index=5, role=Role.MANAGER, text="Хорошо, что скажете про сроки?"),
        Turn(turn_index=6, role=Role.CLIENT, text="Не сейчас, вернёмся к этому позже."),
    ],
)


def _response_for_chunk(chunk: Chunk) -> dict:
    """Build a plausible, schema-valid LLM response referencing this chunk's turns."""
    first_turn = chunk.turns[0]
    return {
        "criteria": [
            {
                "id": "discovery_depth",
                "score": 2,
                "reason": "Клиент назвал задачу и боль.",
                "evidence": [{"turn_index": first_turn.turn_index}],
                "confidence": 0.8,
                "flags": [],
            }
        ],
        "summary_fragment": {
            "client_pain": "дорого обрабатывать заявки вручную"
            if any("дорого" in t.text for t in chunk.turns)
            else None,
            "objections": [
                {"text": "дорого", "evidence": {"turn_index": t.turn_index}}
                for t in chunk.turns
                if "дорого" in t.text
            ]
            + [
                {"text": "не сейчас", "evidence": {"turn_index": t.turn_index}}
                for t in chunk.turns
                if "Не сейчас" in t.text
            ],
            "action_items": [],
        },
    }


class _ScriptedAdapter(LLMAdapter):
    """Returns a schema-valid, chunk-aware response for every chunk it sees."""

    def __init__(self) -> None:
        self.seen_prompts: list[str] = []

    def _request_structured(self, prompt: str, schema: dict) -> str:
        self.seen_prompts.append(prompt)
        # Recover which turns were sent by parsing the rendered "[n] ROLE: text" lines.
        turn_indices = [
            int(line.split("]")[0].lstrip("["))
            for line in prompt.splitlines()
            if line.startswith("[")
        ]
        chunk = Chunk(
            chunk_index=0,
            turns=[t for t in _TRANSCRIPT.turns if t.turn_index in turn_indices],
        )
        return json.dumps(_response_for_chunk(chunk))

    def _request_json_mode(self, prompt: str) -> str:
        raise LLMAdapterError("fallback not exercised in this test")


class TestRunPipelineEndToEnd:
    def test_full_pipeline_produces_a_complete_call_analysis(self) -> None:
        adapter = _ScriptedAdapter()

        result = run_pipeline(_TRANSCRIPT, adapter)

        assert result.call_id == "call-e2e-1"
        criterion_ids = {c.id for c in result.criteria}
        assert "discovery_depth" in criterion_ids
        assert "talk_ratio" in criterion_ids  # deterministic, always present

        assert result.summary.client_pain == "дорого обрабатывать заявки вручную"
        assert "дорого" in result.summary.objections
        assert "не сейчас" in result.summary.objections

        assert 0.0 <= result.overall_score <= 100.0

        discovery = next(c for c in result.criteria if c.id == "discovery_depth")
        assert discovery.resolved_evidence  # evidence-by-reference resolved to text

    def test_chunk_analysis_failure_is_recorded_and_does_not_crash(self) -> None:
        class _AlwaysFailingAdapter(LLMAdapter):
            def _request_structured(self, prompt: str, schema: dict) -> str:
                raise LLMAdapterError("boom")

            def _request_json_mode(self, prompt: str) -> str:
                raise LLMAdapterError("boom")

        result = run_pipeline(_TRANSCRIPT, _AlwaysFailingAdapter())

        assert any("analysis_failed" in flag for flag in result.flags)
        # No LLM-scored criteria resolved evidence, but talk_ratio still did.
        assert result.criteria  # talk_ratio is always present
