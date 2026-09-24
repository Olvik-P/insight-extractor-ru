"""Chunk-analysis orchestration: render chunk -> invoke adapter -> validate."""

from __future__ import annotations

from insight_extractor_ru.analysis.prompts import build_chunk_prompt
from insight_extractor_ru.analysis.scorecard import LLM_SCORED_CRITERION_IDS
from insight_extractor_ru.core.models import Chunk, LLMChunkResponse, RawChunkResult
from insight_extractor_ru.integrations.llm.base import LLMAdapter

_SCHEMA = LLMChunkResponse.model_json_schema()


def _validate(data: dict) -> LLMChunkResponse:
    response = LLMChunkResponse.model_validate(data)
    unknown_ids = {c.id for c in response.criteria} - LLM_SCORED_CRITERION_IDS
    if unknown_ids:
        raise ValueError(f"unknown criterion id(s): {sorted(unknown_ids)}")
    return response


def analyze_chunk(adapter: LLMAdapter, chunk: Chunk) -> RawChunkResult:
    """Run one chunk through the LLM and return a schema-validated result.

    Propagates whatever the adapter raises (e.g. ``LLMResponseInvalid``) if
    the response never validates, even after the adapter's built-in retry.
    Callers should catch that and treat the chunk as contributing no
    evidence, per chunk-analysis's "response still invalid" scenario,
    rather than letting malformed data flow downstream.
    """
    prompt = build_chunk_prompt(chunk)
    response = adapter.analyze_chunk(prompt, _SCHEMA, _validate)
    return RawChunkResult(
        chunk_index=chunk.chunk_index,
        criteria=response.criteria,
        summary_fragment=response.summary_fragment,
    )
