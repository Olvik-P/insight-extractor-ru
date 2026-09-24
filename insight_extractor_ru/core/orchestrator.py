"""Ties chunking, chunk analysis, evidence validation and aggregation together."""

from __future__ import annotations

from insight_extractor_ru.aggregation.merge import merge_criteria, merge_summaries
from insight_extractor_ru.aggregation.scoring import compute_overall_score
from insight_extractor_ru.aggregation.tag_normalization import CanonicalTagTable
from insight_extractor_ru.aggregation.talk_ratio import compute_talk_ratio
from insight_extractor_ru.analysis.chunk_analysis import analyze_chunk
from insight_extractor_ru.core.models import CallAnalysis, RawChunkResult, Transcript
from insight_extractor_ru.integrations.llm.base import LLMAdapter, LLMAdapterError
from insight_extractor_ru.preprocessing.chunking import chunk_by_turns
from insight_extractor_ru.validation.evidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    needs_manual_review,
)


def run_pipeline(
    transcript: Transcript,
    adapter: LLMAdapter,
    *,
    chunk_size: int = 4,
    chunk_overlap: int = 2,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    tag_table: CanonicalTagTable | None = None,
) -> CallAnalysis:
    """Run the full pipeline for one already-anonymized transcript.

    A chunk whose LLM response never validates (``LLMAdapterError``, e.g.
    ``LLMResponseInvalid`` after the adapter's fallback/retry is exhausted)
    is skipped rather than aborting the whole run — it simply contributes
    no evidence to any criterion or summary field, and is recorded in
    ``flags`` so the failure is visible rather than silent.
    """
    chunks = chunk_by_turns(transcript.turns, size=chunk_size, overlap=chunk_overlap)
    chunks_by_index = {chunk.chunk_index: chunk for chunk in chunks}

    chunk_results: list[RawChunkResult] = []
    pipeline_flags: list[str] = []
    for chunk in chunks:
        try:
            chunk_results.append(analyze_chunk(adapter, chunk))
        except LLMAdapterError:
            pipeline_flags.append(f"chunk_{chunk.chunk_index}_analysis_failed")

    merged_criteria = merge_criteria(chunk_results, chunks_by_index)
    merged_criteria["talk_ratio"] = compute_talk_ratio(transcript)

    summary = merge_summaries(chunk_results, chunks_by_index, tag_table)
    overall_score = compute_overall_score(merged_criteria, confidence_threshold)

    manual_review_flags = [
        f"{criterion_id}_manual_review"
        for criterion_id, result in merged_criteria.items()
        if needs_manual_review(result, confidence_threshold)
    ]

    return CallAnalysis(
        call_id=transcript.call_id,
        overall_score=overall_score,
        criteria=list(merged_criteria.values()),
        summary=summary,
        flags=pipeline_flags + manual_review_flags,
    )
