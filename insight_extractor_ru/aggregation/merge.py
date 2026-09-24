"""Merging per-chunk results into the final call-level contract.

Two different merge strategies, matched to the shape of each field (design.md
Decisions 7-8; see also
``openspec/changes/add-transcript-analysis-pipeline/specs/result-aggregation``):

- ``merge_criteria``: Map-Rerank — extraction/QA-shaped, one best chunk wins
  per criterion.
- ``merge_summaries``: Reduce — summarization-shaped, information is unioned
  across chunks rather than picked from one "winning" chunk.
"""

from __future__ import annotations

from insight_extractor_ru.aggregation.tag_normalization import CanonicalTagTable
from insight_extractor_ru.core.models import (
    Chunk,
    RawChunkResult,
    ResolvedActionItem,
    ResolvedCriterionResult,
    ResolvedEvidence,
    Summary,
)
from insight_extractor_ru.validation.evidence import resolve_evidence


def merge_criteria(
    chunk_results: list[RawChunkResult], chunks_by_index: dict[int, Chunk]
) -> dict[str, ResolvedCriterionResult]:
    """Map-Rerank merge: per criterion id, keep the best-evidenced chunk result.

    "Best" ranks first by how much resolved evidence a result cites, and
    only uses confidence as a tie-break — not confidence alone. Live
    testing against DeepSeek found a concrete failure mode for confidence-
    only ranking: an objection (turn 8) and its handling (turn 9) both land
    in one chunk together, but turn 8 alone also lands in an earlier
    overlapping chunk that never saw the response. That earlier chunk
    confidently (0.9) reported "unhandled", while the chunk that actually
    saw both turns reported "handled" with citation of both turns — and on
    a different sampling run, the confidence gap even went the wrong way,
    so ranking by confidence alone occasionally picked the under-informed
    verdict. The chunk that saw the full exchange systematically resolves
    *more* evidence turns for the same criterion (it has more to cite), so
    evidence breadth is a more reliable signal of "this chunk had the
    fuller picture" than the model's self-reported confidence.

    If no chunk has resolvable evidence for a criterion, the (equivalent)
    empty/zero-confidence result is kept.
    """
    resolved_by_id: dict[str, list[ResolvedCriterionResult]] = {}
    for chunk_result in chunk_results:
        chunk = chunks_by_index[chunk_result.chunk_index]
        for criterion in chunk_result.criteria:
            resolved = resolve_evidence(criterion, chunk)
            resolved_by_id.setdefault(criterion.id, []).append(resolved)

    merged: dict[str, ResolvedCriterionResult] = {}
    for criterion_id, results in resolved_by_id.items():
        with_evidence = [r for r in results if r.resolved_evidence]
        merged[criterion_id] = (
            max(
                with_evidence,
                key=lambda r: (len(r.resolved_evidence), r.confidence),
            )
            if with_evidence
            else results[0]
        )
    return merged


def merge_summaries(
    chunk_results: list[RawChunkResult],
    chunks_by_index: dict[int, Chunk],
    tag_table: CanonicalTagTable | None = None,
) -> Summary:
    """Reduce merge: union summary fields across all chunks.

    ``client_pain`` is a scalar field, not a list — the first non-empty
    value encountered (in chunk order) is kept, since there is nothing to
    union for a single string. Objections are deduplicated via the
    canonical tag table (embedding cosine similarity). Action items are
    unioned with their evidence resolved against the chunk that reported
    them; an out-of-range evidence reference is dropped, same rule as for
    criteria.
    """
    tag_table = tag_table or CanonicalTagTable()

    client_pain: str | None = None
    raw_objections: list[str] = []
    action_items: list[ResolvedActionItem] = []

    for chunk_result in chunk_results:
        fragment = chunk_result.summary_fragment
        if client_pain is None and fragment.client_pain:
            client_pain = fragment.client_pain

        raw_objections.extend(obj.text for obj in fragment.objections)

        chunk = chunks_by_index[chunk_result.chunk_index]
        for item in fragment.action_items:
            turn = chunk.turn_by_index(item.evidence.turn_index)
            if turn is None:
                continue
            action_items.append(
                ResolvedActionItem(
                    owner=item.owner,
                    task=item.task,
                    resolved_evidence=ResolvedEvidence(
                        turn_index=turn.turn_index, text=turn.text
                    ),
                )
            )

    return Summary(
        client_pain=client_pain,
        objections=tag_table.normalize(raw_objections),
        action_items=action_items,
    )
