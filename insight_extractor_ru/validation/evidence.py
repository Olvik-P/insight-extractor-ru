"""Evidence resolution and the confidence-based manual-review safety net.

See ``openspec/changes/add-transcript-analysis-pipeline/specs/evidence-validation``.
"""

from __future__ import annotations

from insight_extractor_ru.core.models import (
    Chunk,
    CriterionResult,
    ResolvedCriterionResult,
    ResolvedEvidence,
)

DEFAULT_CONFIDENCE_THRESHOLD = 0.6
NO_VALID_QUOTES_FLAG = "no_valid_quotes"


def resolve_evidence(
    criterion: CriterionResult, chunk: Chunk
) -> ResolvedCriterionResult:
    """Resolve a criterion's evidence references to literal chunk text.

    A reference to a ``turn_index`` outside ``chunk`` (the chunk that
    produced this result) is dropped rather than resolved. Resolved text is
    copied verbatim from the turn — including any PII token it contains —
    with no transformation, so tokens are never altered or reversed here.
    """
    resolved: list[ResolvedEvidence] = [
        ResolvedEvidence(turn_index=turn.turn_index, text=turn.text)
        for ref in criterion.evidence
        if (turn := chunk.turn_by_index(ref.turn_index)) is not None
    ]

    flags = list(criterion.flags)
    confidence = criterion.confidence
    if not resolved:
        confidence = 0.0
        if NO_VALID_QUOTES_FLAG not in flags:
            flags.append(NO_VALID_QUOTES_FLAG)

    return ResolvedCriterionResult(
        id=criterion.id,
        score=criterion.score,
        reason=criterion.reason,
        resolved_evidence=resolved,
        confidence=confidence,
        flags=flags,
    )


def needs_manual_review(
    result: ResolvedCriterionResult,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """Whether a resolved criterion result falls below the confidence threshold."""
    return result.confidence < threshold
