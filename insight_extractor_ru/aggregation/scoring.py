"""Overall-score computation, excluding manual-review criteria."""

from __future__ import annotations

from insight_extractor_ru.analysis.scorecard import ALL_CRITERIA
from insight_extractor_ru.core.models import ResolvedCriterionResult
from insight_extractor_ru.validation.evidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    needs_manual_review,
)


def compute_overall_score(
    criteria: dict[str, ResolvedCriterionResult],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> float:
    """Weighted sum over criteria not routed to manual review, normalized to 0-100.

    A criterion missing from ``criteria`` altogether, or below the
    confidence threshold, is excluded from both the numerator and the
    weight denominator — the remaining criteria's weights are effectively
    renormalized rather than silently capping the achievable score.
    """
    included_weight = 0.0
    weighted_sum = 0.0
    for scorecard_criterion in ALL_CRITERIA:
        result = criteria.get(scorecard_criterion.id)
        if result is None or needs_manual_review(result, threshold):
            continue
        included_weight += scorecard_criterion.weight
        weighted_sum += scorecard_criterion.weight * (
            result.score / scorecard_criterion.max_score
        )

    if included_weight == 0.0:
        return 0.0
    return round(100.0 * weighted_sum / included_weight, 2)
