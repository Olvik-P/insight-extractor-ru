"""Deterministic talk-ratio scoring — not LLM-judged (design.md Decision 10)."""

from __future__ import annotations

from insight_extractor_ru.analysis.scorecard import TALK_RATIO_CRITERION
from insight_extractor_ru.core.models import (
    ResolvedCriterionResult,
    ResolvedEvidence,
    Role,
    Transcript,
)
from insight_extractor_ru.validation.evidence import NO_VALID_QUOTES_FLAG

IDEAL_MANAGER_SHARE = 0.4


def compute_talk_ratio(transcript: Transcript) -> ResolvedCriterionResult:
    """Score the manager/client talk ratio from turn text lengths.

    There is no LLM "evidence" for a whole-transcript statistic like this —
    ``resolved_evidence`` lists every turn that fed into the computed
    ratio, and ``confidence`` is always ``1.0`` (a plain calculation, not a
    judgment call), so it is never routed to manual review unless the
    transcript has no turns at all.
    """
    manager_len = sum(
        len(t.text) for t in transcript.turns if t.role == Role.MANAGER
    )
    client_len = sum(len(t.text) for t in transcript.turns if t.role == Role.CLIENT)
    total = manager_len + client_len

    if total == 0:
        return ResolvedCriterionResult(
            id=TALK_RATIO_CRITERION.id,
            score=0.0,
            reason="No turns to compute a talk ratio from.",
            confidence=0.0,
            flags=[NO_VALID_QUOTES_FLAG],
        )

    manager_share = manager_len / total
    diff = abs(manager_share - IDEAL_MANAGER_SHARE)
    if diff <= 0.05:
        score = 3.0
    elif diff <= 0.15:
        score = 2.0
    elif diff <= 0.25:
        score = 1.0
    else:
        score = 0.0

    return ResolvedCriterionResult(
        id=TALK_RATIO_CRITERION.id,
        score=score,
        reason=(
            f"Manager talk share {manager_share:.0%} vs ideal "
            f"{IDEAL_MANAGER_SHARE:.0%}."
        ),
        resolved_evidence=[
            ResolvedEvidence(turn_index=t.turn_index, text=t.text)
            for t in transcript.turns
        ],
        confidence=1.0,
    )
