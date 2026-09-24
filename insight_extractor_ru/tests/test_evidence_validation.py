from __future__ import annotations

from insight_extractor_ru.core.models import Chunk, CriterionResult, EvidenceRef, Role, Turn
from insight_extractor_ru.validation.evidence import (
    NO_VALID_QUOTES_FLAG,
    needs_manual_review,
    resolve_evidence,
)

_CHUNK = Chunk(
    chunk_index=0,
    turns=[
        Turn(turn_index=3, role=Role.MANAGER, text="Работаете с <ORG_A1B2C3D4>?"),
        Turn(turn_index=4, role=Role.CLIENT, text="Да, уже 2 года."),
    ],
)


class TestResolveEvidence:
    def test_valid_evidence_reference_resolves_to_literal_text(self) -> None:
        criterion = CriterionResult(
            id="discovery_depth",
            score=2,
            reason="ok",
            evidence=[EvidenceRef(turn_index=4)],
            confidence=0.9,
        )

        resolved = resolve_evidence(criterion, _CHUNK)

        assert len(resolved.resolved_evidence) == 1
        assert resolved.resolved_evidence[0].text == "Да, уже 2 года."
        assert resolved.confidence == 0.9
        assert NO_VALID_QUOTES_FLAG not in resolved.flags

    def test_out_of_range_reference_is_dropped(self) -> None:
        criterion = CriterionResult(
            id="discovery_depth",
            score=2,
            reason="ok",
            evidence=[EvidenceRef(turn_index=99)],
            confidence=0.9,
        )

        resolved = resolve_evidence(criterion, _CHUNK)

        assert resolved.resolved_evidence == []

    def test_no_resolvable_evidence_forces_confidence_zero_and_flags(self) -> None:
        criterion = CriterionResult(
            id="discovery_depth",
            score=2,
            reason="ok",
            evidence=[EvidenceRef(turn_index=99)],
            confidence=0.9,
        )

        resolved = resolve_evidence(criterion, _CHUNK)

        assert resolved.confidence == 0.0
        assert NO_VALID_QUOTES_FLAG in resolved.flags

    def test_pii_token_in_resolved_text_is_preserved_verbatim(self) -> None:
        criterion = CriterionResult(
            id="discovery_depth",
            score=2,
            reason="ok",
            evidence=[EvidenceRef(turn_index=3)],
            confidence=0.9,
        )

        resolved = resolve_evidence(criterion, _CHUNK)

        assert resolved.resolved_evidence[0].text == "Работаете с <ORG_A1B2C3D4>?"


class TestNeedsManualReview:
    def test_below_threshold_needs_review(self) -> None:
        criterion = CriterionResult(
            id="discovery_depth",
            score=2,
            reason="ok",
            evidence=[EvidenceRef(turn_index=4)],
            confidence=0.5,
        )
        resolved = resolve_evidence(criterion, _CHUNK)

        assert needs_manual_review(resolved) is True

    def test_at_or_above_threshold_does_not_need_review(self) -> None:
        criterion = CriterionResult(
            id="discovery_depth",
            score=2,
            reason="ok",
            evidence=[EvidenceRef(turn_index=4)],
            confidence=0.6,
        )
        resolved = resolve_evidence(criterion, _CHUNK)

        assert needs_manual_review(resolved) is False

    def test_custom_threshold_is_respected(self) -> None:
        criterion = CriterionResult(
            id="discovery_depth",
            score=2,
            reason="ok",
            evidence=[EvidenceRef(turn_index=4)],
            confidence=0.75,
        )
        resolved = resolve_evidence(criterion, _CHUNK)

        assert needs_manual_review(resolved, threshold=0.8) is True
