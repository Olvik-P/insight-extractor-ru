from __future__ import annotations

from insight_extractor_ru.core.models import CriterionResult, EvidenceRef


class TestEvidenceRefBareIntCoercion:
    """Regression test for a real DeepSeek response shape seen in live testing:
    the model sometimes returns ``"evidence": [8, 9]`` (bare integers)
    instead of ``[{"turn_index": 8}, {"turn_index": 9}]``.
    """

    def test_accepts_bare_int(self) -> None:
        ref = EvidenceRef.model_validate(8)

        assert ref.turn_index == 8

    def test_still_accepts_the_documented_object_shape(self) -> None:
        ref = EvidenceRef.model_validate({"turn_index": 8})

        assert ref.turn_index == 8

    def test_criterion_result_evidence_list_accepts_mixed_bare_ints(self) -> None:
        criterion = CriterionResult.model_validate(
            {
                "id": "discovery_depth",
                "score": 2,
                "reason": "ok",
                "evidence": [8, {"turn_index": 9}],
                "confidence": 0.7,
            }
        )

        assert [e.turn_index for e in criterion.evidence] == [8, 9]
