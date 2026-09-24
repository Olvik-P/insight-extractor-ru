from __future__ import annotations

from insight_extractor_ru.aggregation.merge import merge_criteria, merge_summaries
from insight_extractor_ru.aggregation.scoring import compute_overall_score
from insight_extractor_ru.aggregation.tag_normalization import CanonicalTagTable
from insight_extractor_ru.aggregation.talk_ratio import compute_talk_ratio
from insight_extractor_ru.core.models import (
    ActionItem,
    Chunk,
    CriterionResult,
    EvidenceRef,
    Objection,
    RawChunkResult,
    Role,
    SummaryFragment,
    Transcript,
    Turn,
)

_CHUNK_A = Chunk(
    chunk_index=0,
    turns=[
        Turn(turn_index=1, role=Role.MANAGER, text="Какая у вас боль?"),
        Turn(turn_index=2, role=Role.CLIENT, text="Дорого выходит."),
    ],
)
_CHUNK_B = Chunk(
    chunk_index=1,
    turns=[
        Turn(turn_index=3, role=Role.MANAGER, text="Что скажете про сроки?"),
        Turn(turn_index=4, role=Role.CLIENT, text="Не сейчас, подумаем."),
    ],
)
_CHUNKS_BY_INDEX = {0: _CHUNK_A, 1: _CHUNK_B}


class TestMergeCriteria:
    def test_picks_highest_confidence_among_resolvable_chunks(self) -> None:
        chunk_results = [
            RawChunkResult(
                chunk_index=0,
                criteria=[
                    CriterionResult(
                        id="discovery_depth",
                        score=1,
                        reason="weak",
                        evidence=[EvidenceRef(turn_index=2)],
                        confidence=0.5,
                    )
                ],
                summary_fragment=SummaryFragment(),
            ),
            RawChunkResult(
                chunk_index=1,
                criteria=[
                    CriterionResult(
                        id="discovery_depth",
                        score=3,
                        reason="strong",
                        evidence=[EvidenceRef(turn_index=4)],
                        confidence=0.9,
                    )
                ],
                summary_fragment=SummaryFragment(),
            ),
        ]

        merged = merge_criteria(chunk_results, _CHUNKS_BY_INDEX)

        assert merged["discovery_depth"].score == 3
        assert merged["discovery_depth"].confidence == 0.9

    def test_prefers_more_evidence_over_higher_confidence(self) -> None:
        """Regression test for a real DeepSeek failure mode (live testing):
        an objection and its handling both land in one chunk, but the
        objection alone also lands in an earlier overlapping chunk that
        never saw the response. That earlier, under-informed chunk can
        report a similar-or-higher confidence for its (wrong) verdict than
        the chunk that actually saw the full exchange. Evidence breadth,
        not confidence, should decide.
        """
        narrow_chunk = Chunk(
            chunk_index=0,
            turns=[
                Turn(turn_index=1, role=Role.MANAGER, text="Как вам наше решение?"),
                Turn(turn_index=2, role=Role.CLIENT, text="Дорого выходит."),
            ],
        )
        broad_chunk = Chunk(
            chunk_index=1,
            turns=[
                Turn(turn_index=1, role=Role.MANAGER, text="Как вам наше решение?"),
                Turn(turn_index=2, role=Role.CLIENT, text="Дорого выходит."),
                Turn(
                    turn_index=3,
                    role=Role.MANAGER,
                    text="Зато у нас есть готовая интеграция с вашей CRM.",
                ),
                Turn(turn_index=4, role=Role.CLIENT, text="Хорошо, это меняет дело."),
            ],
        )
        chunks_by_index = {0: narrow_chunk, 1: broad_chunk}

        narrow_chunk_result = RawChunkResult(
            chunk_index=0,
            criteria=[
                CriterionResult(
                    id="objection_handling",
                    score=1,
                    reason="unhandled (only saw the objection, not the reply)",
                    evidence=[EvidenceRef(turn_index=2)],
                    confidence=0.9,  # higher confidence, but less evidence
                )
            ],
            summary_fragment=SummaryFragment(),
        )
        broad_chunk_result = RawChunkResult(
            chunk_index=1,
            criteria=[
                CriterionResult(
                    id="objection_handling",
                    score=3,
                    reason="handled (saw both the objection and the reply)",
                    evidence=[EvidenceRef(turn_index=2), EvidenceRef(turn_index=3)],
                    confidence=0.85,  # lower confidence, but more evidence
                )
            ],
            summary_fragment=SummaryFragment(),
        )

        merged = merge_criteria(
            [narrow_chunk_result, broad_chunk_result], chunks_by_index
        )

        assert merged["objection_handling"].score == 3

    def test_no_resolvable_evidence_anywhere_keeps_zero_confidence(self) -> None:
        chunk_results = [
            RawChunkResult(
                chunk_index=0,
                criteria=[
                    CriterionResult(
                        id="discovery_depth",
                        score=2,
                        reason="hallucinated ref",
                        evidence=[EvidenceRef(turn_index=999)],
                        confidence=0.8,
                    )
                ],
                summary_fragment=SummaryFragment(),
            )
        ]

        merged = merge_criteria(chunk_results, _CHUNKS_BY_INDEX)

        assert merged["discovery_depth"].confidence == 0.0
        assert "no_valid_quotes" in merged["discovery_depth"].flags


class TestMergeSummaries:
    def test_unions_objections_and_action_items_across_chunks(self) -> None:
        chunk_results = [
            RawChunkResult(
                chunk_index=0,
                criteria=[],
                summary_fragment=SummaryFragment(
                    client_pain="Дорого",
                    objections=[
                        Objection(text="дорого", evidence=EvidenceRef(turn_index=2))
                    ],
                    action_items=[
                        ActionItem(
                            owner=Role.MANAGER,
                            task="прислать КП",
                            evidence=EvidenceRef(turn_index=2),
                        )
                    ],
                ),
            ),
            RawChunkResult(
                chunk_index=1,
                criteria=[],
                summary_fragment=SummaryFragment(
                    objections=[
                        Objection(text="не сейчас", evidence=EvidenceRef(turn_index=4))
                    ],
                    action_items=[],
                ),
            ),
        ]

        summary = merge_summaries(chunk_results, _CHUNKS_BY_INDEX)

        assert summary.client_pain == "Дорого"
        assert set(summary.objections) == {"дорого", "не сейчас"}
        assert len(summary.action_items) == 1
        assert summary.action_items[0].resolved_evidence.text == "Дорого выходит."

    def test_out_of_range_action_item_evidence_is_dropped(self) -> None:
        chunk_results = [
            RawChunkResult(
                chunk_index=0,
                criteria=[],
                summary_fragment=SummaryFragment(
                    action_items=[
                        ActionItem(
                            owner=Role.MANAGER,
                            task="перезвонить",
                            evidence=EvidenceRef(turn_index=999),
                        )
                    ]
                ),
            )
        ]

        summary = merge_summaries(chunk_results, _CHUNKS_BY_INDEX)

        assert summary.action_items == []


class TestTagNormalization:
    def test_near_duplicate_tags_collapse_to_one_canonical_tag(self) -> None:
        table = CanonicalTagTable()

        result = table.normalize(["дорого", "слишком дорого", "не сейчас"])

        assert result == ["дорого", "не сейчас"]


class TestComputeOverallScore:
    def test_excludes_manual_review_criteria_from_calculation(self) -> None:
        from insight_extractor_ru.core.models import ResolvedCriterionResult

        criteria = {
            "discovery_depth": ResolvedCriterionResult(
                id="discovery_depth", score=3, reason="ok", confidence=0.9
            ),
            "objection_handling": ResolvedCriterionResult(
                id="objection_handling",
                score=0,
                reason="low confidence",
                confidence=0.3,  # below default threshold -> manual review
            ),
        }

        score_with_low_confidence = compute_overall_score(criteria)

        criteria_without_low_confidence = {
            k: v for k, v in criteria.items() if k != "objection_handling"
        }
        score_without_it = compute_overall_score(criteria_without_low_confidence)

        assert score_with_low_confidence == score_without_it


class TestComputeTalkRatio:
    def test_scores_three_when_close_to_ideal_forty_sixty_split(self) -> None:
        transcript = Transcript(
            call_id="c1",
            turns=[
                Turn(turn_index=1, role=Role.MANAGER, text="x" * 40),
                Turn(turn_index=2, role=Role.CLIENT, text="x" * 60),
            ],
        )

        result = compute_talk_ratio(transcript)

        assert result.score == 3.0
        assert result.confidence == 1.0

    def test_empty_transcript_yields_zero_confidence(self) -> None:
        transcript = Transcript(call_id="c1", turns=[])

        result = compute_talk_ratio(transcript)

        assert result.confidence == 0.0
        assert "no_valid_quotes" in result.flags
