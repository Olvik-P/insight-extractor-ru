"""The finalized scorecard (design.md, Decision 10).

``talk_ratio`` is deliberately excluded from ``LLM_SCORED_CRITERIA``: it is
computed deterministically from turn lengths per role over the whole
transcript (see ``aggregation.talk_ratio``), not judged by the LLM per
chunk — there is no "evidence" for it in the evidence-by-reference sense.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScorecardCriterion:
    id: str
    name: str
    weight: float
    max_score: float = 3.0


LLM_SCORED_CRITERIA: tuple[ScorecardCriterion, ...] = (
    ScorecardCriterion(id="discovery_depth", name="Discovery Depth", weight=0.22),
    ScorecardCriterion(
        id="objection_handling", name="Objection Handling", weight=0.20
    ),
    ScorecardCriterion(
        id="next_step_clarity", name="Next Step Clarity", weight=0.18
    ),
    ScorecardCriterion(
        id="budget_qualification", name="Budget Qualification", weight=0.14
    ),
    ScorecardCriterion(id="tone_empathy", name="Tone & Empathy", weight=0.14),
)

TALK_RATIO_CRITERION = ScorecardCriterion(
    id="talk_ratio", name="Talk Ratio", weight=0.12
)

ALL_CRITERIA: tuple[ScorecardCriterion, ...] = LLM_SCORED_CRITERIA + (
    TALK_RATIO_CRITERION,
)

LLM_SCORED_CRITERION_IDS: frozenset[str] = frozenset(
    c.id for c in LLM_SCORED_CRITERIA
)
