"""Core contract models shared across the analysis pipeline.

These are the shapes that flow between preprocessing, analysis, validation
and aggregation. See ``openspec/changes/add-transcript-analysis-pipeline/``
for the requirements and design decisions behind this contract, in
particular the evidence-by-reference shape (turn_index references instead
of copied quote text).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Role(str, Enum):
    """Speaker role for a transcript turn."""

    MANAGER = "manager"
    CLIENT = "client"


class Turn(BaseModel):
    """One utterance in a transcript, already anonymized upstream."""

    turn_index: int
    role: Role
    text: str


class Transcript(BaseModel):
    """A full, already-anonymized call transcript."""

    call_id: str
    turns: list[Turn]

    @model_validator(mode="after")
    def _check_turn_indices(self) -> Transcript:
        indices = [turn.turn_index for turn in self.turns]
        if indices != sorted(indices):
            raise ValueError("turns must be ordered by ascending turn_index")
        if len(set(indices)) != len(indices):
            raise ValueError("turn_index values must be unique")
        return self


class Chunk(BaseModel):
    """A turn-based, overlapping slice of a transcript sent to the LLM."""

    chunk_index: int
    turns: list[Turn]

    def turn_by_index(self, turn_index: int) -> Turn | None:
        """Look up a turn in this chunk by its whole-transcript index."""
        for turn in self.turns:
            if turn.turn_index == turn_index:
                return turn
        return None


class EvidenceRef(BaseModel):
    """A reference to a turn, as returned by the LLM (not copied text).

    Live testing against DeepSeek showed the model sometimes returns a bare
    ``turn_index`` integer (e.g. ``"evidence": [8, 9]``) instead of the
    requested ``{"turn_index": N}`` object — a "simpler", more natural
    completion for "a list of turn numbers" that even strict JSON-Schema
    mode doesn't reliably prevent, and that persisted through the
    fallback's retry-with-error-feedback in observed cases. Coercing it
    here is cheaper and more robust than only relying on the model to
    self-correct, and costs nothing since the semantics are identical.
    """

    turn_index: int

    @model_validator(mode="before")
    @classmethod
    def _coerce_bare_int(cls, data: object) -> object:
        if isinstance(data, int):
            return {"turn_index": data}
        return data


class ResolvedEvidence(BaseModel):
    """An evidence reference resolved to its literal source text."""

    turn_index: int
    text: str


class CriterionResult(BaseModel):
    """One scorecard criterion result for a single chunk."""

    id: str
    score: float
    reason: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = 0.0
    flags: list[str] = Field(default_factory=list)


class Objection(BaseModel):
    """One client objection, as reported by the LLM for a chunk."""

    text: str
    evidence: EvidenceRef


class ActionItem(BaseModel):
    """One promised follow-up action, as reported by the LLM for a chunk."""

    owner: Role
    task: str
    evidence: EvidenceRef


class SummaryFragment(BaseModel):
    """The summary portion of a single chunk's LLM response."""

    client_pain: str | None = None
    objections: list[Objection] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)


class LLMChunkResponse(BaseModel):
    """The exact shape requested from the LLM for one chunk.

    Deliberately excludes ``chunk_index`` — that is known from context
    (which chunk we sent), not something to ask the model to report.
    """

    criteria: list[CriterionResult]
    summary_fragment: SummaryFragment


class RawChunkResult(BaseModel):
    """The full, schema-validated LLM response for one chunk, with context."""

    chunk_index: int
    criteria: list[CriterionResult]
    summary_fragment: SummaryFragment


class ResolvedCriterionResult(BaseModel):
    """A criterion result after evidence resolution and chunk merge."""

    id: str
    score: float
    reason: str
    resolved_evidence: list[ResolvedEvidence] = Field(default_factory=list)
    confidence: float = 0.0
    flags: list[str] = Field(default_factory=list)


class ResolvedActionItem(BaseModel):
    """An action item after evidence resolution."""

    owner: Role
    task: str
    resolved_evidence: ResolvedEvidence


class Summary(BaseModel):
    """The final, merged summary for a whole call."""

    client_pain: str | None = None
    objections: list[str] = Field(default_factory=list)
    action_items: list[ResolvedActionItem] = Field(default_factory=list)


class CallAnalysis(BaseModel):
    """The final, aggregated analysis contract for one call (project.md §9)."""

    call_id: str
    analysis_version: str = "1.0"
    overall_score: float
    criteria: list[ResolvedCriterionResult]
    summary: Summary
    flags: list[str] = Field(default_factory=list)
