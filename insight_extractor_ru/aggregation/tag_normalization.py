"""Canonical tag table with embedding-based cosine-similarity deduplication.

See ``openspec/changes/add-transcript-analysis-pipeline/specs/result-aggregation``,
"Tag normalization deduplicates near-duplicate objections", and design.md
Decision 9. Kept as a small, swappable interface (an injectable ``embed``
callable) so a persistent vector index can replace the in-memory table
later without changing callers — not needed until a cross-call archive
exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from insight_extractor_ru.utils.embeddings import char_ngram_embedding, cosine_similarity

DEFAULT_SIMILARITY_THRESHOLD = 0.5

EmbedFn = Callable[[str], np.ndarray]


@dataclass
class CanonicalTagTable:
    """Deduplicates tag strings against a growing set of canonical tags."""

    embed: EmbedFn = char_ngram_embedding
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    _canonical: list[tuple[str, np.ndarray]] = field(default_factory=list)

    def normalize(self, tags: Sequence[str]) -> list[str]:
        """Map each tag to its canonical form, deduplicating the result.

        The first surface form seen for a given semantic cluster becomes
        that cluster's canonical tag; later near-duplicates map to it
        instead of appearing as separate entries.
        """
        result: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            canonical = self._match_or_add(tag)
            if canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
        return result

    def _match_or_add(self, tag: str) -> str:
        vector = self.embed(tag)
        for canonical_tag, canonical_vector in self._canonical:
            if cosine_similarity(vector, canonical_vector) >= self.similarity_threshold:
                return canonical_tag
        self._canonical.append((tag, vector))
        return tag
