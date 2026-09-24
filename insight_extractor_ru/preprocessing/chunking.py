"""Turn-based chunking with overlap.

See ``openspec/changes/add-transcript-analysis-pipeline/specs/transcript-chunking``
for the behavior contract this implements.
"""

from __future__ import annotations

from insight_extractor_ru.core.models import Chunk, Role, Turn

_ROLE_LABELS: dict[Role, str] = {
    Role.MANAGER: "МЕНЕДЖЕР",
    Role.CLIENT: "КЛИЕНТ",
}


def chunk_by_turns(
    turns: list[Turn], *, size: int = 4, overlap: int = 2
) -> list[Chunk]:
    """Split turns into overlapping chunks, preserving global turn_index.

    Uses a sliding window with step ``size - overlap``. The last chunk is
    grown to reach the end of ``turns`` exactly once, rather than emitting
    a redundant, smaller trailing chunk.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not (0 <= overlap < size):
        raise ValueError("overlap must be non-negative and smaller than size")
    if not turns:
        return []

    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    total = len(turns)
    while True:
        end = min(start + size, total)
        chunks.append(Chunk(chunk_index=len(chunks), turns=turns[start:end]))
        if end >= total:
            break
        start += step
    return chunks


def render_chunk(chunk: Chunk) -> str:
    """Render a chunk as turn-labeled plain text: ``[turn_index] ROLE: text``."""
    lines = [
        f"[{turn.turn_index}] {_ROLE_LABELS[turn.role]}: {turn.text}"
        for turn in chunk.turns
    ]
    return "\n".join(lines)
