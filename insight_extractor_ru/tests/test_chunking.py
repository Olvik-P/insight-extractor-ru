from __future__ import annotations

import pytest

from insight_extractor_ru.core.models import Role, Turn
from insight_extractor_ru.preprocessing.chunking import chunk_by_turns, render_chunk


def _turns(n: int) -> list[Turn]:
    return [
        Turn(
            turn_index=i,
            role=Role.MANAGER if i % 2 == 0 else Role.CLIENT,
            text=f"text-{i}",
        )
        for i in range(1, n + 1)
    ]


class TestChunkByTurns:
    def test_transcript_longer_than_one_chunk_covers_all_turns_with_overlap(
        self,
    ) -> None:
        turns = _turns(10)
        chunks = chunk_by_turns(turns, size=4, overlap=2)

        assert len(chunks) == 4
        assert [t.turn_index for t in chunks[0].turns] == [1, 2, 3, 4]
        assert [t.turn_index for t in chunks[1].turns] == [3, 4, 5, 6]
        assert [t.turn_index for t in chunks[2].turns] == [5, 6, 7, 8]
        assert [t.turn_index for t in chunks[3].turns] == [7, 8, 9, 10]

        covered = set()
        for chunk in chunks:
            covered.update(t.turn_index for t in chunk.turns)
        assert covered == {t.turn_index for t in turns}

    def test_transcript_shorter_than_one_chunk_yields_single_chunk(self) -> None:
        turns = _turns(3)
        chunks = chunk_by_turns(turns, size=4, overlap=2)

        assert len(chunks) == 1
        assert [t.turn_index for t in chunks[0].turns] == [1, 2, 3]

    def test_empty_transcript_yields_no_chunks(self) -> None:
        assert chunk_by_turns([], size=4, overlap=2) == []

    def test_global_turn_index_preserved_in_later_chunk(self) -> None:
        turns = _turns(10)
        chunks = chunk_by_turns(turns, size=4, overlap=2)

        second_chunk_turn_indices = [t.turn_index for t in chunks[1].turns]
        assert second_chunk_turn_indices == [3, 4, 5, 6]

    def test_pii_tokens_pass_through_unchanged(self) -> None:
        turns = [
            Turn(turn_index=1, role=Role.MANAGER, text="Работаете с <ORG_A1B2C3D4>?"),
            Turn(turn_index=2, role=Role.CLIENT, text="Да, давно."),
        ]
        chunks = chunk_by_turns(turns, size=4, overlap=2)

        assert "<ORG_A1B2C3D4>" in chunks[0].turns[0].text

    def test_rejects_overlap_not_smaller_than_size(self) -> None:
        with pytest.raises(ValueError):
            chunk_by_turns(_turns(5), size=4, overlap=4)


class TestRenderChunk:
    def test_renders_turn_labeled_lines(self) -> None:
        chunk = chunk_by_turns(_turns(2), size=4, overlap=2)[0]

        rendered = render_chunk(chunk)

        assert rendered == "[1] КЛИЕНТ: text-1\n[2] МЕНЕДЖЕР: text-2"
