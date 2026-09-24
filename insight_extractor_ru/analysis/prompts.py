"""Chunk-level prompt template.

Combines the turn-labeled chunk text with the scorecard and the
evidence-by-reference output contract (design.md, Decision 5): the model
must point at a ``turn_index`` for every claim, never copy quoted text.
"""

from __future__ import annotations

from insight_extractor_ru.analysis.scorecard import LLM_SCORED_CRITERIA
from insight_extractor_ru.core.models import Chunk
from insight_extractor_ru.preprocessing.chunking import render_chunk

_INSTRUCTIONS = """\
Ты анализируешь фрагмент транскрипта B2B-звонка между менеджером и клиентом.
Текст уже обезличен — плейсхолдеры вида <TYPE_HASH> являются частью текста,
копируй их дословно, если они попадают в выделенный тобой текст.

Для каждого критерия оцени фрагмент по шкале 0-3 и укажи в поле "evidence"
список номеров реплик (turn_index) из приведённого ниже фрагмента, которые
подтверждают твою оценку. НЕ копируй текст реплики — только её номер.
Если в этом фрагменте нет данных для критерия, верни score=0, confidence=0.0
и пустой список evidence.

Критерии:
{criteria_list}

Также заполни summary_fragment: боль клиента (client_pain, если явно
прозвучала в этом фрагменте, иначе null), список возражений (objections,
каждое — текст возражения своими словами + ссылка на turn_index) и
договорённости (action_items: owner "manager" или "client", задача, ссылка
на turn_index). Как и для критериев, ссылайся на turn_index, не копируй
текст реплики.

Фрагмент транскрипта:
{chunk_text}
"""


def build_chunk_prompt(chunk: Chunk) -> str:
    """Render the full prompt sent to the LLM for one chunk."""
    criteria_list = "\n".join(
        f"- {c.id}: {c.name}" for c in LLM_SCORED_CRITERIA
    )
    return _INSTRUCTIONS.format(
        criteria_list=criteria_list, chunk_text=render_chunk(chunk)
    )
