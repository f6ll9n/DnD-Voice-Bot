"""Текст, который пойдёт в синтез речи: чистка разметки и нарезка на фразы."""

from __future__ import annotations

import re

_ACTIONS = re.compile(r"\*[^*]{0,200}\*")
_BRACKETS = re.compile(r"[\[(][^\])]{0,120}[\])]")
_SPACES = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")

_QUOTE_PAIRS = {'"': '"', "«": "»", "“": "”", "„": "“", "'": "'", "‘": "’"}


def _strip_wrapping_quotes(text: str) -> str:
    """Снимает кавычки только если они охватывают всю реплику целиком.

    Важно: «нет» внутри фразы («Он говорит «нет»») трогать нельзя —
    иначе съедается закрывающая кавычка.
    """
    while len(text) >= 2:
        closer = _QUOTE_PAIRS.get(text[0])
        if closer is None or text[-1] != closer:
            break
        text = text[1:-1].strip()
    return text


def clean_for_speech(text: str) -> str:
    """Убирает то, что нельзя произнести: ремарки, звёздочки, кавычки-обёртки."""
    text = _ACTIONS.sub(" ", text)
    text = _BRACKETS.sub(" ", text)
    text = text.replace("**", "").replace("*", "").replace("`", "")
    text = _SPACES.sub(" ", text).strip()
    return _strip_wrapping_quotes(text)


def split_sentences(text: str, max_len: int = 240) -> list[str]:
    """Режет реплику на куски, чтобы бот начинал говорить, не дожидаясь всего синтеза."""
    text = text.strip()
    if not text:
        return []

    parts: list[str] = []
    for chunk in _SENTENCE_SPLIT.split(text):
        chunk = chunk.strip()
        while len(chunk) > max_len:
            cut = chunk.rfind(",", 0, max_len)
            if cut < max_len // 2:
                cut = chunk.rfind(" ", 0, max_len)
            if cut <= 0:
                cut = max_len
            parts.append(chunk[:cut].strip())
            chunk = chunk[cut:].strip(" ,")
        if chunk:
            parts.append(chunk)
    return parts
