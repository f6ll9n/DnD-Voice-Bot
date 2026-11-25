"""Проверка подготовки текста к озвучке. Запуск: python tests\\test_text.py"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.text import clean_for_speech, split_sentences  # noqa: E402


def case_clean() -> None:
    cases = [
        ('«Чего надо?»', "Чего надо?"),
        ('*чешет затылок* Эль есть?', "Эль есть?"),
        ('Он говорит «нет» (тихо)', "Он говорит «нет»"),
        ("**Плати** вперёд.", "Плати вперёд."),
        ("  двойные   пробелы  ", "двойные пробелы"),
        ("", ""),
    ]
    for src, expected in cases:
        got = clean_for_speech(src)
        assert got == expected, f"{src!r}: ожидалось {expected!r}, получено {got!r}"
    print(f"  очистка: {len(cases)} случаев ок")


def case_split() -> None:
    parts = split_sentences("Плати вперёд. Эль есть? Не видал я твоих цацек.")
    assert parts == ["Плати вперёд.", "Эль есть?", "Не видал я твоих цацек."], parts

    long_text = "слово " * 100
    pieces = split_sentences(long_text.strip(), max_len=120)
    assert all(len(p) <= 120 for p in pieces), [len(p) for p in pieces]
    assert " ".join(pieces).split() == long_text.split(), "текст потерялся при нарезке"

    assert split_sentences("") == []
    assert split_sentences("Одно предложение") == ["Одно предложение"]
    print(f"  нарезка: ок, длинный текст -> {len(pieces)} кусков")


class TextTests(unittest.TestCase):
    def test_clean(self) -> None:
        case_clean()

    def test_split(self) -> None:
        case_split()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    print("\n=== Подготовка текста ===")
    case_clean()
    case_split()
    print("\nВсё прошло.\n")
