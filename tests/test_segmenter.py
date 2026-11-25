"""Проверка нарезки фраз — без Discord, без видеокарты, без webrtcvad.

Запуск:  python tests\\test_segmenter.py

webrtcvad подменяется простым детектором по громкости, поэтому тест проверяет
именно логику сегментации (начало фразы, хвост тишины, отсев коротких
выкриков, принудительный обрез длинной речи), а не качество детектора.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:  # настоящий webrtcvad, если установлен
    import webrtcvad  # noqa: F401
except ModuleNotFoundError:  # подставляем заглушку
    _stub = types.ModuleType("webrtcvad")

    class _EnergyVad:
        def __init__(self, level: int = 2) -> None:
            self.level = level

        def is_speech(self, frame: bytes, rate: int) -> bool:
            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            return float(np.sqrt((samples**2).mean())) > 1000.0

    _stub.Vad = _EnergyVad  # type: ignore[attr-defined]
    sys.modules["webrtcvad"] = _stub

from app.vad import SegmenterCfg, UtteranceSegmenter  # noqa: E402

SR = 48_000
CHUNK = SR // 50  # 20 мс — ровно пакет Discord


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.int16)


def speech(seconds: float, amplitude: float = 0.5) -> np.ndarray:
    n = int(SR * seconds)
    t = np.arange(n, dtype=np.float32) / SR
    return (np.sin(2 * np.pi * 220 * t) * amplitude * 32767).astype(np.int16)


def feed(segmenter: UtteranceSegmenter, uid: int, stream: np.ndarray) -> None:
    for i in range(0, stream.size, CHUNK):
        segmenter.feed(uid, stream[i : i + CHUNK])


def case_basic() -> None:
    got: list[np.ndarray] = []
    seg = UtteranceSegmenter(SegmenterCfg(), on_utterance=lambda _uid, a: got.append(a))

    stream = np.concatenate(
        [
            silence(0.5),
            speech(2.0),
            silence(1.2),   # хвостовая тишина -> конец фразы
            speech(0.05),   # слишком коротко, должно отсеяться
            silence(1.2),
        ]
    )
    feed(seg, 1, stream)

    assert len(got) == 1, f"ожидалась 1 фраза, получено {len(got)}"
    dur = got[0].size / SR
    assert 2.0 <= dur <= 2.5, f"длительность {dur:.2f} с вне ожидаемого диапазона"
    print(f"  базовый случай: 1 фраза, {dur:.2f} с")


def case_long() -> None:
    got: list[np.ndarray] = []
    seg = UtteranceSegmenter(SegmenterCfg(max_utterance_s=3.0), on_utterance=lambda _u, a: got.append(a))
    feed(seg, 2, speech(6.0))
    assert got, "длинная речь не обрезана вообще"
    assert got[0].size / SR <= 3.2, f"обрез не сработал: {got[0].size / SR:.2f} с"
    print(f"  длинная речь: обрезана до {got[0].size / SR:.2f} с")


def case_speakers() -> None:
    """Двое говорят одновременно — у каждого свой буфер."""
    got: dict[int, int] = {}
    seg = UtteranceSegmenter(SegmenterCfg(), on_utterance=lambda uid, _a: got.__setitem__(uid, got.get(uid, 0) + 1))
    feed(seg, 10, np.concatenate([silence(0.3), speech(1.0), silence(1.2)]))
    feed(seg, 20, np.concatenate([silence(0.3), speech(1.0), silence(1.2)]))
    assert got == {10: 1, 20: 1}, f"речь спикеров перемешалась: {got}"
    print("  два говорящих: разведены корректно")


def case_noise_only() -> None:
    got: list[np.ndarray] = []
    seg = UtteranceSegmenter(SegmenterCfg(), on_utterance=lambda _u, a: got.append(a))
    feed(seg, 3, silence(3.0))
    assert not got, "тишина распознана как речь"
    print("  тишина: фраз не найдено")


class SegmenterTests(unittest.TestCase):
    def test_basic(self) -> None:
        case_basic()

    def test_long(self) -> None:
        case_long()

    def test_speakers(self) -> None:
        case_speakers()

    def test_noise_only(self) -> None:
        case_noise_only()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    print("\n=== Сегментация речи ===")
    case_basic()
    case_long()
    case_speakers()
    case_noise_only()
    print("\nВсё прошло.\n")
