"""Сравнение моделей Whisper на синтезированной русской фразе.

    python tools/stt_probe.py base small medium

Пишет результат в logs/stt_probe.md (консоль Windows ломает кириллицу).
"""

from __future__ import annotations

import asyncio
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import audio  # noqa: E402
from app.config import STTCfg, load_config  # noqa: E402

PHRASE = "Бармен, налей мне пива и расскажи про пропавшую девушку."


def synth_phrase() -> np.ndarray:
    """Синтезируем фразу локальным Piper — получаем эталонный PCM 48 кГц."""
    from app.tts import Speaker

    cfg = load_config()
    speaker = Speaker(cfg.tts)
    pcm = asyncio.run(speaker.synthesize_pcm(PHRASE, ""))
    return np.frombuffer(pcm, dtype=np.int16)


def main() -> None:
    models = sys.argv[1:] or ["base", "small"]
    samples48 = synth_phrase()

    cfg = load_config()
    rows = [f"# Эталон: {PHRASE}\n", f"Длительность: {samples48.size / 48000:.1f} с\n"]

    from faster_whisper import WhisperModel

    for name in models:
        stt_cfg = STTCfg(model=name, device="cpu", compute_type="int8", language="ru")
        t0 = time.perf_counter()
        model = WhisperModel(name, device="cpu", compute_type="int8")
        load_s = time.perf_counter() - t0

        wav16 = audio.to_float32(audio.resample(samples48, 48000, 16000))
        t1 = time.perf_counter()
        segments, _ = model.transcribe(
            wav16,
            language="ru",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt=cfg.stt.initial_prompt or None,
            temperature=0.0,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        dur = time.perf_counter() - t1

        rows.append(f"## {name}")
        rows.append(f"- загрузка: {load_s:.1f} с")
        rows.append(f"- распознавание: {dur:.2f} с")
        rows.append(f"- текст: «{text}»\n")

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    out = logs / "stt_probe.md"
    out.write_text("\n".join(rows), encoding="utf-8")
    print(f"готово: {out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    main()
