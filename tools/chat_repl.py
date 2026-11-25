"""Настройка характера без Discord: текстовый чат с персонажем.

    python tools/chat_repl.py            # персонаж по умолчанию
    python tools/chat_repl.py barman     # конкретный
    python tools/chat_repl.py barman --voice   # ещё и озвучивать в файл

Команды: /reset — очистить память, /exit — выйти.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.llm import LLM


async def run(persona_key: str | None, voice: bool) -> None:
    cfg = load_config()
    key = persona_key or cfg.default_persona
    if key is None or key not in cfg.personas:
        print("Доступные персонажи:", ", ".join(cfg.personas))
        return

    persona = cfg.personas[key]
    llm = LLM(cfg.llm)
    speaker = None
    if voice:
        from app.tts import Speaker

        speaker = Speaker(cfg.tts)

    print(f"\n=== {persona.title} ===")
    print(f"Модель: {cfg.llm.model}")
    print("Пиши реплику игрока. /reset — забыть диалог, /exit — выход.\n")

    while True:
        try:
            text = input("Игрок> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not text:
            continue
        if text in {"/exit", "/quit", "exit"}:
            return
        if text == "/reset":
            llm.reset(f"repl:{key}")
            print("(память очищена)\n")
            continue

        answer = await llm.reply(persona, cfg.world, "Игрок", text, f"repl:{key}")
        if not answer:
            print("(промолчал)\n")
            continue

        print(f"{persona.title}> {answer}\n")

        if speaker is not None:
            logs = ROOT / "logs"
            logs.mkdir(exist_ok=True)
            if speaker.engine in {"piper", "silero"}:
                import wave
                import numpy as np

                pcm = await speaker.synthesize_pcm(answer, persona.voice, persona.rate, persona.pitch)
                out = logs / f"tts-{datetime.now():%H%M%S}.wav"
                with wave.open(str(out), "wb") as wf:
                    wf.setnchannels(2)
                    wf.setsampwidth(2)
                    wf.setframerate(48_000)
                    wf.writeframes(np.asarray(np.frombuffer(pcm, dtype=np.int16)).tobytes())
            else:
                mp3 = await speaker.synthesize(answer, persona.voice, persona.rate, persona.pitch)
                out = logs / f"tts-{datetime.now():%H%M%S}.mp3"
                out.write_bytes(mp3)
            print(f"(озвучка: {out})\n")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("persona", nargs="?", default=None)
    parser.add_argument("--voice", action="store_true", help="синтезировать ответ в mp3")
    args = parser.parse_args()
    asyncio.run(run(args.persona, args.voice))


if __name__ == "__main__":
    main()
