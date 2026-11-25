"""Прогон типичных реплик игроков через модель — без Discord.

    python tools/llm_probe.py            # текущий персонаж из config.yaml
    python tools/llm_probe.py dm         # другой персонаж

Результат пишется в logs/llm_probe.md (консоль на Windows ломает кириллицу).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.llm import LLM  # noqa: E402

INPUTS = [
    "Бармен, налей мне пива",
    "Сколько стоит эль?",
    "Слышал что-нибудь про пропавшую девушку?",
    "А пожрать у тебя есть?",
    "Где тут можно переночевать?",
    "Бармен, а ты не врёшь?",
]


async def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else (load_config().default_persona or "barman")
    cfg = load_config()
    persona = cfg.personas[key]
    llm = LLM(cfg.llm)

    rows: list[str] = [f"# Прогон: {persona.title}\n"]
    for text in INPUTS:
        try:
            answer = await llm.reply(persona, cfg.world, "Фрикция", text, "probe")
        except Exception as exc:  # noqa: BLE001
            answer = f"<ошибка: {type(exc).__name__}: {exc}>"
        rows.append(f"**Игрок:** {text}")
        rows.append(f"**{persona.title}:** {answer}\n")

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    out = logs / "llm_probe.md"
    out.write_text("\n".join(rows), encoding="utf-8")
    print(f"готово: {out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    asyncio.run(main())
