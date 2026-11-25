"""Точка входа. Запускается через run.bat."""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request

from app.bot import DnDBot
from app.config import load_config
from app.cuda import register_nvidia_dlls

# GATEWAY_MESSAGE_CONTENT (18) включается только после верификации (100+ серверов).
# До этого Discord ставит GATEWAY_MESSAGE_CONTENT_LIMITED (19). Считаем оба.
MESSAGE_CONTENT_FLAGS = (1 << 18) | (1 << 19)


def setup_logging() -> None:
    from pathlib import Path

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "bot.log", mode="a", encoding="utf-8"),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    for noisy in ("discord", "faster_whisper", "httpx", "httpcore", "httpcore2", "httpx2", "openai", "edge_tts", "miniaudio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def message_content_enabled(token: str) -> bool:
    """Спрашиваем у Discord, включён ли MESSAGE CONTENT INTENT у приложения."""
    req = urllib.request.Request(
        "https://discord.com/api/v10/applications/@me",
        headers={
            "Authorization": f"Bot {token}",
            "User-Agent": "DnDVoiceBot/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        flags = int(json.loads(resp.read()).get("flags", 0))
    return bool(flags & MESSAGE_CONTENT_FLAGS)


def main() -> int:
    setup_logging()
    log = logging.getLogger("main")

    register_nvidia_dlls()

    cfg = load_config()
    token = (os.getenv("DISCORD_TOKEN") or "").strip()
    if not token or token.lower().startswith("вставь"):
        log.error("Нет DISCORD_TOKEN. Скопируй .env.example в .env и вставь токен бота.")
        return 1

    log.info("Модель LLM: %s @ %s", cfg.llm.model, cfg.llm.base_url)
    log.info("Whisper: %s (%s/%s)", cfg.stt.model, cfg.stt.device, cfg.stt.compute_type)

    try:
        has_content = message_content_enabled(token)
    except Exception as exc:
        log.warning("Не смог проверить интенты (%s) — пробую подключиться как есть", exc)
        has_content = True

    if not has_content:
        log.warning(
            "\n"
            "MESSAGE CONTENT INTENT выключен в Developer Portal.\n"
            "  Работаю в режиме только слеш-команд: /join, /npc, /say, /status.\n"
            "  Префиксные команды (!npc) и текстовый вход недоступны.\n"
            "  Включить: discord.com/developers/applications -> Bot -> MESSAGE CONTENT INTENT\n"
        )

    bot = DnDBot(cfg, message_content=has_content)
    try:
        bot.run(token, log_handler=None)
    except KeyboardInterrupt:
        log.info("Остановлено пользователем")
    except Exception as exc:
        if type(exc).__name__ == "LoginFailure":
            log.error("Неверный DISCORD_TOKEN. Запусти set_token.bat и вставь актуальный Bot Token.")
            return 2
        if type(exc).__name__ == "PrivilegedIntentsRequired":
            log.error(
                "\n"
                "Discord не пустил бота: не хватает привилегированного интента.\n"
                "  discord.com/developers/applications -> Bot -> Privileged Gateway Intents\n"
                "  Включи MESSAGE CONTENT INTENT и перезапусти run.bat\n"
            )
            return 2
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
