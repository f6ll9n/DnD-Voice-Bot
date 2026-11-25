"""Самопроверка перед первым запуском. Запускается установщиком и вручную."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OK, WARN, BAD = "[ok]  ", "[warn]", "[fail]"
failures: list[str] = []
warnings: list[str] = []


def line(status: str, label: str, detail: str = "") -> None:
    print(f"{status} {label}" + (f" — {detail}" if detail else ""))


def check(label: str, fn, critical: bool = True) -> bool:
    try:
        detail = fn()
    except Exception as exc:
        line(BAD, label, f"{type(exc).__name__}: {exc}")
        (failures if critical else warnings).append(label)
        return False
    line(OK, label, str(detail) if detail else "")
    return True


def warn(label: str, detail: str) -> None:
    line(WARN, label, detail)
    warnings.append(label)


# --------------------------------------------------------------------- проверки


def check_python() -> str:
    major, minor = sys.version_info[:2]
    if not ((3, 10) <= (major, minor) <= (3, 12)):
        raise RuntimeError(f"Python {major}.{minor} — нужен 3.10–3.12")
    return f"Python {major}.{minor}"


def check_imports() -> str:
    missing = []
    for mod in (
        "discord",
        "discord.ext.voice_recv",
        "davey",
        "faster_whisper",
        "webrtcvad",
        "piper",
        "edge_tts",
        "miniaudio",
        "soxr",
        "openai",
        "yaml",
    ):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        raise RuntimeError("не установлены: " + ", ".join(missing) + " (запусти install.bat)")
    return "все зависимости на месте"


def check_config():
    from app.config import load_config

    cfg = load_config()
    if not cfg.personas:
        raise RuntimeError("в config.yaml нет ни одного personas")
    if cfg.default_persona and cfg.default_persona not in cfg.personas:
        raise RuntimeError(f"default_persona '{cfg.default_persona}' не найден")
    empty = [k for k, p in cfg.personas.items() if not p.prompt.strip()]
    if empty:
        raise RuntimeError("пустые файлы промптов: " + ", ".join(empty))
    return f"{len(cfg.personas)} персонаж(а), мир: {'есть' if cfg.world.strip() else 'нет'}"


def check_token() -> str:
    token = (os.getenv("DISCORD_TOKEN") or "").strip()
    if not token or token.lower().startswith("вставь"):
        raise RuntimeError("DISCORD_TOKEN не задан в .env")

    req = urllib.request.Request(
        "https://discord.com/api/v10/users/@me",
        headers={"Authorization": f"Bot {token}", "User-Agent": "DnDVoiceBot/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError("токен неверный (401). Проверь, что скопировал Bot Token, а не Client Secret") from exc
        raise RuntimeError(f"Discord ответил {exc.code}") from exc
    return f"бот {data.get('username')}#{data.get('discriminator', '0')}"


def check_whisper():
    import numpy as np

    from app.config import load_config
    from app.stt import Transcriber

    stt = Transcriber(load_config().stt)
    stt.load()
    stt.transcribe(np.zeros(48000, dtype=np.int16))
    return stt.describe()


def check_llm():
    from app.config import load_config
    from app.llm import LLM

    cfg = load_config()
    llm = LLM(cfg.llm)
    models = asyncio.run(llm.ping())
    if not models:
        raise RuntimeError(
            "сервер ответил, но моделей нет. Для Ollama: ollama pull qwen2.5:3b"
        )
    has_target = cfg.llm.model in models or (
        ":" not in cfg.llm.model and f"{cfg.llm.model}:latest" in models
    )
    if not has_target:
        raise RuntimeError(
            f"модели '{cfg.llm.model}' нет на сервере. Доступны: {', '.join(models[:6])}. "
            "Для Ollama: ollama pull <модель>"
        )
    return f"{len(models)} модель(ей), '{cfg.llm.model}' на месте"


def check_stt_remote():
    from app.config import load_config
    from app.stt import RemoteTranscriber

    transcriber = RemoteTranscriber(load_config().stt)
    transcriber.load()
    return transcriber.describe()


def check_tts():
    from app.config import load_config
    from app.tts import Speaker

    cfg = load_config()
    speaker = Speaker(cfg.tts)
    if speaker.engine == "silero":
        voices = [p.voice for p in cfg.personas.values() if p.voice] or ["eugene"]
        total_kb = 0
        for voice in dict.fromkeys(voices):
            pcm = speaker._silero_pcm("Проверка связи.", voice, strict=True)
            if not pcm:
                raise RuntimeError(f"пустой ответ Silero для {voice}")
            total_kb += len(pcm) // 1024
        return f"silero (офлайн 48 кГц), голосов: {len(dict.fromkeys(voices))}, pcm {total_kb} КБ"

    if speaker.engine == "piper":
        voices = [p.voice for p in cfg.personas.values() if p.voice] or [cfg.tts.piper_model]
        total_kb = 0
        for voice in dict.fromkeys(voices):
            pcm = speaker._piper_pcm("Проверка связи.", voice, strict=True)
            if not pcm:
                raise RuntimeError(f"пустой ответ Piper для {voice}")
            total_kb += len(pcm) // 1024
        return f"piper (офлайн), голосов: {len(dict.fromkeys(voices))}, pcm {total_kb} КБ"

    mp3 = asyncio.run(speaker.synthesize("Проверка связи.", "ru-RU-DmitryNeural"))
    if not mp3:
        raise RuntimeError("пустой ответ от синтеза")
    pcm = Speaker.to_pcm48_stereo(mp3)
    return f"edge-tts, mp3 {len(mp3) // 1024} КБ -> pcm {len(pcm) // 1024} КБ"


def check_folders() -> str:
    for name in ("logs", "world", "personas"):
        (ROOT / name).mkdir(exist_ok=True)
    return "logs/, world/, personas/"


# ------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка окружения DnD Voice Bot")
    parser.add_argument("--quick", action="store_true", help="не загружать Whisper (быстро)")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    print("\n=== Проверка окружения ===\n")

    check("Python", check_python)
    check("Зависимости", check_imports)
    check("Конфиг и промпты", check_config)
    check_folders()
    print()

    check("Токен Discord", check_token, critical=False)

    from app.config import load_config as _load

    cfg = _load()
    if cfg.input.mode == "text":
        line(OK, "Whisper", "не нужен (вход текстовый)")
    elif (cfg.stt.engine or "local").lower() == "remote":
        check("Whisper по сети", check_stt_remote)
    else:
        try:
            from app.stt import cuda_device_count

            devices = cuda_device_count()
        except Exception as exc:
            devices = 0
            line(WARN, "CUDA", f"недоступна ({type(exc).__name__}) — Whisper пойдёт на CPU")
        else:
            if devices:
                line(OK, "CUDA", f"устройств: {devices}")
            else:
                line(WARN, "CUDA", "нет — Whisper пойдёт на CPU, это работает, просто медленнее")

        print()
        if args.quick:
            line(WARN, "Whisper", "пропущено (--quick)")
        else:
            check("Whisper", check_whisper)

    print()
    check("LLM-сервер (Ollama / LM Studio)", check_llm)
    check("Синтез речи", check_tts)

    print("\n=== Итог ===")
    if failures:
        print(f"{BAD} Не готово. Проблемы: {', '.join(failures)}")
        return 1
    if warnings:
        print(f"{WARN} Работать будет, но обрати внимание: {', '.join(warnings)}")
    else:
        print(f"{OK} Всё готово. Запускай run.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
