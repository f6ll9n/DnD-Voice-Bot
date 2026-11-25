"""Ввод токена Discord в окне + мгновенная проверка.

    python tools/set_token.py          — спросить токен, записать, проверить
    python tools/set_token.py --check  — только проверить текущий
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

API = "https://discord.com/api/v10/users/@me"


def read_token() -> str:
    if not ENV_FILE.exists():
        return ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("DISCORD_TOKEN="):
            value = line.split("=", 1)[1].strip()
            if value and "вставь" not in value.lower():
                return value
    return ""


def write_token(token: str) -> None:
    text = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""
    if not text:
        example = ROOT / ".env.example"
        text = example.read_text(encoding="utf-8") if example.exists() else ""

    lines, replaced = [], False
    for line in text.splitlines():
        if line.startswith("DISCORD_TOKEN="):
            lines.append(f"DISCORD_TOKEN={token}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.append(f"DISCORD_TOKEN={token}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(token: str) -> str:
    """Возвращает имя бота или бросает исключение с понятным текстом."""
    req = urllib.request.Request(
        API,
        headers={"Authorization": f"Bot {token}", "User-Agent": "DnDVoiceBot/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError(
                "Discord говорит 401 Unauthorized — токен неверный или уже сброшен."
            ) from exc
        raise RuntimeError(f"Discord ответил {exc.code}") from exc
    except Exception as exc:
        raise RuntimeError(f"Нет связи с Discord: {type(exc).__name__}") from exc

    name = data.get("username", "?")
    disc = data.get("discriminator", "0")
    return f"{name}#{disc}" if disc and disc != "0" else str(name)


def ask(prompt: str) -> str:
    """Окно ввода. Если tkinter недоступен — падаем в консоль."""
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        value = simpledialog.askstring("DnD Voice Bot", prompt, show="•") or ""
        root.destroy()
        return value.strip()
    except Exception:
        try:
            return input(prompt + " ").strip()
        except EOFError:
            return ""


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    token = read_token()

    if "--check" in sys.argv:
        if not token:
            print("Токен не задан.")
            return 1
        try:
            print(f"Токен рабочий. Бот: {validate(token)}")
            return 0
        except Exception as exc:
            print(f"Проблема: {exc}")
            return 1

    if token:
        try:
            print(f"Токен уже есть и рабочий. Бот: {validate(token)}")
            return 0
        except Exception:
            print("Сохранённый токен не работает — введи новый.")

    print("Открываю окно ввода...")
    for attempt in range(3):
        value = ask("Вставь токен бота Discord (Bot -> Reset Token):")
        if not value:
            print("Токен не введён.")
            return 1

        value = value.strip().removeprefix("Bot ").strip()
        try:
            name = validate(value)
        except Exception as exc:
            print(f"  {exc}")
            print(f"  Попытка {attempt + 1} из 3.")
            continue

        write_token(value)
        print(f"Готово. Токен записан в .env, бот опознан как: {name}")
        print("Теперь запускай run.bat")
        return 0

    print("Три неудачные попытки. Проверь, что копируешь Bot Token, а не Client Secret.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
