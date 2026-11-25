"""Переключает config.yaml между ролями машин.

    python tools/configure.py --role bot   --brain 100.64.0.5
    python tools/configure.py --role brain
    python tools/configure.py --role all

  bot   — эта машина держит Discord-бота, Whisper и LLM живут на другой
  brain — эта машина считает Whisper и LLM (там видеокарта)
  all   — всё на одной машине

Правит файл построчно и сохраняет комментарии.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"


def set_in_section(lines: list[str], section: str, key: str, value: str) -> bool:
    """Меняет `key: ...` только внутри нужной секции, сохраняя отступ и комментарий."""
    inside = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not line.startswith((" ", "\t")):
            inside = stripped.rstrip() == f"{section}:"
            continue

        if not inside:
            continue

        name, _, _rest = stripped.partition(":")
        if name.strip() != key:
            continue

        indent = line[: len(line) - len(line.lstrip())]
        comment = ""
        if "#" in line:
            comment = "  " + line.split("#", 1)[1].rstrip()
        lines[i] = f"{indent}{key}: {value}{comment}\n"
        return True

    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Настройка роли машины")
    parser.add_argument("--role", choices=["bot", "brain", "all"], required=True)
    parser.add_argument("--brain", default="", help="IP машины с видеокартой")
    args = parser.parse_args()

    if not CONFIG.exists():
        print(f"Не найден {CONFIG}")
        return 1

    lines = CONFIG.read_text(encoding="utf-8").splitlines(keepends=True)

    if args.role == "bot":
        if not args.brain:
            print("Для роли bot нужен --brain <IP машины с видеокартой>")
            return 1
        brain = args.brain.strip().removeprefix("http://").removeprefix("https://").rstrip("/")
        engine, url, base = "remote", f'"http://{brain}:8765"', f'"http://{brain}:11434/v1"'
    else:
        engine, url, base = "local", '"http://127.0.0.1:8765"', '"http://127.0.0.1:11434/v1"'

    ok = [
        set_in_section(lines, "stt", "engine", engine),
        set_in_section(lines, "stt", "remote_url", url),
        set_in_section(lines, "llm", "base_url", base),
    ]

    CONFIG.write_text("".join(lines), encoding="utf-8")

    if not all(ok):
        print("Предупреждение: не все ключи нашлись в config.yaml — проверь файл глазами.")

    print(f"config.yaml настроен на роль: {args.role}")
    if args.role == "bot":
        print(f"  Whisper: http://{args.brain}:8765")
        print(f"  LLM:     http://{args.brain}:11434/v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
