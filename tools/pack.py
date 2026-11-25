"""Собирает архив для передачи другу.

Исключает то, что нельзя переносить: .venv (в нём абсолютные пути),
логи с расшифровками сессий и кэш Python.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "DnD-Voice-Bot.zip"

SKIP_DIRS = {".venv", "logs", "__pycache__", ".git", ".idea", ".vscode", ".pytest_cache"}
SKIP_FILES = {".env"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".zip"}


def keep(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    if path.name in SKIP_FILES:
        return False
    return path.suffix not in SKIP_SUFFIXES


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    if ARCHIVE.exists():
        ARCHIVE.unlink()

    count = 0
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(ROOT.rglob("*")):
            rel = path.relative_to(ROOT)
            if not path.is_file() or not keep(rel):
                continue
            zf.write(path, rel)
            count += 1

    size = ARCHIVE.stat().st_size / 1024
    print(f"\nГотово: {ARCHIVE.name} — {count} файлов, {size:.0f} КБ")
    print("Отправляй другу этот архив. Там нет .env, логов и .venv — это правильно:")
    print("на новой машине он запускает install.bat, и окружение собирается заново.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
