"""Подключаем CUDA-библиотеки, которые pip ставит внутрь пакета nvidia.

CTranslate2 (движок faster-whisper) на Windows не находит cuBLAS/cuDNN сам,
если не установлен CUDA Toolkit. Пакеты nvidia-cublas-cu12 / nvidia-cudnn-cu12
кладут DLL внутрь site-packages, но не прописывают их в PATH.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_SUBDIRS = ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime")


def _site_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        import site

        dirs.extend(Path(p) for p in site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site:
            dirs.append(Path(user_site))
    except Exception:  # pragma: no cover
        pass
    # на всякий случай — рядом с текущим интерпретатором
    dirs.append(Path(sys.prefix) / "Lib" / "site-packages")
    return dirs


def register_nvidia_dlls() -> list[str]:
    """Добавляет каталоги DLL в путь поиска. Возвращает список добавленных путей."""
    if os.name != "nt":
        return []

    added: list[str] = []
    seen: set[Path] = set()
    for base in _site_dirs():
        nvidia = base / "nvidia"
        if not nvidia.is_dir() or nvidia in seen:
            continue
        seen.add(nvidia)
        for pkg in _SUBDIRS:
            for candidate in (nvidia / pkg / "bin", nvidia / pkg / "lib"):
                if not candidate.is_dir():
                    continue
                try:
                    os.add_dll_directory(str(candidate))
                    added.append(str(candidate))
                except (OSError, AttributeError):
                    pass
    if added:
        log.debug("CUDA DLL: %s", ", ".join(added))
    return added
