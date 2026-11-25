"""Речь -> текст.

Два режима:
  local  — Whisper крутится на этой машине
  remote — Whisper живёт на другом ПК (там, где видеокарта),
           сюда приходит только текст. См. tools/stt_server.py
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

import numpy as np

from . import audio
from .config import STTCfg

log = logging.getLogger(__name__)

# Что брать, когда в конфиге стоит "auto"
# base для русского выдаёт абракадабру — минимально пригодный это small.
_AUTO_MODEL = {"cuda": "medium", "cpu": "small"}
_AUTO_COMPUTE = {"cuda": "int8_float16", "cpu": "int8"}


def cuda_device_count() -> int:
    """Сколько GPU видит CTranslate2. 0 — работаем на CPU."""
    try:
        from .cuda import register_nvidia_dlls

        register_nvidia_dlls()
    except Exception:
        pass
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def resolve(cfg: STTCfg) -> tuple[str, str, str]:
    """(device, model, compute_type) с учётом auto и реального железа."""
    device = (cfg.device or "auto").lower()
    if device == "auto":
        device = "cuda" if cuda_device_count() > 0 else "cpu"

    compute = (cfg.compute_type or "auto").lower()
    if compute == "auto":
        compute = _AUTO_COMPUTE.get(device, "int8")
    if device == "cpu" and compute in {"int8_float16", "float16", "bfloat16"}:
        compute = "int8"

    model = cfg.model or "auto"
    if model == "auto":
        model = _AUTO_MODEL.get(device, "base")

    return device, model, compute


class _Stats:
    def __init__(self) -> None:
        self._stats: list[float] = []

    def average_latency(self) -> float:
        return sum(self._stats) / len(self._stats) if self._stats else 0.0


# --------------------------------------------------------------------- локально


class Transcriber(_Stats):
    """Whisper на этой машине."""

    def __init__(self, cfg: STTCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._model = None
        self.device = "?"
        self.model_name = "?"
        self.compute_type = "?"

    def load(self) -> None:
        from faster_whisper import WhisperModel

        device, model_name, compute = resolve(self.cfg)
        self.device, self.model_name, self.compute_type = device, model_name, compute

        t0 = time.perf_counter()
        log.info("Загружаю Whisper '%s' (%s / %s)...", model_name, device, compute)
        try:
            # Сначала пробуем локальный кэш, чтобы не ходить на huggingface.co при каждом запуске.
            self._model = WhisperModel(
                model_name, device=device, compute_type=compute, local_files_only=True
            )
        except Exception:
            self._model = WhisperModel(model_name, device=device, compute_type=compute)
        log.info("Whisper готов за %.1f с", time.perf_counter() - t0)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def describe(self) -> str:
        if not self.loaded:
            return "не загружен"
        return f"{self.model_name} / {self.device} ({self.compute_type})"

    def transcribe(self, pcm48: np.ndarray) -> str:
        if self._model is None:
            raise RuntimeError("Whisper не загружен — вызови load()")

        samples = audio.to_float32(audio.resample(pcm48, audio.DISCORD_RATE, audio.STT_RATE))
        if samples.size < audio.STT_RATE // 5:  # меньше 0.2 с
            return ""

        t0 = time.perf_counter()
        segments, _info = self._model.transcribe(
            samples,
            language=self.cfg.language or None,
            beam_size=self.cfg.beam_size,
            condition_on_previous_text=False,
            vad_filter=True,
            initial_prompt=self.cfg.initial_prompt or None,
            temperature=0.0,
        )
        text = " ".join(
            s.text.strip() for s in segments if getattr(s, "no_speech_prob", 0.0) <= 0.85
        ).strip()
        self._stats.append(time.perf_counter() - t0)
        return text


# --------------------------------------------------------------------- по сети


class RemoteTranscriber(_Stats):
    """Whisper на другом ПК: отправляем PCM, получаем текст."""

    def __init__(self, cfg: STTCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.url = (cfg.remote_url or "").rstrip("/")
        self.token = cfg.remote_token or ""
        self._ok = False

    @property
    def loaded(self) -> bool:
        return self._ok

    def describe(self) -> str:
        state = "ok" if self._ok else "нет связи"
        return f"remote {self.url} [{state}]"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/octet-stream"}
        if self.token:
            headers["X-Token"] = self.token
        return headers

    def load(self) -> None:
        if not self.url:
            raise RuntimeError("stt.remote_url не задан в config.yaml")
        req = urllib.request.Request(f"{self.url}/health", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                info = json.loads(resp.read() or b"{}")
        except Exception as exc:
            raise RuntimeError(
                f"STT-сервер недоступен по адресу {self.url}.\n"
                "На машине с видеокартой запусти stt_server.bat и проверь адрес/сеть."
            ) from exc
        self._ok = True
        log.info("STT-сервер на связи: %s (%s)", self.url, info.get("model", "?"))

    def transcribe(self, pcm48: np.ndarray) -> str:
        if not self._ok:
            raise RuntimeError("STT-сервер не подключён")

        body = pcm48.astype(np.int16).tobytes()
        req = urllib.request.Request(
            f"{self.url}/transcribe", data=body, headers=self._headers(), method="POST"
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:200].decode("utf-8", "replace")
            raise RuntimeError(f"STT-сервер вернул {exc.code}: {detail}") from exc
        finally:
            self._stats.append(time.perf_counter() - t0)
        return str(data.get("text", "")).strip()


def create_transcriber(cfg: STTCfg):
    engine = (cfg.engine or "local").lower()
    if engine == "remote":
        return RemoteTranscriber(cfg)
    return Transcriber(cfg)
