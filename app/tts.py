"""Текст -> речь.

Три движка:
  silero — локальный нейросетевой, офлайн, 48 кГц, живые русские голоса. Основной.
  piper  — локальный, офлайн, 22 кГц, самый быстрый на слабых CPU.
  edge   — Microsoft, онлайн, бывает недоступен и даёт задержки.

Наружу отдаём готовый PCM 48 кГц stereo — ровно то, что ест discord.PCMAudio.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import warnings
import wave
from pathlib import Path

import edge_tts
import miniaudio
import numpy as np

from . import audio
from .config import ROOT
from .text import split_sentences  # реэкспорт: pipeline берёт его отсюда

__all__ = ["SILERO_SPEAKERS", "Speaker", "rate_to_length_scale", "split_sentences"]

log = logging.getLogger(__name__)

OUT_RATE = audio.DISCORD_RATE

# Голоса edge-tts, которые хорошо звучат по-русски
RU_VOICES = ("ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural", "ru-RU-DariyaNeural")
SILERO_SPEAKERS = ("eugene", "aidar", "xenia", "baya", "kseniya")


def rate_to_length_scale(rate: str) -> float | None:
    """Преобразует строку вроде '-6%' или '+15%' в length_scale для Piper."""
    raw = (rate or "").strip()
    if not raw or raw in {"+0%", "0%", "-0%"}:
        return None
    try:
        pct = float(raw.rstrip("%").strip())
    except ValueError:
        return None
    speed = 1.0 + pct / 100.0
    if speed <= 0.1:
        speed = 0.1
    scale = 1.0 / speed
    return round(max(0.5, min(2.0, scale)), 3)


class Speaker:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.engine = (getattr(cfg, "engine", "edge") or "edge").lower()
        self._voices: dict[str, object] = {}
        self._silero = None
        self._silero_lock = threading.Lock()
        self._voice_ok: str | None = None

    # ------------------------------------------------------------- публичный API

    async def synthesize_pcm(
        self,
        text: str,
        voice: str = "",
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> bytes:
        """Текст -> PCM 48 кГц stereo s16le."""
        text = text.strip()
        if not text:
            return b""

        if self.engine == "silero":
            return await asyncio.to_thread(self._silero_pcm, text, voice)

        if self.engine == "piper":
            return await asyncio.to_thread(self._piper_pcm, text, voice, rate)

        mp3 = await self.synthesize(text, voice, rate, pitch)
        return await asyncio.to_thread(self.to_pcm48_stereo, mp3)

    # ------------------------------------------------------------------ silero

    def _silero_path(self) -> Path:
        raw = getattr(self.cfg, "silero_model", "") or "models/silero/v5_ru.pt"
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p

    def _load_silero(self):
        if self._silero is not None:
            return self._silero
        import torch

        path = self._silero_path()
        if not path.is_file():
            raise FileNotFoundError(f"Модель Silero не найдена: {path}")

        log.info("Загружаю Silero TTS: %s", path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
        model.to("cpu")
        self._silero = model
        return model

    def _silero_pcm(self, text: str, voice: str = "", *, strict: bool = False) -> bytes:
        if not any(ch.isalnum() for ch in text):
            return b""
        try:
            model = self._load_silero()
        except Exception as exc:
            if strict:
                raise
            log.error("Ошибка загрузки Silero (%s): %s", self._silero_path(), exc)
            return b""

        speakers = tuple(getattr(model, "speakers", SILERO_SPEAKERS) or SILERO_SPEAKERS)
        speaker = voice if voice in speakers else ("eugene" if "eugene" in speakers else speakers[0])

        try:
            with self._silero_lock:
                tensor = model.apply_tts(
                    text=text,
                    speaker=speaker,
                    sample_rate=OUT_RATE,
                    put_accent=True,
                    put_stress_homo=True,
                    put_yo=True,
                )
        except Exception as exc:
            if strict:
                raise
            log.warning("Silero не смог озвучить '%s': %s", text[:60], exc)
            return b""

        arr = tensor.detach().cpu().numpy()
        samples = np.clip(np.rint(arr * 32767.0), -32768, 32767).astype(np.int16)
        return audio.to_stereo_bytes(samples)

    # ------------------------------------------------------------------- piper

    def _piper_path(self, voice: str) -> str:
        raw = (voice or "").strip()
        if raw and raw.lower().endswith(".onnx"):
            candidate = Path(raw)
        else:
            fallback = getattr(self.cfg, "piper_model", "") or "models/piper/ru_RU-ruslan-medium.onnx"
            candidate = Path(fallback)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        return str(candidate)

    def _load_piper(self, path: str):
        cached = self._voices.get(path)
        if cached is not None:
            return cached
        from piper import PiperVoice

        if not Path(path).is_file():
            raise FileNotFoundError(f"Модель голоса Piper не найдена: {path}")

        log.info("Загружаю голос Piper: %s", path)
        voice = PiperVoice.load(path)
        self._voices[path] = voice
        return voice

    def _piper_pcm(self, text: str, voice: str, rate: str = "+0%", *, strict: bool = False) -> bytes:
        path = self._piper_path(voice)
        try:
            engine = self._load_piper(path)
        except Exception as exc:
            if strict:
                raise
            log.error("Ошибка загрузки Piper (%s): %s", path, exc)
            return b""

        syn_config = None
        length_scale = rate_to_length_scale(rate)
        if length_scale is not None:
            from piper import SynthesisConfig

            syn_config = SynthesisConfig(length_scale=length_scale)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            engine.synthesize_wav(text, wf, syn_config=syn_config)

        buf.seek(0)
        with wave.open(buf, "rb") as rf:
            src_rate = rf.getframerate()
            channels = rf.getnchannels()
            raw = rf.readframes(rf.getnframes())

        samples = np.frombuffer(raw, dtype=np.int16)
        if channels > 1:
            samples = samples.reshape(-1, channels)[:, 0].copy()

        samples = audio.resample(samples, src_rate, OUT_RATE)
        return audio.to_stereo_bytes(samples)

    # -------------------------------------------------------------------- edge

    async def synthesize(self, text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> bytes:
        """Текст -> mp3-байты (edge-tts с повторами и подменой голоса)."""
        text = text.strip()
        if not text:
            return b""

        primary = voice or RU_VOICES[0]
        chain: list[str] = []
        for name in ([self._voice_ok] if self._voice_ok else []) + [primary, *RU_VOICES]:
            if name and name not in chain:
                chain.append(name)

        last: Exception | None = None
        for name in chain:
            try:
                data = await asyncio.wait_for(self._synth(text, name, rate, pitch), timeout=8)
            except Exception as exc:
                last = exc
                log.warning("Озвучка не удалась (%s): %s", name, type(exc).__name__)
                continue
            if data:
                if self._voice_ok != name:
                    self._voice_ok = name
                    log.info("Озвучка: рабочий голос %s", name)
                return data

        raise RuntimeError(f"Синтез речи не отвечает: {last}")

    @staticmethod
    async def _synth(text: str, voice: str, rate: str, pitch: str) -> bytes:
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    @staticmethod
    def to_pcm48_stereo(mp3: bytes) -> bytes:
        """mp3 -> PCM 48 кГц stereo s16le."""
        if not mp3:
            return b""
        decoded = miniaudio.decode(
            mp3,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=OUT_RATE,
        )
        samples = np.asarray(decoded.samples, dtype=np.int16)
        return audio.to_stereo_bytes(samples)

    async def warmup(self, voices: list[str] | None = None) -> None:
        """Прогрев движка, чтобы первая реплика не тормозила."""
        if self.engine == "silero":
            first = next((v for v in (voices or []) if v), "eugene")
            await asyncio.to_thread(self._silero_pcm, "проверка", first, strict=True)
            return
        if self.engine == "piper":
            targets = list(dict.fromkeys(v for v in (voices or [""]) if v is not None)) or [""]
            for v in targets:
                await asyncio.to_thread(self._piper_pcm, "проверка", v, "+0%", strict=True)
        else:
            await self.synthesize("проверка", RU_VOICES[0])
