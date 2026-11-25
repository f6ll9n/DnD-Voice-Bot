"""Аудио-примитивы: байты Discord <-> numpy, ресемплинг, моно/стерео."""

from __future__ import annotations

import numpy as np
import soxr

DISCORD_RATE = 48_000
STT_RATE = 16_000


def to_i16(pcm: bytes) -> np.ndarray:
    """Сырые s16le байты -> массив int16 (без копирования)."""
    if not pcm:
        return np.zeros(0, dtype=np.int16)
    if len(pcm) % 2:
        pcm = pcm[:-1]
    return np.frombuffer(pcm, dtype=np.int16)


def downmix(pcm: bytes, channels: int = 2) -> np.ndarray:
    """Стерео Discord (48 кГц s16le) -> моно int16."""
    data = to_i16(pcm)
    if channels <= 1 or data.size == 0:
        return np.array(data, dtype=np.int16)
    if data.size % channels:
        data = data[: data.size - (data.size % channels)]
    frames = data.reshape(-1, channels).astype(np.int32)
    return (frames.sum(axis=1) // channels).astype(np.int16)


def resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or data.size == 0:
        return np.array(data, dtype=np.int16)
    out = soxr.resample(data.astype(np.float32) / 32768.0, src_rate, dst_rate, quality="HQ")
    return np.clip(np.rint(out * 32768.0), -32768, 32767).astype(np.int16)


def to_float32(data: np.ndarray) -> np.ndarray:
    """int16 [-32768..32767] -> float32 [-1..1], как ждёт Whisper."""
    return data.astype(np.float32) / 32768.0


def to_stereo_bytes(data: np.ndarray) -> bytes:
    """Моно int16 -> стерео s16le байты (как ждёт discord.PCMAudio)."""
    if data.size == 0:
        return b""
    stereo = np.repeat(data.reshape(-1, 1), 2, axis=1)
    return np.ascontiguousarray(stereo, dtype=np.int16).tobytes()


def seconds(data: np.ndarray, rate: int) -> float:
    return data.size / float(rate)
