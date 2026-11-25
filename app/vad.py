"""Нарезка непрерывного потока на фразы по детектору речи (webrtcvad).

Работает с моно 48 кГц int16 — ровно в том виде, в какем приходит голос из Discord.
Один экземпляр держит состояние на каждого участника отдельно.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import webrtcvad

log = logging.getLogger(__name__)

FRAME_MS = 20


@dataclass
class SegmenterCfg:
    aggressiveness: int = 2
    start_ms: int = 140
    end_ms: int = 700
    min_speech_ms: int = 350
    max_utterance_s: float = 25.0
    preroll_ms: int = 300
    keep_tail_ms: int = 200
    sample_rate: int = 48_000


class _State:
    __slots__ = ("pending", "preroll", "active", "in_speech", "voiced_ms", "silence_ms", "speech_ms")

    def __init__(self, preroll_frames: int) -> None:
        self.pending: np.ndarray = np.zeros(0, dtype=np.int16)
        self.preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)
        self.active: list[np.ndarray] = []
        self.in_speech = False
        self.voiced_ms = 0
        self.silence_ms = 0
        self.speech_ms = 0


class UtteranceSegmenter:
    """Скармливаешь куски моно-PCM, получаешь готовые фразы."""

    def __init__(self, cfg: SegmenterCfg, on_utterance: Callable[[int, np.ndarray], None]) -> None:
        self.cfg = cfg
        self._vad = webrtcvad.Vad(max(0, min(3, cfg.aggressiveness)))
        self._frame_samples = cfg.sample_rate * FRAME_MS // 1000
        self._frame_bytes = self._frame_samples * 2
        self._preroll_frames = max(1, cfg.preroll_ms // FRAME_MS)
        self._states: dict[int, _State] = {}
        self._on_utterance = on_utterance
        self._lock = threading.Lock()
        self.voiced_frames = 0  # сколько кадров VAD счёл речью (диагностика)

    # ------------------------------------------------------------------ public

    def feed(self, uid: int, mono: np.ndarray) -> None:
        if mono.size == 0:
            return
        with self._lock:
            self._feed(uid, mono)

    def _feed(self, uid: int, mono: np.ndarray) -> None:
        st = self._states.get(uid)
        if st is None:
            st = _State(self._preroll_frames)
            self._states[uid] = st

        st.pending = np.concatenate((st.pending, mono)) if st.pending.size else mono

        n = self._frame_samples
        while st.pending.size >= n:
            frame = st.pending[:n].copy()
            st.pending = st.pending[n:]
            self._process(uid, st, frame)

    def reset(self, uid: int) -> None:
        self._states.pop(uid, None)

    def flush(self, uid: int) -> None:
        """Принудительно завершить текущую фразу (например, человек вышел из канала)."""
        with self._lock:
            self._flush(uid)

    def _flush(self, uid: int) -> None:
        st = self._states.get(uid)
        if st is None or not st.in_speech or not st.active:
            self._states.pop(uid, None)
            return
        audio = np.concatenate(st.active) if len(st.active) > 1 else st.active[0]
        speech_ms = st.speech_ms
        self._states.pop(uid, None)
        if speech_ms >= self.cfg.min_speech_ms:
            self._on_utterance(uid, audio)

    # ----------------------------------------------------------------- private

    def _process(self, uid: int, st: _State, frame: np.ndarray) -> None:
        voiced = bool(self._vad.is_speech(frame.tobytes(), self.cfg.sample_rate))
        if voiced:
            self.voiced_frames += 1

        if not st.in_speech:
            st.preroll.append(frame)
            if voiced:
                st.voiced_ms += FRAME_MS
                if st.voiced_ms >= self.cfg.start_ms:
                    st.in_speech = True
                    st.active = list(st.preroll)
                    st.preroll.clear()
                    st.speech_ms = st.voiced_ms
                    st.silence_ms = 0
            else:
                st.voiced_ms = 0
            return

        st.active.append(frame)
        if voiced:
            st.silence_ms = 0
            st.speech_ms += FRAME_MS
        else:
            st.silence_ms += FRAME_MS

        total_s = len(st.active) * FRAME_MS / 1000.0
        if st.silence_ms < self.cfg.end_ms and total_s < self.cfg.max_utterance_s:
            return

        tail_silence_ms = st.silence_ms
        speech_ms = st.speech_ms
        chunks = st.active
        st.in_speech = False
        st.active = []
        st.voiced_ms = 0
        st.silence_ms = 0
        st.speech_ms = 0

        if speech_ms < self.cfg.min_speech_ms:
            return

        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        cut = int((tail_silence_ms - self.cfg.keep_tail_ms) * self.cfg.sample_rate / 1000)
        if 0 < cut < audio.size:
            audio = audio[: audio.size - cut]

        self._on_utterance(uid, audio)
