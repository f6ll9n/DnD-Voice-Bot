"""Приём голоса с поддержкой DAVE (E2EE).

discord-ext-voice-recv расшифровывает только транспорт (XSalsa20-Poly1305),
а с 2025 года Discord обязательно включает ещё и DAVE — сквозное шифрование
поверх транспорта. Без него Opus-декодер получает зашифрованные байты,
падает с "corrupted stream", и вся ветка приёма умирает.

Отключить DAVE нельзя: Discord закрывает соединение с кодом 4017.

Поэтому подменяем `PacketDecoder._decode_packet`: перед декодированием
расшифровываем кадр ключом из dave-сессии.
"""

from __future__ import annotations

import logging
import time

import davey
from discord.opus import OpusError

log = logging.getLogger(__name__)

_installed = False
_LAST_WARN: dict[tuple[str, int], float] = {}
OPUS_SILENCE = b"\xf8\xff\xfe"

# Расшифрованные кадры: RTPPacket — с __slots__, вешать атрибуты нельзя.
# Ключ — (ssrc, sequence, timestamp), чтобы не путать пакеты при сбросе sequence.
_CACHE: dict[tuple[int, int, int | None], bytes] = {}
_CACHE_MAX = 512


def _warn_throttled(kind: str, key_id: int, msg: str, *args) -> None:
    now = time.monotonic()
    k = (kind, key_id)
    if now - _LAST_WARN.get(k, 0.0) >= 5.0:
        _LAST_WARN[k] = now
        log.warning(msg, *args)


def _connection_of(decoder):
    try:
        voice_client = decoder.sink.voice_client
        return getattr(voice_client, "_connection", None)
    except Exception:
        return None


def _session_of(decoder):
    """Возвращает (dave_protocol_version, dave_session) из голосового клиента."""
    connection = _connection_of(decoder)
    if connection is None:
        return 0, None
    dave_ver = int(getattr(connection, "dave_protocol_version", 0) or 0)
    session = getattr(connection, "dave_session", None)
    return dave_ver, session


def _user_id_of(decoder) -> int | None:
    uid = decoder._cached_id
    if uid is None:
        try:
            uid = decoder.sink.voice_client._get_id_from_ssrc(decoder.ssrc)
        except Exception:
            uid = None
        decoder._cached_id = uid
    return uid


def _apply_dave(decoder, packet) -> bool:
    """Расшифровывает пакет на месте. False — декодировать нельзя, отдаём тишину."""
    raw = getattr(packet, "decrypted_data", None)
    if not raw or raw == OPUS_SILENCE:
        return True

    dave_ver, session = _session_of(decoder)
    if dave_ver <= 0 and (session is None or not getattr(session, "ready", False)):
        return True  # DAVE не активен, данные уже открытые

    if session is None or not getattr(session, "ready", False):
        _warn_throttled(
            "not-ready",
            decoder.ssrc,
            "DAVE: сессия ещё не готова (ver=%s, ssrc=%s), пропускаю пакет",
            dave_ver,
            decoder.ssrc,
        )
        return False

    seq = getattr(packet, "sequence", -1)
    ts = getattr(packet, "timestamp", None)
    cache_key = (decoder.ssrc, seq, ts)
    if seq >= 0:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            packet.decrypted_data = cached
            return True

    uid = _user_id_of(decoder)
    if not uid:
        _warn_throttled(
            "no-uid",
            decoder.ssrc,
            "DAVE: неизвестен user_id для ssrc=%s, пропускаю пакет",
            decoder.ssrc,
        )
        return False

    try:
        plain = session.decrypt(uid, davey.MediaType.audio, bytes(raw))
    except Exception as exc:
        _warn_throttled(
            "decrypt",
            uid,
            "DAVE: ошибка расшифровки user=%s ssrc=%s (%s: %s)",
            uid,
            decoder.ssrc,
            type(exc).__name__,
            exc,
        )
        return False

    if plain is None:
        return True  # кадр не шифровался (passthrough)

    if seq >= 0:
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.clear()
        _CACHE[cache_key] = plain
    packet.decrypted_data = plain
    return True


def _safe_decode(decoder, data: bytes | None, *, fec: bool) -> bytes:
    try:
        return decoder._decoder.decode(data, fec=fec)
    except OpusError as exc:
        dave_ver, session = _session_of(decoder)
        head = bytes(data[:8]).hex() if data else ""
        _warn_throttled(
            "opus",
            decoder.ssrc,
            "OpusError (%s) ssrc=%s len=%s dave_ver=%s ready=%s head=%s — сбрасываю кадр",
            exc,
            decoder.ssrc,
            len(data) if data else 0,
            dave_ver,
            getattr(session, "ready", None),
            head,
        )
        return decoder._decoder.decode(None, fec=False)


def _decode_packet(self, packet):
    assert self._decoder is not None

    if packet:
        if _apply_dave(self, packet):
            return packet, _safe_decode(self, packet.decrypted_data, fec=False)
        return packet, self._decoder.decode(None, fec=False)

    next_packet = self._buffer.peek_next()
    if next_packet is not None and _apply_dave(self, next_packet):
        pcm = _safe_decode(self, next_packet.decrypted_data, fec=True)
    else:
        pcm = self._decoder.decode(None, fec=False)

    return packet, pcm


def install() -> None:
    global _installed
    if _installed:
        return

    from discord.ext.voice_recv import opus as vr_opus

    vr_opus.PacketDecoder._decode_packet = _decode_packet
    _installed = True
    log.info("DAVE-расшифровка приёма включена")
