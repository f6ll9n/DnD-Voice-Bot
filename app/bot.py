"""Discord-бот: приём голоса, команды, воспроизведение."""

from __future__ import annotations

import asyncio
import logging
import time
import wave
from typing import Any

import discord
import numpy as np
from discord.ext import commands

try:
    from discord.ext import voice_recv
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Не установлен discord-ext-voice-recv.\n"
        "Запусти install.bat или вручную: pip install discord-ext-voice-recv"
    ) from exc

from . import audio
from .dave import install as install_dave_receive
from .commands import BotCommands
from .config import Config
from .pipeline import Pipeline
from .vad import SegmenterCfg, UtteranceSegmenter

log = logging.getLogger(__name__)

# Без этого приём голоса падает с OpusError: corrupted stream (см. app/dave.py)
install_dave_receive()


# --------------------------------------------------------------------------- приём


class AudioHub:
    """Мост между потоком приёма голоса и asyncio-миром бота."""

    def __init__(self, bot: "DnDBot", pipeline: Pipeline, guild: discord.Guild) -> None:
        self.bot = bot
        self.pipeline = pipeline
        self.guild = guild
        self.loop = bot.loop
        self._seen: set[int] = set()
        self._users: dict[int, discord.abc.User] = {}
        self._count: dict[int, int] = {}
        self._dump: dict[int, bytearray] = {}
        self._dumped: set[int] = set()
        self._last: dict[int, float] = {}
        self.stopped = False
        cfg = bot.cfg.vad
        self.segmenter = UtteranceSegmenter(
            SegmenterCfg(
                aggressiveness=cfg.aggressiveness,
                start_ms=cfg.start_ms,
                end_ms=cfg.end_ms,
                min_speech_ms=cfg.min_speech_ms,
                max_utterance_s=cfg.max_utterance_s,
            ),
            on_utterance=self._emit,
        )

    def write(self, user: discord.abc.User, pcm: bytes) -> None:
        if user.bot:
            return
        member = self.guild.get_member(user.id) or user
        self._users[user.id] = member
        mono = audio.downmix(pcm, 2)
        self._last[member.id] = time.monotonic()
        peak = int(np.abs(mono).max()) if mono.size else 0

        count = self._count.get(member.id, 0) + 1
        self._count[member.id] = count
        self._dump_audio(member, mono)

        if count == 1 or count % 250 == 0:  # первый пакет и далее раз в ~5 секунд
            log.info(
                "🎧 %s: пакетов %d, пик %d, кадров речи %d",
                getattr(member, "display_name", str(member)),
                count,
                peak,
                self.segmenter.voiced_frames,
            )

        self.segmenter.feed(member.id, mono)

    async def watchdog(self) -> None:
        """Discord не шлёт пакеты, когда человек молчит (DTX).

        Без этого «фраза закончилась» никогда не наступает: ждём тишину,
        а тишины в потоке нет. Поэтому сами досыпаем её, если пакетов нет.
        """
        silence = np.zeros(4800, dtype=np.int16)  # 100 мс
        try:
            while not self.stopped:
                await asyncio.sleep(0.1)
                now = time.monotonic()
                for uid, last in list(self._last.items()):
                    if now - last < 0.15:
                        continue
                    self._last[uid] = now
                    self.segmenter.feed(uid, silence)
        except asyncio.CancelledError:
            pass

    def _dump_audio(self, member: discord.abc.User, mono: np.ndarray) -> None:
        """Первые 3 секунды речи каждого участника — в WAV, для отладки."""
        if member.id in self._dumped:
            return
        buf = self._dump.setdefault(member.id, bytearray())
        limit = 48_000 * 2 * 3
        if len(buf) >= limit:
            return
        buf.extend(mono.tobytes())
        if len(buf) < limit:
            return

        self._dumped.add(member.id)
        try:
            from .config import ROOT

            logs = ROOT / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            path = logs / f"debug-{member.id}.wav"
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48_000)
                wf.writeframes(bytes(buf))
            log.info("дамп аудио: %s", path)
        except Exception:
            log.exception("не смог записать дамп аудио")

    def _emit(self, uid: int, pcm48: np.ndarray) -> None:
        member = self.guild.get_member(uid) or self._users.get(uid)
        if member is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.pipeline.on_utterance(member, pcm48, guild=self.guild), self.loop
            )
        except RuntimeError:
            pass  # цикл уже закрывается


class VoiceSink(voice_recv.AudioSink):
    def __init__(self, hub: AudioHub) -> None:
        super().__init__()
        self.hub = hub

    def wants_opus(self) -> bool:  # False -> библиотека декодирует Opus сама
        return False

    def write(self, user, data) -> None:  # вызывается из потока приёма
        if user is None:
            return
        self.hub.write(user, data.pcm)

    def cleanup(self) -> None:
        pass


# ----------------------------------------------------------------------------- бот


class DnDBot(commands.Bot):
    def __init__(self, cfg: Config, *, message_content: bool = True) -> None:
        intents = discord.Intents.default()
        # Без MESSAGE CONTENT INTENT Discord не пустит бота вообще.
        # Тогда работаем только на слеш-командах.
        intents.message_content = message_content
        intents.voice_states = True
        super().__init__(command_prefix=cfg.discord.prefix, intents=intents, help_command=None)

        self.cfg = cfg
        self.pipeline: Pipeline | None = None
        self._persona: dict[int, str | None] = {}
        self._voice: dict[int, voice_recv.VoiceRecvClient] = {}
        self._hubs: dict[int, "AudioHub"] = {}
        self._channel: dict[int, discord.TextChannel] = {}
        self._auto_joined = False
        self.has_message_content = message_content

    # ------------------------------------------------------------- жизненный цикл

    async def setup_hook(self) -> None:
        # Команды живут в отдельном Cog — только так discord.py
        # регистрирует их и как слеш, и как префиксные.
        await self.add_cog(BotCommands(self))
        log.info(
            "Команд: %d, из них слеш в дереве: %d",
            len(self.commands),
            len(self.tree.get_commands()),
        )

        self.pipeline = Pipeline(self.cfg, self)
        await self.pipeline.start()

    async def on_ready(self) -> None:
        assert self.user is not None
        perms = discord.Permissions(
            view_channel=True,
            send_messages=True,
            read_message_history=True,
            connect=True,
            speak=True,
            use_voice_activation=True,
        )
        invite = discord.utils.oauth_url(
            self.user.id,
            scopes=("bot", "applications.commands"),
            permissions=perms,
        )
        log.info("Бот в сети: %s (id %s)", self.user, self.user.id)
        log.info("Ссылка для приглашения на сервер:\n%s", invite)

        if not self._auto_joined:
            self._auto_joined = True
            await self._sync_commands()
            await self._auto_join()

    async def on_guild_join(self, guild: discord.Guild) -> None:
        """Бота добавили на сервер — сразу регистрируем там команды."""
        try:
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            log.info("Слеш-команды зарегистрированы на «%s»", guild.name)
        except Exception:
            log.exception("Не удалось зарегистрировать команды на «%s»", guild.name)

    async def _sync_commands(self) -> None:
        """Слеш-команды на каждом сервере — так они появляются сразу, а не через час."""
        for guild in self.guilds:
            try:
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
            except Exception:
                log.exception("Не удалось синхронизировать слеш-команды на «%s»", guild.name)
        log.info("Слеш-команды готовы на %d сервер(ах)", len(self.guilds))

    # --------------------------------------------------------- подключение

    async def connect_voice(self, channel: discord.VoiceChannel) -> voice_recv.VoiceRecvClient | None:
        """Зайти в канал и начать слушать. Общая логика для команды и автозапуска."""
        if self.pipeline is None:
            return None

        vc: Any = channel.guild.voice_client
        if vc is not None and not isinstance(vc, voice_recv.VoiceRecvClient):
            await vc.disconnect(force=True)
            vc = None

        if vc is None:
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False)
        elif vc.channel != channel:
            await vc.move_to(channel)

        if not isinstance(vc, voice_recv.VoiceRecvClient):  # pragma: no cover
            return None

        if vc.is_listening():
            vc.stop_listening()

        old = self._hubs.pop(channel.guild.id, None)
        if old is not None:
            old.stopped = True

        hub = AudioHub(self, self.pipeline, channel.guild)
        self._hubs[channel.guild.id] = hub
        vc.listen(VoiceSink(hub))
        asyncio.create_task(hub.watchdog(), name=f"vad-watchdog-{channel.guild.id}")
        self._voice[channel.guild.id] = vc
        return vc

    async def _auto_join(self) -> None:
        """Автоподключение из config.yaml -> discord.auto_join_voice."""
        voice_id = self.cfg.discord.auto_join_voice
        if not voice_id:
            return

        channel = self.get_channel(voice_id)
        if not isinstance(channel, discord.VoiceChannel):
            log.warning("auto_join_voice: канал %s не найден или это не голосовой", voice_id)
            return

        try:
            vc = await self.connect_voice(channel)
        except Exception:
            log.exception("Автоподключение не удалось")
            return
        if vc is None:
            return

        text_id = self.cfg.discord.auto_join_text
        if text_id:
            text_channel = self.get_channel(text_id)
            if isinstance(text_channel, discord.TextChannel):
                self._channel[channel.guild.id] = text_channel

        persona = self.active_persona(channel.guild)
        log.info(
            "Автоподключён к «%s». Персонаж: %s",
            channel.name,
            persona.title if persona else "выключен",
        )

    # ------------------------------------------------------------ текстовый вход

    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None:
            return

        if not message.author.bot:
            await self.process_commands(message)
            if message.content.startswith(self.command_prefix):
                return

        await self._handle_chat_line(message)

    async def _handle_chat_line(self, message: discord.Message) -> None:
        """Любая строка в канале персонажа — реплика игрока."""
        cfg = self.cfg.input
        if self.pipeline is None or cfg.mode not in ("text", "both"):
            return

        if message.author.bot:
            if not cfg.accept_bots:
                return
            if cfg.bot_whitelist and message.author.id not in cfg.bot_whitelist:
                return

        channel = self.npc_text_channel(message.guild)
        if channel is None or message.channel.id != channel.id:
            return

        text = message.clean_content.strip()
        if not text:
            return

        author = message.guild.get_member(message.author.id) or message.author
        await self.pipeline.on_text(author, text)

    def npc_text_channel(self, guild: discord.Guild) -> discord.TextChannel | None:
        configured = self.cfg.input.text_channel
        if configured:
            channel = guild.get_channel(configured)
            if isinstance(channel, discord.TextChannel):
                return channel
        return self._channel.get(guild.id)

    async def close(self) -> None:
        for vc in list(self._voice.values()):
            try:
                vc.stop_listening()
            except Exception:
                pass
        if self.pipeline is not None:
            await self.pipeline.stop()
        await super().close()

    # ------------------------------------------------------------------ доступ

    def active_persona(self, guild: discord.Guild):
        key = self._persona.get(guild.id, self.cfg.default_persona)
        if key is None:
            return None
        return self.cfg.personas.get(key)

    def voice_client_for(self, guild_id: int) -> voice_recv.VoiceRecvClient | None:
        vc = self._voice.get(guild_id)
        if vc is not None and vc.is_connected():
            return vc
        return None

    def post_channel(self, guild: discord.Guild) -> discord.TextChannel | None:
        return self._channel.get(guild.id)

    def is_playing(self, guild: discord.Guild) -> bool:
        vc = self.voice_client_for(guild.id)
        return bool(vc and vc.is_playing())

    def stop_playback(self, guild: discord.Guild) -> None:
        vc = self.voice_client_for(guild.id)
        if vc and vc.is_playing():
            vc.stop()

