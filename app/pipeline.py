"""Оркестратор: фраза -> текст -> реплика персонажа -> голос."""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import discord
import numpy as np

from . import tts as tts_mod
from .config import ROOT, Config, PersonaCfg
from .llm import LLM
from .stt import create_transcriber
from .tts import Speaker

log = logging.getLogger(__name__)


@dataclass
class SpeechJob:
    guild_id: int
    persona: PersonaCfg
    text: str


class Pipeline:
    def __init__(self, cfg: Config, bot) -> None:
        self.cfg = cfg
        self.bot = bot
        self.stt = create_transcriber(cfg.stt)
        self.llm = LLM(cfg.llm)
        self.tts = Speaker(cfg.tts)
        self.enabled = True
        self._queue: asyncio.Queue[SpeechJob] = asyncio.Queue()
        self._stt_lock = asyncio.Lock()
        self._last_reply_at: dict[str, float] = {}
        self._player: asyncio.Task | None = None
        self._log_path: Path | None = None

    # ------------------------------------------------------------------ старт

    async def start(self) -> None:
        if self.cfg.input.mode in ("voice", "both"):
            try:
                await asyncio.to_thread(self.stt.load)
            except Exception as exc:
                raise RuntimeError(
                    "Не удалось загрузить Whisper. Проверь config.yaml -> stt.device "
                    "(cuda или cpu) и запусти tools\\preflight.py.\n"
                    f"Причина: {type(exc).__name__}: {exc}"
                ) from exc
        else:
            log.info("Режим '%s': распознавание речи не нужно", self.cfg.input.mode)

        await self._check_llm()
        await self._warmup_llm()
        persona_voices = [p.voice for p in self.cfg.personas.values() if p.voice]
        await self.tts.warmup(persona_voices)
        self._player = asyncio.create_task(self._player_loop(), name="tts-player")
        self._open_log()

    async def _check_llm(self) -> None:
        """Предупредить заранее, а не в середине игровой сцены."""
        try:
            models = await self.llm.ping()
        except Exception as exc:
            log.warning(
                "LLM-сервер недоступен (%s) по адресу %s. Проверь Ollama.",
                exc,
                self.cfg.llm.base_url,
            )
            return
        if self.cfg.llm.model not in models:
            log.warning(
                "Модели '%s' нет на сервере. Доступны: %s",
                self.cfg.llm.model,
                ", ".join(models[:6]) or "—",
            )
        else:
            log.info("LLM готова: %s", self.cfg.llm.model)

    async def _warmup_llm(self) -> None:
        """Первый запрос грузит модель в память — до 15 с. Лучше это сделать до игры."""
        try:
            await self.llm.client.chat.completions.create(
                model=self.cfg.llm.model,
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=1,
                temperature=0.0,
            )
            log.info("LLM прогрета")
        except Exception as exc:
            log.warning("Не удалось прогреть LLM: %s", exc)

    async def stop(self) -> None:
        if self._player is not None:
            self._player.cancel()
            try:
                await self._player
            except (asyncio.CancelledError, Exception):
                pass
        await self.llm.aclose()
        self._close_log()

    # --------------------------------------------------------------- вход

    async def on_utterance(
        self,
        member: discord.abc.User,
        pcm48: np.ndarray,
        *,
        guild: discord.Guild | None = None,
    ) -> None:
        """Реплика пришла голосом — сначала распознаём."""
        guild = guild or getattr(member, "guild", None)
        if not self.enabled or guild is None:
            return
        if self.cfg.input.mode not in ("voice", "both"):
            return
        if self.bot.active_persona(guild) is None:
            return

        t0 = time.perf_counter()
        try:
            async with self._stt_lock:
                text = await asyncio.to_thread(self.stt.transcribe, pcm48)
        except Exception:
            log.exception("Ошибка распознавания речи")
            return
        speaker_name = getattr(member, "display_name", str(member))
        if not text:
            log.info("🔇 %s: речь не распознана (шум/тишина, %.2f с)", speaker_name, pcm48.size / 48_000)
            return

        log.info("🎤 %s: %s   (STT %.2f с)", speaker_name, text, time.perf_counter() - t0)
        await self._respond(member, text, source="voice", elapsed=time.perf_counter() - t0, guild=guild)

    async def on_text(self, member: discord.Member, text: str) -> None:
        """Реплика пришла сообщением в чат — распознавать нечего."""
        if not self.enabled or getattr(member, "guild", None) is None:
            return
        if self.cfg.input.mode not in ("text", "both"):
            return

        text = " ".join(text.split())
        if not text:
            return

        log.info("💬 %s: %s", member.display_name, text)
        await self._respond(member, text, source="text")

    # --------------------------------------------------------------- ответ

    async def _respond(
        self,
        member: discord.abc.User,
        text: str,
        *,
        source: str,
        elapsed: float = 0.0,
        guild: discord.Guild | None = None,
    ) -> None:
        guild = guild or getattr(member, "guild", None)
        if guild is None:
            return
        persona = self.bot.active_persona(guild)
        if persona is None:
            return
        speaker_name = getattr(member, "display_name", str(member))

        if source == "voice":
            # В текстовом режиме реплика игрока и так видна в чате — не дублируем
            self._append_log(f"**{speaker_name}:** {text}")
            if self.cfg.post_to_chat and self.cfg.discord.post_mode != "replies":
                await self._post(guild, f"**{speaker_name}:** {text}")

        if not await self._should_answer(persona, text, source=source, history_key=f"{guild.id}:{persona.key}"):
            log.info("⏭ Пропущено (не найдено обращение к %s): %s", persona.title, text)
            return

        # Прерываем бота только на реальное обращение к нему, а не на фоновую болтовню других игроков.
        if self.cfg.interrupt_on_speech and self.bot.is_playing(guild):
            self.bot.stop_playback(guild)
            self._clear_queue()

        t1 = time.perf_counter()
        try:
            answer = await self.llm.reply(
                persona=persona,
                world=self.cfg.world,
                speaker=speaker_name,
                text=text,
                history_key=f"{guild.id}:{persona.key}",
            )
        except Exception:
            log.exception("LLM не ответил")
            return
        if not answer:
            return

        log.info(
            "🗣 %s: %s   (LLM %.2f с, всего %.2f с)",
            persona.title,
            answer,
            time.perf_counter() - t1,
            elapsed + (time.perf_counter() - t1),
        )
        self._append_log(f"**{persona.title}:** {answer}")
        self._last_reply_at[f"{guild.id}:{persona.key}"] = time.monotonic()

        if self.cfg.post_to_chat:
            await self._post(guild, f"**{persona.title}:** {answer}")

        await self._queue.put(SpeechJob(guild.id, persona, answer))

    # ------------------------------------------------------------------ gate

    @staticmethod
    def _looks_like_followup(text: str) -> bool:
        """Отсекает короткие междометия вроде 'ну да' и фразы в 3-м лице ('он понял')."""
        words = [w.strip(".,!?…:;«»\"'()—-") for w in text.lower().split()]
        words = [w for w in words if w]
        if len(words) < 3:
            return False
        if "он" in words or "его" in words or "ему" in words or "бот" in words:
            if "ты" not in words and "тебе" not in words and "тебя" not in words:
                return False
        if "?" in text:
            return True
        cues = {
            "ты", "тебе", "тебя", "твой", "сделай", "налей", "дай", "держи",
            "сколько", "почем", "почём", "скинь", "добавь", "приготовь", "расскажи",
        }
        return any(w in cues for w in words)

    async def _should_answer(
        self,
        persona: PersonaCfg,
        text: str,
        *,
        source: str = "voice",
        history_key: str = "",
    ) -> bool:
        # В текстовом канале сам факт сообщения — уже обращение
        if source == "text" and not self.cfg.input.text_requires_name:
            return True

        mode = (persona.addressing or "name").lower()
        if mode == "always":
            return True
        if mode == "llm":
            try:
                return await self.llm.is_addressed(persona, text)
            except Exception:
                log.exception("Классификатор не ответил — пропускаю реплику")
                return False
        # по умолчанию: ищем кодовое слово/имя
        low = text.lower()
        triggers = [w.lower() for w in persona.trigger_words if w]
        if any(w in low for w in triggers) if triggers else True:
            return True

        window = float(getattr(self.cfg.input, "follow_up_s", 0.0) or 0.0)
        if window > 0 and history_key:
            last_t = self._last_reply_at.get(history_key, 0.0)
            if last_t and (time.monotonic() - last_t) <= window and self._looks_like_followup(text):
                log.info("💬 Продолжение диалога без повтора имени: %s", text)
                return True
        return False

    # ------------------------------------------------------------- воспроизведение

    async def _player_loop(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._speak(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Ошибка озвучки")
            finally:
                self._queue.task_done()

    async def _speak(self, job: SpeechJob) -> None:
        vc = self.bot.voice_client_for(job.guild_id)
        if vc is None or not vc.is_connected():
            log.debug("Голосовой клиент недоступен — пропускаю озвучку")
            return

        for sentence in tts_mod.split_sentences(job.text):
            pcm = await self.tts.synthesize_pcm(
                sentence, job.persona.voice, job.persona.rate, job.persona.pitch
            )
            if not pcm:
                continue

            vc = self.bot.voice_client_for(job.guild_id)
            if vc is None or not vc.is_connected():
                return

            done = asyncio.Event()
            source = discord.PCMAudio(io.BytesIO(pcm))

            def _after(err: Exception | None) -> None:
                if err is not None:
                    log.error("Ошибка воспроизведения в голосовой канал: %s", err)
                self.bot.loop.call_soon_threadsafe(done.set)

            vc.play(source, after=_after)
            try:
                await asyncio.wait_for(done.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                log.warning("Таймаут воспроизведения озвучки — сбрасываю плеер")
                if vc.is_playing():
                    vc.stop()

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                self._queue.task_done()

    # ------------------------------------------------------------------ прочее

    async def _post(self, guild, text: str) -> None:
        channel = self.bot.post_channel(guild)
        if channel is None:
            return
        try:
            await channel.send(text[:1900])
        except discord.HTTPException:
            log.warning("Не смог отправить текст в канал")

    def _open_log(self) -> None:
        log_dir = Path(self.cfg.log_dir)
        if not log_dir.is_absolute():
            log_dir = ROOT / log_dir
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M")
            self._log_path = log_dir / f"session-{stamp}.md"
            self._log_path.write_text(
                f"# Сессия {datetime.now():%d.%m.%Y %H:%M}\n\n", encoding="utf-8"
            )
            log.info("Транскрипт сессии: %s", self._log_path)
        except OSError:
            log.warning("Не смог создать файл лога — пишу только в консоль")
            self._log_path = None

    def _append_log(self, line: str) -> None:
        if self._log_path is None:
            return
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"- {datetime.now():%H:%M:%S}  {line}\n")
        except OSError:
            pass

    def _close_log(self) -> None:
        self._log_path = None
