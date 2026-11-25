"""Команды бота.

Отдельный Cog — обязательное условие: discord.py не регистрирует команды,
объявленные методами класса Bot, и вдобавок тащит `self` в параметры
слеш-команды, из-за чего Discord отвечает «Приложение не отвечает».
В Cog всё это обрабатывается правильно.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from .pipeline import SpeechJob

if TYPE_CHECKING:
    from .bot import DnDBot

log = logging.getLogger(__name__)


class BotCommands(commands.Cog):
    def __init__(self, bot: "DnDBot") -> None:
        self.bot = bot

    # ------------------------------------------------------------------ голос

    @commands.guild_only()
    @commands.hybrid_command(
        name="join",
        aliases=["войти", "зайти"],
        description="Позвать бота в голосовой канал, где ты сидишь",
    )
    async def join_cmd(self, ctx: commands.Context) -> None:
        """Заходит в голосовой канал вызывающего и начинает слушать."""
        assert ctx.guild is not None
        # Слеш-команда обязана ответить за 3 секунды, префиксная — нет
        if ctx.interaction is not None and not ctx.interaction.response.is_done():
            await ctx.defer()

        voice_state = getattr(ctx.author, "voice", None)
        channel = getattr(voice_state, "channel", None)
        if channel is None:
            await ctx.reply("Сначала зайди в голосовой канал, потом зови.")
            return

        if await self.bot.connect_voice(channel) is None:  # pragma: no cover
            await ctx.reply("Не удалось подключиться в режиме приёма голоса.")
            return

        self.bot._channel[ctx.guild.id] = ctx.channel
        persona = self.bot.active_persona(ctx.guild)

        await ctx.reply(
            f"Слушаю «{channel.name}». "
            + (f"Активный персонаж: **{persona.title}**." if persona else "Персонаж выключен (!npc <имя>).")
            + "\nСкажи его имя, чтобы он ответил."
        )

    @commands.guild_only()
    @commands.hybrid_command(
        name="leave", aliases=["выйти", "уйти"], description="Выйти из голосового канала"
    )
    async def leave_cmd(self, ctx: commands.Context) -> None:
        assert ctx.guild is not None
        hub = self.bot._hubs.pop(ctx.guild.id, None)
        if hub is not None:
            hub.stopped = True

        vc = self.bot._voice.pop(ctx.guild.id, None)
        if vc is None:
            await ctx.reply("Я и так не в канале.")
            return
        try:
            vc.stop_listening()
        except Exception:
            pass
        await vc.disconnect()
        await ctx.reply("Вышел.")

    # ------------------------------------------------------------- персонажи

    @commands.guild_only()
    @commands.hybrid_command(
        name="npc", aliases=["нпс"], description="Выбрать персонажа или показать список"
    )
    async def npc_cmd(self, ctx: commands.Context, *, arg: str = "") -> None:
        assert ctx.guild is not None
        value = arg.strip().lower()

        if not value or value in {"list", "список"}:
            lines = [f"`{k}` — {p.title} (`{p.voice}`)" for k, p in self.bot.cfg.personas.items()]
            current = self.bot.active_persona(ctx.guild)
            lines.append("")
            lines.append(f"Сейчас: **{current.title}**" if current else "Сейчас: выключен")
            lines.append("Выключить: `/npc off`")
            await ctx.reply("\n".join(lines))
            return

        if value in {"off", "none", "выкл"}:
            self.bot._persona[ctx.guild.id] = None
            await ctx.reply("Персонаж выключен. Голос не слушаю.")
            return

        persona = self.bot.cfg.personas.get(value)
        if persona is None:
            await ctx.reply(f"Нет такого персонажа. Есть: {', '.join(self.bot.cfg.personas)}")
            return

        self.bot._persona[ctx.guild.id] = value
        if self.bot.pipeline is not None:
            self.bot.pipeline.llm.reset(f"{ctx.guild.id}:{value}")
        await ctx.reply(f"Активный персонаж: **{persona.title}** ({persona.voice}).")

    # ------------------------------------------------------------------ речь

    @commands.guild_only()
    @commands.hybrid_command(
        name="say", aliases=["скажи"], description="Озвучить текст голосом персонажа"
    )
    async def say_cmd(self, ctx: commands.Context, *, text: str = "") -> None:
        assert ctx.guild is not None
        persona = self.bot.active_persona(ctx.guild)
        if persona is None or not text.strip():
            await ctx.reply("Нужен активный персонаж и текст: `/say Привет, милок`")
            return
        if self.bot.voice_client_for(ctx.guild.id) is None:
            await ctx.reply("Я не в голосовом канале — сначала `/join`.")
            return
        if self.bot.pipeline is not None:
            await self.bot.pipeline._queue.put(SpeechJob(ctx.guild.id, persona, text.strip()))
        await ctx.reply(f"🔊 {persona.title}: {text.strip()}")

    @commands.guild_only()
    @commands.hybrid_command(
        name="listen", aliases=["слушай"], description="Включить или выключить приём речи"
    )
    async def listen_cmd(self, ctx: commands.Context, mode: str = "") -> None:
        if self.bot.pipeline is None:
            return
        if mode.lower() in {"off", "выкл", "0"}:
            self.bot.pipeline.enabled = False
            await ctx.reply("Микрофон бота выключен (озвучка через `/say` работает).")
        elif mode.lower() in {"on", "вкл", "1"}:
            self.bot.pipeline.enabled = True
            await ctx.reply("Микрофон бота включён.")
        else:
            state = "включён" if self.bot.pipeline.enabled else "выключен"
            await ctx.reply(f"Сейчас: {state}. Используй `/listen on` или `/listen off`.")

    @commands.guild_only()
    @commands.hybrid_command(
        name="reset", aliases=["сброс"], description="Очистить память диалога"
    )
    async def reset_cmd(self, ctx: commands.Context) -> None:
        assert ctx.guild is not None
        persona = self.bot.active_persona(ctx.guild)
        if persona is not None and self.bot.pipeline is not None:
            self.bot.pipeline.llm.reset(f"{ctx.guild.id}:{persona.key}")
        await ctx.reply("Память диалога очищена.")

    # --------------------------------------------------------------- служебное

    @commands.guild_only()
    @commands.hybrid_command(
        name="status", aliases=["статус"], description="Показать состояние бота"
    )
    async def status_cmd(self, ctx: commands.Context) -> None:
        assert ctx.guild is not None
        pipeline = self.bot.pipeline
        if pipeline is None:
            await ctx.reply("Бот ещё запускается.")
            return

        persona = self.bot.active_persona(ctx.guild)
        vc = self.bot.voice_client_for(ctx.guild.id)
        stt_ms = pipeline.stt.average_latency() * 1000
        await ctx.reply(
            "\n".join(
                [
                    f"Персонаж: **{persona.title}**" if persona else "Персонаж: выключен",
                    f"Модель: `{self.bot.cfg.llm.model}`",
                    f"Whisper: {pipeline.stt.describe()}",
                    f"Вход: {self.bot.cfg.input.mode}",
                    f"В канале: {'да — ' + vc.channel.name if vc else 'нет'}",
                    f"Слушаю: {'да' if pipeline.enabled else 'нет'}",
                    f"Средний STT: {stt_ms:.0f} мс",
                ]
            )
        )

    @commands.hybrid_command(name="help", aliases=["помощь", "хелп"], description="Список команд")
    async def help_cmd(self, ctx: commands.Context) -> None:
        p = self.bot.cfg.discord.prefix
        await ctx.reply(
            "\n".join(
                [
                    "`/join` — зайти в твой голосовой канал и слушать",
                    "`/leave` — выйти",
                    "`/npc <имя>` — выбрать персонажа, `/npc list` — список, `/npc off` — выключить",
                    "`/say <текст>` — озвучить текст без микрофона",
                    "`/listen on|off` — включить/выключить приём речи",
                    "`/reset` — очистить память диалога",
                    "`/status` — состояние",
                    f"Те же команды работают с префиксом `{p}`",
                ]
            )
        )
