"""LLM через любой OpenAI-совместимый сервер (LM Studio, Ollama, vLLM)."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
from collections import defaultdict, deque
from urllib.parse import urlparse

from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from .config import LLMCfg, PersonaCfg
from .text import clean_for_speech

log = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def is_local_url(url: str) -> bool:
    """True для localhost/loopback, чтобы системный прокси не ломал локальную LLM."""
    try:
        host = (urlparse(url).hostname or "").strip("[]").lower()
    except ValueError:
        return False
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

COMMON_RULES = """\
Ты — голосовой NPC в настольной ролевой игре (D&D) в Discord.
Твой ответ сразу озвучивается синтезом речи, поэтому пиши только живую устную речь своего персонажа.

Жёсткие правила:
- Отвечай СТРОГО от первого лица своего персонажа и ТОЛЬКО на последнюю реплику собеседника.
- Никогда не пиши реплики или действия за игрока (не говори «вот деньги», если игрок их ещё не дал).
- Только прямая речь: никаких пояснений, никаких «как ИИ», никаких ремарок в *звёздочках* и (скобках).
- Длина ответа: 1–2 коротких, естественных предложения на русском языке.
- Опирайся на историю диалога: помни, о чём уже договорились и заплатил ли игрок.
- Не повторяй дословно свои прошлые фразы.
"""


def strip_speaker_prefix(text: str, persona: PersonaCfg, speaker: str) -> str:
    """Срезает случайный префикс вида 'Хома:' или 'Бармен:', если модель его дописала."""
    prefixes = [persona.title, persona.title.split(",", 1)[0].strip(), *persona.trigger_words]
    cleaned = text.strip()
    for pref in prefixes:
        p = pref.strip()
        if not p:
            continue
        pattern = re.compile(rf"^(?:{re.escape(p)})\s*:\s*", re.IGNORECASE)
        updated = pattern.sub("", cleaned, count=1).strip()
        if updated != cleaned:
            cleaned = updated
            break
    # Если модель случайно ответила от лица игрока — отбрасываем
    if re.match(rf"^(?:игрок|{re.escape(speaker)})\s*:", cleaned, re.IGNORECASE):
        return ""
    return cleaned


def is_low_quality_reply(answer: str, last_reply: str = "") -> bool:
    """Отлавливает явные срывы 3B-модели: иероглифы, пустоту, точный повтор."""
    if not answer or not any(ch.isalpha() for ch in answer):
        return True
    if _CJK_RE.search(answer):
        return True
    if last_reply and answer.strip().lower() == last_reply.strip().lower():
        return True
    return False


class LLM:
    def __init__(self, cfg: LLMCfg) -> None:
        self.cfg = cfg
        self._http_client: DefaultAsyncHttpxClient | None = None
        if is_local_url(cfg.base_url):
            # Локальная Ollama/LM Studio не должна идти через системный HTTP-прокси Windows.
            self._http_client = DefaultAsyncHttpxClient(trust_env=False, timeout=90.0)
        self.client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=90.0,
            http_client=self._http_client,
        )
        self._history: dict[str, deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=max(2, cfg.history_turns * 2))
        )
        self._memory: dict[str, str] = {}
        self._turns_since_summary: dict[str, int] = defaultdict(int)
        self._bg_tasks: set[asyncio.Task] = set()

    async def aclose(self) -> None:
        for task in list(self._bg_tasks):
            task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        self._bg_tasks.clear()
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    # ------------------------------------------------------------------ prompt

    def _system(self, persona: PersonaCfg, world: str, memory: str = "") -> str:
        parts = [persona.prompt.strip(), COMMON_RULES.strip()]
        if world.strip():
            parts.append("КОНТЕКСТ МИРА (используй, но не пересказывай):\n" + world.strip())
        if memory.strip():
            parts.append("ПАМЯТЬ ТЕКУЩЕЙ СЦЕНЫ (уже случилось в этом разговоре):\n" + memory.strip())
        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------- calls

    async def _generate_once(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
    ) -> str:
        kwargs: dict = {}
        if is_local_url(self.cfg.base_url):
            kwargs["extra_body"] = {"options": {"num_ctx": 4096}}
        resp = await self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            top_p=0.9,
            max_tokens=self.cfg.max_tokens,
            presence_penalty=0.1,
            frequency_penalty=0.15,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()

    async def reply(
        self,
        persona: PersonaCfg,
        world: str,
        speaker: str,
        text: str,
        history_key: str,
    ) -> str:
        history = self._history[history_key]
        memory = self._memory.get(history_key, "")
        user_msg = f"Игрок {speaker} говорит тебе: {text}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system(persona, world, memory)},
            *history,
            {"role": "user", "content": user_msg},
        ]

        last_assistant = next(
            (m["content"] for m in reversed(history) if m.get("role") == "assistant"),
            "",
        )

        answer = ""
        for attempt in range(2):
            temp = self.cfg.temperature if attempt == 0 else min(0.85, self.cfg.temperature + 0.18)
            raw = await self._generate_once(messages, temperature=temp)
            candidate = clean_for_speech(raw)
            candidate = strip_speaker_prefix(candidate, persona, speaker)
            if not is_low_quality_reply(candidate, last_assistant):
                answer = candidate
                break
            if candidate and not answer and not _CJK_RE.search(candidate):
                answer = candidate

        if not answer:
            return ""

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": answer})

        self._turns_since_summary[history_key] += 1
        if self._turns_since_summary[history_key] >= 6 or len(history) >= (history.maxlen or 16):
            self._turns_since_summary[history_key] = 0
            snapshot = list(history)[-10:]
            self._schedule_summary(history_key, persona, memory, snapshot)

        return answer

    async def warmup(self) -> None:
        """Прогревает модель с тем же num_ctx=4096, чтобы не перегружать её на первом ответе."""
        kwargs: dict = {}
        if is_local_url(self.cfg.base_url):
            kwargs["extra_body"] = {"options": {"num_ctx": 4096}}
        await self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1,
            temperature=0.0,
            **kwargs,
        )

    def _schedule_summary(
        self,
        history_key: str,
        persona: PersonaCfg,
        prev_memory: str,
        recent_messages: list[dict[str, str]],
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(
            self._refresh_memory(history_key, persona, prev_memory, recent_messages),
            name=f"llm-memory-{history_key}",
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _refresh_memory(
        self,
        history_key: str,
        persona: PersonaCfg,
        prev_memory: str,
        recent_messages: list[dict[str, str]],
    ) -> None:
        """В фоне сжимает ключевые факты диалога (кто заплатил, что заказали, о чём договорились)."""
        dialogue = "\n".join(
            f"{'NPC' if m['role'] == 'assistant' else 'Диалог'}: {m['content']}"
            for m in recent_messages
        )
        prompt = (
            f"Персонаж: {persona.title}.\n"
            "Обнови краткую память сцены (до 4 коротких пунктов, максимум 55 слов на русском).\n"
            "Фиксируй ТОЛЬКО важные факты: кто из игроков что заказал, кто уже заплатил (и сколько), "
            "кто нагрубил, о каких сделках/слухах договорились.\n"
            "Без воды и художественных описаний.\n\n"
            f"Старая память:\n{prev_memory or 'пусто'}\n\n"
            f"Последние реплики:\n{dialogue}\n\n"
            "Новая краткая память:"
        )
        try:
            kwargs: dict = {}
            if is_local_url(self.cfg.base_url):
                kwargs["extra_body"] = {"options": {"num_ctx": 4096}}
            resp = await self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=80,
                **kwargs,
            )
            summary = (resp.choices[0].message.content or "").strip()
            if summary and not _CJK_RE.search(summary):
                self._memory[history_key] = summary[:500]
                log.debug("Память сцены (%s): %s", history_key, self._memory[history_key])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("Не удалось обновить память сцены: %s", exc)

    async def is_addressed(self, persona: PersonaCfg, text: str) -> bool:
        """Дешёвый классификатор: обращаются ли к этому персонажу."""
        prompt = (
            f"Персонаж в голосовом чате: {persona.title}.\n"
            f"Реплика игрока: «{text}»\n\n"
            "Обращается ли игрок к этому персонажу напрямую — по имени или явно к нему? "
            "Учти, что игроки могут обсуждать игру между собой и говорить о персонаже в третьем лице.\n"
            "Ответь одним словом: ДА или НЕТ."
        )
        resp = await self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=6,
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        return ans.startswith("да") or ans.startswith("yes")

    async def ping(self) -> list[str]:
        """Проверка доступности сервера. Возвращает список имён моделей."""
        models = await self.client.models.list()
        return [m.id for m in models.data]

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._history.clear()
            self._memory.clear()
            self._turns_since_summary.clear()
        else:
            self._history.pop(key, None)
            self._memory.pop(key, None)
            self._turns_since_summary.pop(key, None)
