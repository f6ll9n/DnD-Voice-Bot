"""Загрузка config.yaml + промптов персонажей."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _resolve(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def _read_text(path_str: str | None) -> str:
    p = _resolve(path_str)
    if p is None or not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()


@dataclass
class DiscordCfg:
    prefix: str = "!"
    guild_id: int = 0
    post_to_chat: bool = True
    post_mode: str = "all"  # all — реплики игроков и NPC; replies — только ответы NPC
    # Автоподключение при старте: ID голосового канала (0 = выключено)
    auto_join_voice: int = 0
    # ID текстового канала для реплик и логов (0 = не задан)
    auto_join_text: int = 0


@dataclass
class LLMCfg:
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = "lm-studio"
    model: str = "qwen2.5-7b-instruct"
    temperature: float = 0.9
    max_tokens: int = 120
    history_turns: int = 12


@dataclass
class STTCfg:
    # local  — Whisper на этой машине
    # remote — Whisper на другой машине (tools/stt_server.py)
    engine: str = "local"
    remote_url: str = "http://127.0.0.1:8765"
    remote_token: str = ""
    model: str = "small"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    language: str = "ru"
    beam_size: int = 1
    initial_prompt: str = ""


@dataclass
class VADCfg:
    aggressiveness: int = 2
    start_ms: int = 140
    end_ms: int = 700
    min_speech_ms: int = 350
    max_utterance_s: float = 25.0


@dataclass
class InputCfg:
    """Откуда берутся реплики игроков."""

    mode: str = "voice"  # voice | text | both
    text_channel: int = 0  # 0 = канал, где вызвали !join
    text_requires_name: bool = False
    # Принимать реплики от других ботов (транскрайберов).
    # Пустой bot_whitelist = верить любому боту в этом канале.
    accept_bots: bool = False
    bot_whitelist: list[int] = field(default_factory=list)
    # Секунд после ответа NPC, пока он отвечает на продолжение диалога без повтора имени (0 = выкл)
    follow_up_s: float = 10.0


@dataclass
class TTSCfg:
    engine: str = "silero"  # silero — локально 48 кГц, piper — локально 22 кГц, edge — онлайн
    silero_model: str = "models/silero/v5_ru.pt"
    piper_model: str = "models/piper/ru_RU-ruslan-medium.onnx"


@dataclass
class PersonaCfg:
    key: str
    title: str
    file: str
    voice: str = "ru-RU-DmitryNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"
    addressing: str = "name"  # name | always | llm
    trigger_words: list[str] = field(default_factory=list)
    prompt: str = ""

    def __post_init__(self) -> None:
        if not self.prompt:
            self.prompt = _read_text(self.file)


@dataclass
class Config:
    discord: DiscordCfg
    llm: LLMCfg
    stt: STTCfg
    vad: VADCfg
    tts: TTSCfg
    personas: dict[str, PersonaCfg]
    input: InputCfg = field(default_factory=InputCfg)
    default_persona: str | None = None
    world_file: str | None = None
    world: str = ""
    interrupt_on_speech: bool = True
    post_to_chat: bool = True
    log_dir: str = "logs"


def _section(cls: type, data: dict[str, Any] | None):
    data = data or {}
    known = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    cfg_path = Path(path or os.getenv("BOT_CONFIG", ROOT / "config.yaml"))
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    personas: dict[str, PersonaCfg] = {}
    for key, pdata in (raw.get("personas") or {}).items():
        pdata = dict(pdata or {})
        personas[key] = PersonaCfg(
            key=key,
            title=pdata.get("title", key),
            file=pdata.get("file", f"personas/{key}.md"),
            voice=pdata.get("voice", "ru-RU-DmitryNeural"),
            rate=pdata.get("rate", "+0%"),
            pitch=pdata.get("pitch", "+0Hz"),
            addressing=pdata.get("addressing", "name"),
            trigger_words=list(pdata.get("trigger_words") or []),
        )

    default_persona = raw.get("default_persona")
    if default_persona and default_persona not in personas:
        raise SystemExit(f"default_persona '{default_persona}' отсутствует в personas")

    world_file = raw.get("world_file")

    # Секреты живут в .env, а не в config.yaml: конфиг уезжает в архив другу.
    llm_cfg = _section(LLMCfg, raw.get("llm"))
    llm_cfg.api_key = os.getenv("LLM_API_KEY", llm_cfg.api_key)
    llm_cfg.base_url = os.getenv("LLM_BASE_URL", llm_cfg.base_url)
    llm_cfg.model = os.getenv("LLM_MODEL", llm_cfg.model)

    return Config(
        discord=_section(DiscordCfg, raw.get("discord")),
        llm=llm_cfg,
        stt=_section(STTCfg, raw.get("stt")),
        vad=_section(VADCfg, raw.get("vad")),
        tts=_section(TTSCfg, raw.get("tts")),
        input=_section(InputCfg, raw.get("input")),
        personas=personas,
        default_persona=default_persona,
        world_file=world_file,
        world=_read_text(world_file),
        interrupt_on_speech=bool(raw.get("interrupt_on_speech", True)),
        post_to_chat=bool((raw.get("discord") or {}).get("post_to_chat", True)),
        log_dir=str(raw.get("log_dir", "logs")),
    )
