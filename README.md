# DnD Voice Bot

Голосовой NPC для Discord. Сидит в голосовом канале, режет речь по паузам (`webrtcvad`), распознаёт через `faster-whisper`, прогоняет через LLM с характером из `.md`-файла и отвечает голосом обратно в канал (с дублированием реплик в текст).

Характеры персонажей лежат в `personas/`, общий лор кампании — в `world/campaign.md`.

## Что нужно

- Windows 10/11 и Python 3.11 или 3.12 (на 3.13 может не собраться `webrtcvad`).
- Токен Discord-бота с включённым **Message Content Intent** и правами `Connect`, `Speak`, `View Channels`, `Send Messages`.
- Локальная LLM (Ollama / LM Studio) или ключ от OpenAI-совместимого API.

## Запуск

1. Запусти `install.bat` — скрипт создаст `.venv`, поставит зависимости, при наличии NVIDIA подтянет CUDA, спросит роль машины и запишет `DISCORD_TOKEN` в `.env`.
2. Если крутишь через Ollama, скачай модель:
   ```bat
   pull_model.bat qwen2.5:3b
   ```
3. Запускай через `run.bat` (остановка — `STOP_BOT.bat`).
4. В Discord зайди в голосовой канал и напиши `/join` или `!join`.

Для ручной настройки скопируй `.env.example` в `.env`, впиши токен и настрой `config.yaml`.

## Режимы и конфиг

Всё основное правится в `config.yaml`:
- **Две машины** — можно оставить лёгкого бота у себя, а `stt_server.bat` + Ollama поднять на ПК с видеокартой (по локалке или Tailscale). Выбирается при установке или через `tools/configure.py`.
- **Текстовый вход (`input.mode: text`)** — читает реплики из чата и отвечает голосом, Whisper тогда не нужен.
- **Озвучка (`tts.engine`)** — по умолчанию локальный `silero`, для слабых CPU есть `piper`, онлайн-вариант — `edge`.
- **Облачная LLM** — ключ кладётся в `.env` как `LLM_API_KEY`, чтобы не светить его в `config.yaml`.

Основные команды в Discord: `/join` (`!join`), `!leave`, `!npc list`, `!npc <имя>`, `!npc off`, `!say <текст>`, `!listen on|off`, `!reset`, `!status`.

## Локальная отладка

Пообщаться с персонажем в консоли без Discord:

```bat
.venv\Scripts\python.exe tools\chat_repl.py barman
```

Юнит-тесты (Discord и GPU не нужны):

```bat
.venv\Scripts\python.exe tests\test_segmenter.py
.venv\Scripts\python.exe tests\test_text.py
```
