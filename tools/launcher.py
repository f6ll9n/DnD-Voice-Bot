"""Подготовка среды перед запуском main.py (запускается из run.bat / DND Voice Bot.bat).

Делает запуск по-настоящему «в один клик»:
  1. Если LLM локальная (127.0.0.1/localhost) — запускает Ollama в фоне и проверяет модель.
     При наличии models/Modelfile и .gguf импортирует модель офлайн через `ollama create`.
  2. Если не задан DISCORD_TOKEN — открывает окно ввода токена.
  3. Проверяет доступ к Discord и активность обхода (zapret / winws / GoodbyeDPI).

Коды возврата:
  0 — всё готово к запуску main.py
  2 — критическая ошибка (не нужно перезапускать в цикле)
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.llm import is_local_url  # noqa: E402

DISCORD_GATEWAY_URL = "https://discord.com/api/v10/gateway"
_OLLAMA_PROC: subprocess.Popen | None = None
_BOT_PROC: subprocess.Popen | None = None
_TARGET_MODEL: str = ""
_CLEANED_UP = False
_CTRL_HANDLER_REF = None


def _direct_opener() -> urllib.request.OpenerDirector:
    """HTTP-клиент без системного прокси для локальных запросов к Ollama."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def find_ollama() -> Path | None:
    found = shutil.which("ollama")
    if found:
        return Path(found)
    for base in (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama",
        Path(os.environ.get("ProgramFiles", "")) / "Ollama",
    ):
        exe = base / "ollama.exe"
        if exe.is_file():
            return exe
    return None


def _ollama_origin(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc or "127.0.0.1:11434"
    return f"{scheme}://{netloc}"


def _ollama_models(origin: str, timeout: float = 3.0) -> list[str] | None:
    opener = _direct_opener()
    req = urllib.request.Request(f"{origin}/api/tags")
    try:
        with opener.open(req, timeout=timeout) as resp:
            data = json.loads(resp.read() or b"{}")
    except Exception:
        return None
    return [str(m.get("name", "")) for m in data.get("models", []) if m.get("name")]


def _has_model(models: list[str], target: str) -> bool:
    if target in models:
        return True
    if ":" not in target and f"{target}:latest" in models:
        return True
    return False


def ensure_ollama(base_url: str, model: str) -> bool:
    """Гарантирует запущенную локальную Ollama и наличие нужной модели."""
    global _OLLAMA_PROC, _TARGET_MODEL
    _TARGET_MODEL = model
    if not is_local_url(base_url):
        return True

    origin = _ollama_origin(base_url)
    models = _ollama_models(origin)

    if models is None:
        exe = find_ollama()
        if exe is None:
            print("[!] Ollama не найдена. Установи с https://ollama.com/download (бесплатно) или запусти install.bat.")
            return False

        print("[*] Запускаю локальную Ollama в фоне...")
        env = os.environ.copy()
        env["NO_PROXY"] = "127.0.0.1,localhost"
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        _OLLAMA_PROC = subprocess.Popen(
            [str(exe), "serve"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )

        for _ in range(30):
            time.sleep(0.5)
            models = _ollama_models(origin, timeout=2.0)
            if models is not None:
                break

        if models is None:
            print("[!] Не удалось запустить Ollama на", origin)
            return False
        print("[ok] Ollama запущена.")

    if _has_model(models, model):
        print(f"[ok] Локальная LLM готова: {model}")
        return True

    exe = find_ollama()
    if exe is None:
        print(f"[!] Модель '{model}' не найдена, а ollama.exe недоступна.")
        return False

    modelfile = ROOT / "models" / "Modelfile"
    if modelfile.is_file() and model == "qwen2.5:3b":
        gguf = ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
        if gguf.is_file():
            print(f"[*] Регистрирую локальную модель {model} из {gguf.name} (без интернета)...")
            rc = subprocess.call([str(exe), "create", model, "-f", str(modelfile)], cwd=modelfile.parent)
            if rc == 0:
                print(f"[ok] Модель {model} готова.")
                return True

    print(f"[*] Скачиваю модель {model} в Ollama (один раз, бесплатно)...")
    rc = subprocess.call([str(exe), "pull", model], cwd=ROOT)
    if rc != 0:
        print(f"[!] Не удалось получить модель '{model}'. Попробуй pull_model.bat.")
        return False
    print(f"[ok] Модель {model} готова.")
    return True


def ensure_token() -> bool:
    """Если токен Discord ещё не настроен — сразу просит его через окно ввода."""
    from tools import set_token

    token = set_token.read_token()
    if token:
        return True

    print("[!] DISCORD_TOKEN не задан — открываю окно ввода токена...")
    return set_token.main() == 0


def _running_bypass_processes() -> list[str]:
    """Проверяет запущенные процессы обхода DPI (winws / goodbyedpi)."""
    if os.name != "nt":
        return []
    try:
        out = subprocess.check_output(
            ["tasklist", "/FO", "CSV", "/NH"],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except Exception:
        return []

    found: list[str] = []
    low = out.lower()
    for name in ("winws.exe", "winws2.exe", "goodbyedpi.exe"):
        if f'"{name}"' in low and name not in found:
            found.append(name)
    return found


def find_zapret_dirs() -> list[Path]:
    """Ищет папки zapret / GoodbyeDPI в Загрузках и на Рабочем столе."""
    home = Path.home()
    candidates: list[Path] = []
    for base in (home / "Downloads", home / "Desktop"):
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            low = entry.name.lower()
            if "zapret" in low or "goodbyedpi" in low:
                nested = entry / entry.name
                candidates.append(nested if nested.is_dir() else entry)
    return candidates


def _discord_reachable(timeout: float = 6.0) -> tuple[bool, str]:
    req = urllib.request.Request(DISCORD_GATEWAY_URL, headers={"User-Agent": "DnDVoiceBot/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (200 <= resp.status < 400), f"HTTP {resp.status}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_discord_and_bypass() -> bool:
    """Проверяет связь с Discord и предупреждает, если не запущен zapret/GoodbyeDPI."""
    while True:
        ok, detail = _discord_reachable()
        bypass = _running_bypass_processes()

        if ok:
            if bypass:
                print(f"[ok] Discord доступен (активен обход: {', '.join(bypass)})")
            else:
                print("[ok] Discord API доступен.")
                print(
                    "[warn] Процесс zapret (winws.exe) не обнаружен.\n"
                    "       Если бот заходит в чат, но НЕ слышит/не говорит в голосовом канале,\n"
                    "       включи zapret для Discord (нужен обход UDP-трафика голоса)."
                )
            return True

        print(f"\n[!] Нет доступа к Discord ({detail}).")
        print("    Без обхода блокировки (zapret / GoodbyeDPI) бот не подключится к Discord.")
        dirs = find_zapret_dirs()
        if dirs:
            print("    Найдены папки обхода на компьютере:")
            for d in dirs:
                print(f"      - {d}")

        if not sys.stdin.isatty():
            return False

        choice = input("\nВключи zapret / GoodbyeDPI и нажми Enter для повторной проверки (или 'q' для выхода, 'o' открыть папку): ").strip().lower()
        if choice in {"q", "quit", "exit", "в", "выход"}:
            return False
        if choice in {"o", "open", "о"} and dirs:
            os.startfile(str(dirs[0]))  # type: ignore[attr-defined]


def cleanup_processes() -> None:
    """Выгружает модель и останавливает процессы, запущенные лаунчером."""
    global _CLEANED_UP, _BOT_PROC, _OLLAMA_PROC
    if _CLEANED_UP:
        return
    _CLEANED_UP = True

    if _BOT_PROC is not None and _BOT_PROC.poll() is None:
        try:
            _BOT_PROC.terminate()
            _BOT_PROC.wait(timeout=3)
        except Exception:
            try:
                _BOT_PROC.kill()
            except Exception:
                pass

    if _OLLAMA_PROC is not None:
        exe = find_ollama()
        if exe is not None and _TARGET_MODEL:
            try:
                subprocess.run(
                    [str(exe), "stop", _TARGET_MODEL],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
            except Exception:
                pass
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(_OLLAMA_PROC.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        else:
            _OLLAMA_PROC.terminate()


def _install_console_cleanup() -> None:
    """На Windows ловит даже закрытие окна крестиком (CTRL_CLOSE_EVENT), чтобы выгрузить VRAM."""
    atexit.register(cleanup_processes)
    if os.name != "nt":
        return
    global _CTRL_HANDLER_REF
    handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

    def _on_ctrl(_event: int) -> bool:
        cleanup_processes()
        return False

    _CTRL_HANDLER_REF = handler_type(_on_ctrl)
    try:
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_CTRL_HANDLER_REF, True)
    except Exception:
        pass


def run_bot_loop() -> int:
    """Запускает main.py с автоперезапуском при временных сбоях (до 5 раз)."""
    global _BOT_PROC
    failures = 0
    while True:
        _BOT_PROC = subprocess.Popen([sys.executable, str(ROOT / "main.py")], cwd=ROOT)
        rc = _BOT_PROC.wait()
        _BOT_PROC = None
        if rc == 0:
            print("Бот остановлен.")
            return 0
        if rc == 2:
            print("\n[!] Критическая ошибка конфигурации или токена. Перезапуск отменён.")
            return 2

        failures += 1
        if failures >= 5:
            print("\n[!] Бот упал 5 раз подряд. Проверь текст ошибки выше.")
            return 1

        print(f"\n[!] Бот остановился (код {rc}). Перезапуск {failures}/5 через 10 секунд...")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nОстановлено пользователем.")
            return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    run_mode = "--run" in sys.argv[1:]
    if run_mode:
        _install_console_cleanup()

    print("=== DnD Voice Bot: проверка перед стартом ===")
    cfg = load_config()

    if not ensure_ollama(cfg.llm.base_url, cfg.llm.model):
        return 2

    if not ensure_token():
        return 2

    if not check_discord_and_bypass():
        return 2

    if not run_mode:
        print("=== Запускаю бота ===\n")
        return 0

    print("=== Бот запускается ===\n")
    try:
        return run_bot_loop()
    finally:
        cleanup_processes()


if __name__ == "__main__":
    raise SystemExit(main())
