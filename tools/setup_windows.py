"""Установщик «одна кнопка» для Windows. Запускается из install.bat.

Делает всё сам:
  1. проверяет Python
  2. создаёт .venv и ставит зависимости
  3. ставит CUDA-библиотеки, только если есть NVIDIA
  4. ставит Ollama (если её нет) и скачивает модель
  5. спрашивает токен Discord и пишет его в .env
  6. прописывает автозапуск при входе в Windows
  7. прогоняет самопроверку и предлагает запустить бота
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
VENV_PY = VENV / "Scripts" / "python.exe"
ENV_FILE = ROOT / ".env"
DEFAULT_MODEL = "qwen2.5:3b"
OLLAMA_URL = "https://ollama.com/download/OllamaSetup.exe"

BAR = "=" * 64


def hr(title: str = "") -> None:
    print(f"\n{BAR}\n  {title}\n{BAR}\n" if title else f"\n{BAR}\n")


def run(cmd: list[str]) -> int:
    print(">", " ".join(Path(c).name if i == 0 else str(c) for i, c in enumerate(cmd)))
    return subprocess.call(cmd, cwd=ROOT)


# ------------------------------------------------------------------ проверки


def has_nvidia() -> bool:
    """Есть ли NVIDIA, которой можно отдать Whisper."""
    if os.name != "nt":
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, timeout=20, check=False
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def ollama_exe() -> Path | None:
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


# ------------------------------------------------------------------- шаги


def install_ollama() -> bool:
    if ollama_exe() is not None:
        print("Ollama уже установлена.")
        return True

    print("Ollama не найдена — устанавливаю (бесплатно, ~200 МБ).")

    if shutil.which("winget"):
        print("\nПробую через winget...")
        rc = subprocess.call(
            [
                "winget",
                "install",
                "--id",
                "Ollama.Ollama",
                "-e",
                "--silent",
                "--accept-source-agreements",
                "--accept-package-agreements",
            ]
        )
        if rc == 0:
            for _ in range(30):
                if ollama_exe() is not None:
                    print("Ollama установлена.")
                    return True
                time.sleep(2)

    dest = Path(tempfile.gettempdir()) / "OllamaSetup.exe"
    print("\nСкачиваю установщик Ollama...")
    try:
        urllib.request.urlretrieve(OLLAMA_URL, dest)
    except Exception as exc:
        print(f"\n[!] Не удалось скачать автоматически: {exc}")
        print(f"    Скачай вручную и установи: {OLLAMA_URL}")
        return False

    print("\nСейчас откроется установщик Ollama. Нажми Install и дождись конца.")
    subprocess.call([str(dest)])
    for _ in range(120):
        if ollama_exe() is not None:
            print("Ollama установлена.")
            return True
        time.sleep(2)

    print("\n[!] Ollama так и не появилась. Перезапусти install.bat после установки.")
    return False


def pull_model(model: str) -> bool:
    exe = ollama_exe()
    if exe is None:
        return False

    print(f"\nСкачиваю модель {model} (~2 ГБ, только один раз)...")
    rc = subprocess.call([str(exe), "pull", model], cwd=ROOT)
    if rc == 0:
        return True

    print("\nСервер Ollama не отвечает — запускаю его и пробую снова.")
    subprocess.Popen(
        [str(exe), "serve"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(6)
    rc = subprocess.call([str(exe), "pull", model], cwd=ROOT)
    return rc == 0


# ------------------------------------------------------------------- токен


def read_token() -> str:
    if not ENV_FILE.exists():
        return ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("DISCORD_TOKEN="):
            value = line.split("=", 1)[1].strip()
            if value and not value.lower().startswith("вставь"):
                return value
    return ""


def write_token(token: str) -> None:
    text = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""
    if not text:
        example = ROOT / ".env.example"
        text = example.read_text(encoding="utf-8") if example.exists() else ""

    lines, replaced = [], False
    for line in text.splitlines():
        if line.startswith("DISCORD_TOKEN="):
            lines.append(f"DISCORD_TOKEN={token}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.append(f"DISCORD_TOKEN={token}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ask_token() -> str | None:
    token = read_token()
    if token:
        print(f"Токен Discord уже сохранён ({token[:8]}...).")
        return token

    hr("Токен бота Discord")
    print("Вставь токен из discord.com/developers -> твоё приложение -> Bot -> Reset Token\n")

    token = ""
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        token = (
            simpledialog.askstring(
                "DnD Voice Bot",
                "Вставь токен бота Discord:",
                show="✱",
            )
            or ""
        )
        root.destroy()
    except Exception:
        token = ""

    if not token.strip():
        try:
            token = input("Токен: ")
        except EOFError:
            token = ""

    token = token.strip()
    if not token:
        print("\n[!] Токен не введён. Пропиши его позже в .env вручную.")
        return None

    write_token(token)
    print("Токен сохранён в .env")
    return token


# ------------------------------------------------------------ автозапуск


def make_autostart() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    startup = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    if not startup.is_dir():
        return None

    lnk = startup / "DnD Voice Bot.lnk"
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{}');"
        "$s.TargetPath = '{}';"
        "$s.WorkingDirectory = '{}';"
        "$s.WindowStyle = 7;"
        "$s.Save()"
    ).format(lnk, ROOT / "run.bat", ROOT)

    rc = subprocess.call(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return lnk if rc == 0 else None


# --------------------------------------------------------------------- main


def ask_role() -> tuple[str, str]:
    print("""
Что ставим на эту машину?

  1) Бота   — Discord, приём голоса, озвучка. Лёгкая часть, без видеокарты.
  2) Мозги  — Whisper + LLM. Нужна видеокарта. (ПК друга)
  3) Всё сразу — одна машина на всё.
""")
    choice = input("Выбор [1/2/3]: ").strip() or "1"
    role = {"1": "bot", "2": "brain", "3": "all"}.get(choice, "bot")

    brain = ""
    if role == "bot":
        print("\nАдрес машины с видеокартой — IP в локальной сети или Tailscale.")
        print("Узнать на той машине: ipconfig -> IPv4, либо Tailscale IP.")
        while not brain:
            brain = input("IP: ").strip()
    return role, brain


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    hr("DnD Voice Bot — установка")

    major, minor = sys.version_info[:2]
    if not ((3, 10) <= (major, minor) <= (3, 12)):
        print(f"Нужен Python 3.10–3.12, а запущен {major}.{minor}.")
        print("Скачай 3.12 с python.org и поставь галочку 'Add python.exe to PATH'.")
        return 1
    print(f"Python {major}.{minor} — подходит.")

    role, brain = ask_role()

    if VENV_PY.exists():
        print(".venv уже есть — обновляю зависимости.")
    else:
        print("\nСоздаю изолированное окружение .venv...")
        if run([sys.executable, "-m", "venv", str(VENV)]) != 0:
            print("Не удалось создать окружение.")
            return 1

    hr("Зависимости (5–10 минут)")
    run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "wheel", "-q"])
    if run([str(VENV_PY), "-m", "pip", "install", "-r", "requirements.txt"]) != 0:
        print("\n[!] Зависимости не встали. Скопируй вывод выше.")
        return 1

    if role == "bot":
        print("\nРоль «бот»: CUDA и Ollama не нужны — они на машине с видеокартой.")
    else:
        if has_nvidia():
            hr("CUDA-библиотеки для Whisper на видеокарте")
            print("~1.2 ГБ. Если сорвётся — Whisper пойдёт на CPU.\n")
            if run([str(VENV_PY), "-m", "pip", "install", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"]) != 0:
                print("\n[!] CUDA не встала — не критично.")
        else:
            print("\nNVIDIA не найдена — CUDA не нужна. Whisper будет на CPU.")

        hr("Локальная модель")
        if install_ollama():
            if pull_model(DEFAULT_MODEL):
                print(f"Модель {DEFAULT_MODEL} готова.")
            else:
                print("\n[!] Модель не скачалась. Потом: pull_model.bat")
        else:
            print("\n[!] Ollama не установлена — бот не сможет отвечать. Поставь с ollama.com")

    if role in {"bot", "all"}:
        ask_token()
    else:
        print("\nРоль «мозги»: Discord-токен не нужен.")

    hr("Настройка config.yaml")
    cfg_cmd = [str(VENV_PY), str(ROOT / "tools" / "configure.py"), "--role", role]
    if role == "bot":
        cfg_cmd += ["--brain", brain]
    run(cfg_cmd)

    if role in {"bot", "all"}:
        hr("Автозапуск")
        lnk = make_autostart()
        if lnk:
            print(f"Бот будет запускаться при входе в Windows:\n  {lnk}")
            print("Удалить — убери этот ярлык из папки «Автозагрузка».")
        else:
            print("Не удалось создать автозапуск — запускай run.bat вручную.")

    if not ENV_FILE.exists():
        ENV_FILE.write_text((ROOT / ".env.example").read_text(encoding="utf-8"), encoding="utf-8")

    hr("Самопроверка")
    rc = run([str(VENV_PY), str(ROOT / "tools" / "preflight.py")])

    hr("Готово")
    if role in {"bot", "all"}:
        print("Дальше:")
        print("  1. В Discord: правой кнопкой по каналу -> Копировать ID")
        print("  2. Вписать в config.yaml -> discord.auto_join_voice / auto_join_text")
        print("  3. Запустить run.bat (или перезайти в Windows — запустится сам)\n")
    else:
        print("Дальше на этой машине:")
        print("  держи открытым окно stt_server.bat, пока играете\n")

    if rc == 0:
        question = (
            "Запустить бота сейчас? [Y/n] "
            if role in {"bot", "all"}
            else "Запустить STT-сервер сейчас? [Y/n] "
        )
        try:
            answer = input(question).strip().lower()
        except EOFError:
            answer = "n"
        if answer in {"", "y", "yes", "д", "да"}:
            target = ROOT / ("run.bat" if role in {"bot", "all"} else "stt_server.bat")
            subprocess.Popen([str(target)], cwd=ROOT, shell=True)
            print("Запускаю в отдельном окне...")
            return 0

    if rc != 0:
        print("\n[!] Проверка нашла проблемы — см. список выше.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
