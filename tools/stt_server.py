"""Мини-сервер распознавания речи.

Нужен, когда бот живёт на одном ПК, а видеокарта — на другом.
Запускается на машине с видеокартой, бот обращается к нему по сети.

    stt_server.bat
    .venv\\Scripts\\python.exe tools\\stt_server.py --host 0.0.0.0 --port 8765

Эндпоинты:
    GET  /health      -> {"ok": true, "model": "small / cuda (int8_float16)"}
    POST /transcribe  -> тело: сырой PCM s16le mono 48 кГц; ответ {"text": "..."}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.stt import Transcriber  # noqa: E402

log = logging.getLogger("stt_server")

MODEL: Transcriber | None = None
TOKEN = ""
MAX_BODY = 32 * 1024 * 1024  # 32 МБ — с запасом на минуту речи


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self) -> bool:
        if not TOKEN:
            return True
        return self.headers.get("X-Token") == TOKEN

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._json(404, {"error": "not found"})
            return
        self._json(200, {"ok": True, "model": MODEL.describe() if MODEL else "?"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/transcribe":
            self._json(404, {"error": "not found"})
            return
        if not self._auth_ok():
            self._json(401, {"error": "bad token"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._json(400, {"error": "bad content-length"})
            return
        if length <= 0 or length > MAX_BODY:
            self._json(400, {"error": "empty or too big body"})
            return

        body = self.rfile.read(length)
        if len(body) % 2:
            body = body[:-1]
        if not body:
            self._json(400, {"error": "empty body"})
            return

        pcm = np.frombuffer(body, dtype=np.int16)
        try:
            text = MODEL.transcribe(pcm) if MODEL else ""
        except Exception as exc:
            log.exception("Ошибка распознавания")
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return

        self._json(200, {"text": text})

    def log_message(self, fmt: str, *args) -> None:
        log.debug("%s — %s", self.address_string(), fmt % args)


def main() -> int:
    global MODEL, TOKEN

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="STT-сервер для DnD Voice Bot")
    parser.add_argument("--host", default="0.0.0.0", help="адрес прослушивания")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default="", help="пароль (по умолчанию из config.yaml)")
    args = parser.parse_args()

    cfg = load_config()
    TOKEN = args.token or cfg.stt.remote_token

    stt_cfg = cfg.stt
    stt_cfg.engine = "local"  # сервер всегда считает сам
    MODEL = Transcriber(stt_cfg)
    MODEL.load()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("STT-сервер слушает: http://%s:%d/transcribe", args.host, args.port)
    log.info("Проверка: http://%s:%d/health", args.host, args.port)
    if not TOKEN:
        log.warning("Токен не задан — сервер открыт всем в локальной сети.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Остановлено")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
