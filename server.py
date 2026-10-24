#!/usr/bin/env python3
"""Small same-origin web chat proxy for the SmolLM llama.cpp server."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = (
    "You are SmolLM, a concise and helpful assistant. Reply in the same language "
    "as the user. Do not repeat the user's question. If you do not know, say so."
)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    upstream = "http://127.0.0.1:8080"

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/health":
            try:
                with urllib.request.urlopen(self.upstream + "/health", timeout=4) as response:
                    data = json.loads(response.read().decode("utf-8"))
                self.send_json(200, {"ok": data.get("status") == "ok", "model": "SmolLM2-135M-Instruct"})
            except Exception as exc:  # upstream may be offline or the speaker may have rebooted
                self.send_json(503, {"ok": False, "error": f"Станция не отвечает: {exc}"})
            return

        if path in ("/", "/index.html"):
            body = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 256_000:
                self.send_json(413, {"error": "Слишком большой запрос"})
                return
            incoming = json.loads(self.rfile.read(length))
            messages = incoming.get("messages")
            if not isinstance(messages, list) or not messages:
                self.send_json(400, {"error": "Нет сообщений для модели"})
                return

            clean_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for item in messages[-8:]:
                role = item.get("role")
                content = item.get("content")
                if role not in ("user", "assistant") or not isinstance(content, str):
                    self.send_json(400, {"error": "Некорректный формат истории чата"})
                    return
                clean_messages.append({"role": role, "content": content[:8000]})

            request_body = json.dumps(
                {
                    "model": "smollm2-135m",
                    "messages": clean_messages,
                    "max_tokens": 96,
                    "temperature": 0.2,
                    "stream": True,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                self.upstream + "/v1/chat/completions",
                data=request_body,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                method="POST",
            )
            upstream_response = urllib.request.urlopen(request, timeout=180)
        except urllib.error.HTTPError as exc:
            self.send_json(exc.code, {"error": exc.read().decode("utf-8", "replace")[:4000]})
            return
        except Exception as exc:
            self.send_json(502, {"error": f"Не удалось обратиться к модели: {exc}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = upstream_response.read1(2048)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # The browser stopped generation or navigated away.
        finally:
            upstream_response.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1", help="listen address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--upstream", default="http://127.0.0.1:8080", help="Station llama-server URL")
    args = parser.parse_args()
    Handler.upstream = args.upstream.rstrip("/")
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"SmolLM chat: http://{args.bind}:{args.port} → {Handler.upstream}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping SmolLM chat", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
