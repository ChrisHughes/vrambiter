"""A fake ``llama-server`` in router mode, faithful to the endpoints vrambiter uses.

usage: llama_router.py GPU_JSON PORT CONFIG_JSON

CONFIG_JSON: {"models": {"<id>": <bytes>}, "load_delay": 0.2, "fail": ["<id>"]}

Each loaded model runs in its own child process (as the real router does), which allocates its
bytes on the shared FakeGpu under its own pid. ``POST /models/load`` returns immediately and the
status goes loading -> loaded, like the real thing.
"""

from __future__ import annotations

import http.server
import json
import signal
import subprocess
import sys
import threading
import time

CHILD = (
    "import os, signal, sys, time\n"
    "from vrambiter.gpu import FakeGpu\n"
    "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
    "time.sleep(float(sys.argv[3]))\n"
    "FakeGpu.attach(sys.argv[1]).allocate(os.getpid(), int(sys.argv[2]))\n"
    "print('up', flush=True)\n"
    "while True: time.sleep(0.05)\n"
)


class Router:
    def __init__(self, gpu: str, config: dict) -> None:
        self.gpu = gpu
        self.sizes: dict[str, int] = config["models"]
        self.load_delay = float(config.get("load_delay", 0.1))
        self.fail = set(config.get("fail", []))
        self.lock = threading.Lock()
        self.children: dict[str, subprocess.Popen] = {}
        self.status: dict[str, dict] = {m: {"value": "unloaded"} for m in self.sizes}
        self.requests: list[tuple[str, str]] = []

    def load(self, model: str) -> None:
        with self.lock:
            if self.status[model]["value"] in ("loaded", "loading"):
                return
            self.status[model] = {"value": "loading", "args": ["llama-server"]}
        if model in self.fail:

            def fail() -> None:
                time.sleep(self.load_delay)
                with self.lock:
                    self.status[model] = {"value": "unloaded", "failed": True, "exit_code": 1}

            threading.Thread(target=fail, daemon=True).start()
            return
        child = subprocess.Popen(
            [sys.executable, "-c", CHILD, self.gpu, str(self.sizes[model]), str(self.load_delay)],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.children[model] = child

        def wait_up() -> None:
            child.stdout.readline()
            with self.lock:
                if self.children.get(model) is child:
                    self.status[model] = {"value": "loaded", "args": ["llama-server"]}

        threading.Thread(target=wait_up, daemon=True).start()

    def unload(self, model: str) -> None:
        child = self.children.pop(model, None)
        if child is not None:
            child.terminate()
            child.wait()
        with self.lock:
            self.status[model] = {"value": "unloaded", "args": ["llama-server"]}

    def listing(self) -> dict:
        with self.lock:
            return {
                "data": [
                    {"id": m, "path": f"/models/{m}.gguf", "status": dict(s)}
                    for m, s in self.status.items()
                ]
            }


def main() -> None:
    gpu, port, config_json = sys.argv[1:4]
    router = Router(gpu, json.loads(config_json))
    signal.signal(
        signal.SIGTERM, lambda *_: ([router.unload(m) for m in list(router.children)], sys.exit(0))
    )

    class Handler(http.server.BaseHTTPRequestHandler):
        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            router.requests.append(("GET", self.path))
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path.startswith("/models"):
                self._json(200, router.listing())
            else:
                self._json(404, {"error": {"code": 404, "message": "not found"}})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            model = body.get("model")
            router.requests.append(("POST", f"{self.path} {model}"))
            if model not in router.sizes:
                self._json(
                    400,
                    {
                        "error": {
                            "code": 400,
                            "message": f"model {model} not found",
                            "type": "invalid_request_error",
                        }
                    },
                )
                return
            if self.path == "/models/load":
                router.load(model)
                self._json(200, {"success": True})
            elif self.path == "/models/unload":
                router.unload(model)
                self._json(200, {"success": True})
            else:
                self._json(404, {"error": {"code": 404, "message": "not found"}})

        def log_message(self, *a: object) -> None:
            pass

    http.server.ThreadingHTTPServer.allow_reuse_address = True
    server = http.server.ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
