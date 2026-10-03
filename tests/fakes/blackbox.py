"""A black-box "model server": holds GPU memory on a shared FakeGpu until it is stopped.

usage: blackbox.py GPU_JSON BYTES [--ignore-term] [--health-port N] [--crash-after S]
"""

from __future__ import annotations

import argparse
import http.server
import os
import signal
import sys
import threading
import time

from vrambiter.gpu import FakeGpu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu")
    ap.add_argument("nbytes", type=int)
    ap.add_argument("--ignore-term", action="store_true")
    ap.add_argument("--health-port", type=int)
    ap.add_argument("--crash-after", type=float)
    args = ap.parse_args()

    if args.ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    FakeGpu.attach(args.gpu).allocate(os.getpid(), args.nbytes)

    if args.health_port:

        class Health(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *a: object) -> None:
                pass

        http.server.HTTPServer.allow_reuse_address = True
        server = http.server.HTTPServer(("127.0.0.1", args.health_port), Health)
        threading.Thread(target=server.serve_forever, daemon=True).start()

    start = time.monotonic()
    while True:
        time.sleep(0.05)
        if args.crash_after is not None and time.monotonic() - start > args.crash_after:
            os._exit(3)


if __name__ == "__main__":
    main()
