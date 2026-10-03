"""A minimal cooperative PyTorch satellite: one model, one HTTP endpoint, a lease per request.

    vrambiter daemon &                                  # or let the daemon start it (see README)
    vrambiter run --name toy -- python examples/torch_satellite.py --port 8101
    curl -s localhost:8101/infer -d '{"n": 4}'

Without a daemon it runs standalone: the model loads on the first request and stays resident.
With one, it loads when admitted, is busy for exactly the duration of each request, and is
unloaded (with the CUDA cache emptied) when another satellite needs the room.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

import vrambiter
from vrambiter.integrations.torch import unload as drop

state: dict[str, torch.nn.Module | None] = {"net": None}
state_lock = threading.Lock()


def load_weights() -> None:
    """Called by vrambiter when the model must become resident (under its per-model lock)."""
    net = torch.nn.Sequential(*[torch.nn.Linear(4096, 4096) for _ in range(32)])  # ~2 GiB fp32
    state["net"] = net.to("cuda").eval()


def free_weights() -> None:
    """Called on eviction. Drop every reference; vrambiter then runs gc + empty_cache."""
    drop(state, "net")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8101)
    args = ap.parse_args()

    arb = vrambiter.connect("toy")  # NullArbiter (standalone) when no daemon is running
    model = arb.register(
        "mlp",
        vram_peak="3GiB",  # weights plus activations: the most it ever needs
        vram_resident="2GiB",  # what it holds while idle (measured anyway)
        host_peak="3GiB",  # host RAM while loading (fp32 tensors are built on the CPU first)
        load=load_weights,
        unload=free_weights,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            try:
                # Admission, loading and "busy" all happen here. Inside the block the model is
                # resident and will not be evicted.
                with model.lease(timeout=120), torch.inference_mode():
                    x = torch.randn(int(body.get("n", 1)), 4096, device="cuda")
                    y = state["net"](x).sum().item()
                reply, code = {"result": y}, 200
            except vrambiter.VramUnavailable as exc:  # says who holds the memory
                reply, code = {"error": str(exc)}, 503
            data = json.dumps(reply).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"toy satellite on :{args.port} ({'coordinated' if arb.connected else 'standalone'})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
