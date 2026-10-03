"""A cooperative satellite: registers one model, loads it once, then serves evictions.

usage: cooperative.py GPU_JSON MODEL BYTES [--hold S]
Reads VRAMBITER_SOCKET / VRAMBITER_NAME from the environment, as a managed child does. Writes
"ready" to stdout after its first lease, and exits cleanly on SIGTERM.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import vrambiter
from vrambiter.gpu import FakeGpu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu")
    ap.add_argument("model")
    ap.add_argument("nbytes", type=int)
    ap.add_argument("--hold", type=float, default=0.0)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    gpu = FakeGpu.attach(args.gpu)
    pid = os.getpid()
    arb = vrambiter.connect("unnamed", required=True, wait_s=10)
    model = arb.register(
        args.model,
        vram_peak=args.nbytes,
        load=lambda: gpu.allocate(pid, args.nbytes),
        unload=lambda: gpu.free(pid),
    )
    with model.lease():
        time.sleep(args.hold)
    print("ready", flush=True)
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
