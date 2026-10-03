"""A randomised end-to-end run: several satellites, many concurrent leases, a small card.

Every model allocates its peak while loading and again while working (activations), on a FakeGpu
that raises a fake OOM the moment anything over-commits. If the arbiter ever admitted beyond what
fits (counting reservations, working headroom and loads in flight) a lease would raise here.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time

import pytest

from tests.helpers import make_daemon
from vrambiter.clock import LoopClock
from vrambiter.errors import VramUnavailable
from vrambiter.gpu import FakeGpu, FakeOutOfMemory
from vrambiter.units import GiB

WORKERS = 8
ROUNDS = 25


class Weights:
    def __init__(self, gpu: FakeGpu, pid: int, resident: int, peak: int) -> None:
        self.gpu, self.pid, self.resident, self.peak = gpu, pid, resident, peak
        self.held = 0
        # A model's peak is per model, not per lease: like a real server, it runs one job at a
        # time on it (concurrency inside a model is the satellite's business and its peak's).
        self.busy = threading.Lock()

    def load(self) -> None:
        self.gpu.allocate(self.pid, self.peak)  # staging buffers during the load
        time.sleep(0.002)
        self.gpu.free(self.pid, self.peak - self.resident)
        self.held = self.resident

    def unload(self) -> None:
        self.gpu.free(self.pid, self.held)
        self.held = 0

    def work(self) -> None:
        with self.busy:
            extra = self.peak - self.resident  # activations: the model grows to its peak
            self.gpu.allocate(self.pid, extra)
            time.sleep(0.001)
            self.gpu.free(self.pid, extra)


@pytest.mark.parametrize(
    ("seed", "vram", "slots", "priorities"),
    [
        (1234, "32GiB", 2, (0,)),
        (7, "24GiB", 1, (0,)),
        (2024, "20GiB", 3, (0,)),
        # Mixed priorities: a low-priority request may legitimately fail (it may not evict a
        # higher-priority idle model), but nothing may ever over-commit the card.
        (99, "20GiB", 2, (0, 0, 1)),
    ],
)
async def test_random_leases_never_overcommit_the_card(sock_path, seed, vram, slots, priorities):
    rng = random.Random(seed)
    # Mixed priorities can starve low-priority requests (by design: priority, then FIFO); keep
    # their timeouts short so starvation shows up as a quick VramUnavailable, not a slow test.
    lease_timeout = 30 if len(set(priorities)) == 1 else 2
    d = make_daemon(
        sock_path,
        vram=vram,
        clock=LoopClock(),
        max_concurrent_loads=slots,
        poll_interval_s=0.05,
        settle_poll_s=0.01,
        settle_timeout_s=1,
        refuse_cooldown_s=0.05,
    )
    await d.start()
    errors: list[BaseException] = []
    done = 0
    lock = threading.Lock()
    try:
        models = []
        for s in range(4):
            client = await d.client(f"sat{s}", pid=1000 + s)
            for m in range(3):
                resident = rng.randint(2, 7) * GiB
                peak = resident + rng.randint(0, 4) * GiB
                w = Weights(d.gpu, 1000 + s, resident, peak)
                model = await asyncio.to_thread(
                    client.register,
                    f"m{m}",
                    vram_peak=peak,
                    vram_resident=resident,
                    priority=rng.choice(priorities),
                    load=w.load,
                    unload=w.unload,
                    measure=lambda w=w: w.held,
                )
                models.append((model, w))

        def worker(seed: int) -> None:
            nonlocal done
            r = random.Random(seed)
            for _ in range(ROUNDS):
                model, w = r.choice(models)
                try:
                    with model.lease(timeout=lease_timeout):
                        w.work()
                except BaseException as exc:  # an OOM here is an arbiter bug
                    errors.append(exc)
                    continue
                with lock:
                    done += 1

        await asyncio.gather(*(asyncio.to_thread(worker, seed + i) for i in range(WORKERS)))
        oom = [e for e in errors if isinstance(e, FakeOutOfMemory)]
        assert oom == [], oom[:3]  # safety: never admit beyond what fits
        if len(set(priorities)) == 1:
            # Liveness: with equal priorities every model fits once others are evicted, so every
            # lease eventually succeeds.
            assert errors == [], errors[:3]
            assert done == WORKERS * ROUNDS
        else:
            assert all(isinstance(e, VramUnavailable) for e in errors), errors[:3]
    finally:
        for c in d.clients:
            c.close()
        await d.stop()
