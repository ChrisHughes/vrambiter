"""Shared test scaffolding: an in-process daemon on a real Unix socket, and polling helpers."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vrambiter.arbiter import Arbiter, ArbiterSettings
from vrambiter.client import ArbiterClient
from vrambiter.clock import Clock, FakeClock
from vrambiter.gpu import FakeGpu
from vrambiter.host import FakeHost
from vrambiter.server import Server
from vrambiter.units import GiB


def short_tmpdir() -> str:
    """A temp dir short enough for a Unix socket path (macOS caps them at ~104 bytes)."""
    return tempfile.mkdtemp(dir="/tmp", prefix="vb")


def remove_tree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


async def wait_for(cond: Callable[[], Any], timeout: float = 5.0, interval: float = 0.01) -> Any:
    """Poll ``cond`` (in real time) until it returns something truthy."""
    deadline = time.monotonic() + timeout
    while True:
        value = cond()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout}s: {cond}")
        await asyncio.sleep(interval)


@dataclass
class Daemon:
    """An arbiter + server in the test's event loop, with programmable fakes."""

    path: str
    gpu: FakeGpu
    host: FakeHost
    clock: Clock
    settings: ArbiterSettings
    arb: Arbiter | None = None
    server: Server | None = None
    clients: list[ArbiterClient] = field(default_factory=list)

    async def start(self) -> Daemon:
        self.arb = Arbiter(self.gpu, self.host, settings=self.settings, clock=self.clock)
        await self.arb.start()
        self.server = Server(self.arb, self.path)
        await self.server.start()
        return self

    async def stop(self) -> None:
        if self.server is not None:
            await self.server.close()
        if self.arb is not None:
            await self.arb.close()
        self.server = self.arb = None

    async def client(self, name: str, pid: int, **kwargs: Any) -> ArbiterClient:
        """A connected client (constructed off-loop: its constructor blocks on the handshake)."""
        client = await asyncio.to_thread(ArbiterClient, name, socket=self.path, pid=pid, **kwargs)
        self.clients.append(client)
        return client

    async def quiesce(self) -> None:
        assert self.arb is not None
        await self.arb.quiesce()

    def model(self, model_id: str):
        assert self.arb is not None
        return self.arb.registry.models.get(model_id)

    def state(self, model_id: str) -> str | None:
        m = self.model(model_id)
        return m.state_label if m else None

    async def advance(self, seconds: float) -> None:
        assert isinstance(self.clock, FakeClock)
        self.clock.advance(seconds)
        await self.quiesce()


def make_daemon(
    path: str,
    *,
    vram: str | int = "24GiB",
    host_ram: str | int = "64GiB",
    clock: Clock | None = None,
    **settings: Any,
) -> Daemon:
    defaults: dict[str, Any] = dict(headroom=0, host_headroom=0, settle_timeout_s=5)
    defaults.update(settings)
    return Daemon(
        path=path,
        gpu=FakeGpu({0: vram}),
        host=FakeHost(host_ram),
        clock=clock or FakeClock(),
        settings=ArbiterSettings(**defaults),
    )


class FakeWeights:
    """A model's load/unload callbacks that allocate on the fake GPU, with call counters.

    ``peak`` is allocated during the load and trimmed to ``resident`` when it finishes, like a
    real loader whose staging buffers are freed at the end.
    """

    def __init__(
        self,
        gpu: FakeGpu,
        pid: int,
        resident: float,
        peak: float | None = None,
        device: int = 0,
        load_delay: float = 0.0,
        host: FakeHost | None = None,
        host_peak: float = 0.0,
    ):
        self.gpu, self.pid, self.device = gpu, pid, device
        self.resident = int(resident * GiB)
        self.peak = int((peak or resident) * GiB)
        self.load_delay = load_delay
        self.host = host
        self.host_peak = int(host_peak * GiB)
        self.loads = 0
        self.unloads = 0
        self.fail_next = False
        self.held = 0

    def load(self) -> None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("weights corrupted")
        if self.host is not None and self.host_peak:
            self.host.take(self.host_peak)
        try:
            self.gpu.allocate(self.pid, self.peak, self.device)
            if self.load_delay:
                time.sleep(self.load_delay)
            self.gpu.free(self.pid, self.peak - self.resident, self.device)
        finally:
            if self.host is not None and self.host_peak:
                self.host.give(self.host_peak)
        self.held = self.resident
        self.loads += 1

    def unload(self) -> None:
        self.gpu.free(self.pid, self.held, self.device)
        self.held = 0
        self.unloads += 1


__all__ = [
    "Daemon",
    "FakeWeights",
    "make_daemon",
    "short_tmpdir",
    "remove_tree",
    "wait_for",
    "os",
]


def free_port() -> int:
    """A TCP port that was free a moment ago (good enough for tests)."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); ask the process table when psutil is around.
    try:
        import psutil

        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return True


FAKES = os.path.join(os.path.dirname(__file__), "fakes")
