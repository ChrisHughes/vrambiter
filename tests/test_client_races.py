"""Client races found in review, each with its window widened so the test is deterministic.

Real time (``LoopClock``): these are about thread and socket interleavings, not timeouts.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import time

import vrambiter.client as client_mod
from tests.helpers import make_daemon, wait_for
from vrambiter.client import ArbiterClient
from vrambiter.clock import LoopClock

GIB = 1 << 30


async def test_grant_arriving_as_the_caller_gives_up_is_released(sock_path, monkeypatch):
    """Cancelled just as its grant lands: the lease must not stay busy (or LOADING) forever."""
    real_abandon = client_mod.ArbiterClient._abandon

    def slow_abandon(self, request_id):
        time.sleep(0.3)  # the worker thread is scheduled late: the grant arrives first
        real_abandon(self, request_id)

    monkeypatch.setattr(client_mod.ArbiterClient, "_abandon", slow_abandon)
    d = make_daemon(sock_path, clock=LoopClock(), max_concurrent_loads=1)
    await d.start()
    try:
        client = await d.client("sat", pid=4242)
        gate = threading.Event()
        b = await asyncio.to_thread(client.register, "b", vram_peak=GIB, load=gate.wait)
        a = await asyncio.to_thread(client.register, "a", vram_peak=GIB, load=lambda: None)
        threading.Thread(target=lambda: b.acquire().release(), daemon=True).start()
        await wait_for(lambda: d.state("sat/b") == "loading")  # holds the only load slot
        task = asyncio.ensure_future(a.aacquire())
        await wait_for(lambda: d.arb.queue)
        task.cancel()  # the caller gives up...
        gate.set()  # ...just as the slot frees and a is granted "load"
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await wait_for(lambda: d.state("sat/a") == "unloaded" and not d.model("sat/a").leases)
        c = await asyncio.to_thread(client.register, "c", vram_peak=GIB, load=lambda: None)
        lease = await c.aacquire(timeout=5)  # the load slot is free again
        lease.release()
    finally:
        for c in d.clients:
            c.close()
        await d.stop()


async def test_ready_grant_for_a_model_being_unloaded_reloads_through_admission(sock_path):
    """A lease racing Model.unload(): the client must not reload behind the arbiter's back."""
    d = make_daemon(sock_path, clock=LoopClock())
    await d.start()
    try:
        client = await d.client("sat", pid=4242)
        loads = []
        a = await asyncio.to_thread(
            client.register,
            "a",
            vram_peak=GIB,
            load=lambda: loads.append(1),
            unload=lambda: time.sleep(0.5),
        )
        await asyncio.to_thread(lambda: a.acquire().release())
        await wait_for(lambda: d.state("sat/a") == "resident")
        unloader = threading.Thread(target=a.unload)
        unloader.start()
        await asyncio.sleep(0.1)  # unload() is running; the arbiter still says resident
        lease = await asyncio.to_thread(a.acquire)
        unloader.join(5)
        await wait_for(lambda: d.state("sat/a") == "busy")  # `loaded` travels asynchronously
        assert len(loads) == 2  # reloaded...
        assert lease.arbiter_lease in d.model("sat/a").leases  # ...and the arbiter knows
        lease.release()
    finally:
        for c in d.clients:
            c.close()
        await d.stop()


class FlakyProxy:
    """A Unix-socket proxy in front of the daemon whose connections the test can cut or slow."""

    def __init__(self, real: str, path: str) -> None:
        self.real, self.path = real, path
        self.count = 0
        self.links: dict[int, tuple[asyncio.StreamWriter, asyncio.StreamWriter]] = {}
        self.slow_reply_on: dict[int, int] = {}  # connection -> delay its Nth reply line
        self.drop_after_first_request: set[int] = set()
        self.server: asyncio.base_events.Server | None = None

    async def start(self) -> None:
        self.server = await asyncio.start_unix_server(self._handle, path=self.path)

    async def _handle(self, cr, cw) -> None:
        self.count += 1
        n = self.count
        ar, aw = await asyncio.open_unix_connection(self.real)
        self.links[n] = (cw, aw)

        async def pipe(reader, writer, upstream: bool) -> None:
            k = 0
            try:
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    k += 1
                    if not upstream and self.slow_reply_on.get(n) == k:
                        await asyncio.sleep(0.5)
                    writer.write(line)
                    if upstream and n in self.drop_after_first_request:
                        await asyncio.sleep(0.05)
                        self.cut(n)
                        return
            except Exception:
                pass
            finally:
                writer.close()

        await asyncio.gather(pipe(cr, aw, True), pipe(ar, cw, False))

    def cut(self, n: int) -> None:
        for w in self.links[n]:
            w.close()

    def close(self) -> None:
        if self.server is not None:
            self.server.close()


async def test_link_drop_mid_handshake_leaves_one_consistent_connection(sock_dir):
    real, proxy_path = os.path.join(sock_dir, "r.sock"), os.path.join(sock_dir, "p.sock")
    d = make_daemon(real, clock=LoopClock(), max_concurrent_loads=1)
    await d.start()
    proxy = FlakyProxy(real, proxy_path)
    proxy.drop_after_first_request.add(2)  # the first reconnect dies during its hello
    await proxy.start()
    client = await asyncio.to_thread(ArbiterClient, "sat", socket=proxy_path, pid=4242)
    try:
        gate = threading.Event()
        b = await asyncio.to_thread(client.register, "b", vram_peak=GIB, load=gate.wait)
        a = await asyncio.to_thread(client.register, "a", vram_peak=GIB, load=lambda: None)
        proxy.cut(1)
        await wait_for(lambda: client.connected and proxy.count >= 3, timeout=10)
        threading.Thread(target=lambda: b.acquire().release(), daemon=True).start()
        await wait_for(lambda: d.state("sat/b") == "loading")
        result: dict[str, object] = {}
        taker = threading.Thread(target=lambda: result.setdefault("lease", a.acquire()))
        taker.start()
        await asyncio.sleep(0.3)
        gate.set()
        await asyncio.to_thread(taker.join, 5)
        assert not taker.is_alive(), "an acquire on the live connection must be answered"
        assert proxy.count == 3, "exactly one reconnect loop"
        result["lease"].release()  # type: ignore[attr-defined]
    finally:
        proxy.close()
        client.close()
        await d.stop()


async def test_model_registered_during_a_reconnect_is_registered(sock_dir):
    real, proxy_path = os.path.join(sock_dir, "r.sock"), os.path.join(sock_dir, "p.sock")
    d = make_daemon(real, clock=LoopClock())
    await d.start()
    proxy = FlakyProxy(real, proxy_path)
    proxy.slow_reply_on[2] = 2  # the arbiter is slow to answer the re-registration
    await proxy.start()
    client = await asyncio.to_thread(ArbiterClient, "sat", socket=proxy_path, pid=4242)
    try:
        await asyncio.to_thread(client.register, "a", vram_peak=GIB, load=lambda: None)
        proxy.cut(1)
        await wait_for(lambda: client._epoch >= 2)
        await asyncio.sleep(0.15)  # re-registering "a", reply delayed
        late = await asyncio.to_thread(client.register, "late", vram_peak=GIB, load=lambda: None)
        await wait_for(lambda: d.model("sat/late") is not None, timeout=10)
        # The client counts it registered once the arbiter's reply is back.
        await wait_for(lambda: client.connected and late._registered_epoch == client._epoch)
        lease = await asyncio.to_thread(late.acquire)
        assert not lease.standalone
        lease.release()
    finally:
        proxy.close()
        client.close()
        await d.stop()
