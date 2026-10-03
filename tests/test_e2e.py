"""End to end: real Unix socket, real client library, FakeGpu, FakeHost and a fake clock.

Each "satellite" is an ``ArbiterClient`` with its own fake pid; its load/unload callbacks allocate
and free on the shared FakeGpu, which raises a fake OOM if the arbiter ever over-admits. Sync
client calls run in worker threads so the daemon (in the test's event loop) keeps serving.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

import vrambiter
from tests.helpers import FakeWeights, make_daemon, wait_for
from vrambiter.errors import HostRamUnavailable, VramUnavailable
from vrambiter.units import GiB


async def register(client, name, weights: FakeWeights, peak: float | None = None, **kw):
    return await asyncio.to_thread(
        client.register,
        name,
        vram_peak=int((peak or weights.peak / GiB) * GiB),
        load=weights.load,
        unload=weights.unload,
        measure=lambda: weights.held,
        **kw,
    )


def use(model, seconds: float = 0.0, **kw):
    """Take a lease, hold it, release it; returns the lease action."""
    with model.lease(**kw) as lease:
        if seconds:
            time.sleep(seconds)
        return lease.action


async def test_admission_and_reuse(daemon):
    tts = await daemon.client("tts", pid=101)
    w = FakeWeights(daemon.gpu, 101, resident=8, peak=9)
    voice = await register(tts, "voice", w)
    assert await asyncio.to_thread(use, voice) == "load"
    assert await asyncio.to_thread(use, voice) == "ready"
    assert w.loads == 1 and daemon.gpu.usage(101) == 8 * GiB


async def test_lru_eviction_across_two_satellites(daemon):
    a = await daemon.client("a", pid=101)
    b = await daemon.client("b", pid=202)
    wa1 = FakeWeights(daemon.gpu, 101, resident=8)
    wa2 = FakeWeights(daemon.gpu, 101, resident=8)
    wb = FakeWeights(daemon.gpu, 202, resident=14)
    old = await register(a, "old", wa1)
    await asyncio.to_thread(use, old)
    await wait_for(lambda: daemon.state("a/old") == "resident")  # release processed
    daemon.clock.advance(5)
    new = await register(a, "new", wa2)
    await asyncio.to_thread(use, new)
    big = await register(b, "big", wb)
    # 24 GiB card, 16 held, 14 needed: only the least recently used of a's models goes.
    assert await asyncio.to_thread(use, big) == "load"
    assert wa1.unloads == 1 and wa2.unloads == 0
    assert not old.loaded and new.loaded and big.loaded
    await wait_for(lambda: daemon.state("a/old") == "unloaded")


async def test_local_lease_refuses_an_eviction_over_the_wire(daemon, monkeypatch):
    a = await daemon.client("a", pid=101)
    ctl = await daemon.client("ops", pid=1, role="control")
    w = FakeWeights(daemon.gpu, 101, resident=8)
    model = await register(a, "m", w)
    await asyncio.to_thread(use, model)

    # Hold a's next acquire after the lease is counted locally but before the arbiter hears of
    # it: exactly the window in which the arbiter still believes the model is idle.
    gate = threading.Event()
    counted = threading.Event()
    real_acquire = a._acquire

    def gated(*args, **kwargs):
        counted.set()
        gate.wait(5)
        return real_acquire(*args, **kwargs)

    monkeypatch.setattr(a, "_acquire", gated)
    lease_task = asyncio.ensure_future(asyncio.to_thread(model.acquire))
    await asyncio.to_thread(counted.wait, 5)
    await asyncio.to_thread(ctl.evict, "a/m")
    # The client refuses: the model stays loaded and the arbiter puts it back.
    await wait_for(lambda: daemon.state("a/m") == "resident")
    assert w.unloads == 0 and model.loaded
    gate.set()
    lease = await lease_task
    assert lease.action == "ready" and w.loads == 1
    lease.release()


async def test_request_timeout(daemon):
    a = await daemon.client("a", pid=101)
    b = await daemon.client("b", pid=202)
    big = await register(a, "big", FakeWeights(daemon.gpu, 101, resident=20))
    other = await register(b, "other", FakeWeights(daemon.gpu, 202, resident=10))
    held = await asyncio.to_thread(big.acquire)
    attempt = asyncio.ensure_future(asyncio.to_thread(use, other, timeout=30))
    await wait_for(lambda: daemon.arb.queue)
    await daemon.advance(29)
    assert not attempt.done()
    await daemon.advance(1.5)
    with pytest.raises(VramUnavailable, match="timed out"):
        await attempt
    held.release()


async def test_evict_timeout_marks_unresponsive_and_fails_fast(daemon):
    a = await daemon.client("a", pid=101)
    b = await daemon.client("b", pid=202)
    stuck = threading.Event()
    wa = FakeWeights(daemon.gpu, 101, resident=20)
    real_unload = wa.unload

    def slow_unload():
        stuck.wait(10)  # a satellite wedged in its unload callback
        real_unload()

    m = await asyncio.to_thread(
        a.register, "m", vram_peak="20GiB", load=wa.load, unload=slow_unload
    )
    await asyncio.to_thread(use, m)
    n = await register(b, "n", FakeWeights(daemon.gpu, 202, resident=10))
    attempt = asyncio.ensure_future(asyncio.to_thread(use, n))
    await wait_for(lambda: daemon.state("a/m") == "evicting")
    await daemon.advance(30)
    with pytest.raises(VramUnavailable, match="unresponsive"):
        await attempt
    assert daemon.model("a/m").unresponsive
    stuck.set()  # it finally unloads; the late `evicted` is accepted
    await wait_for(lambda: daemon.state("a/m") == "unloaded")
    assert await asyncio.to_thread(use, n) == "load"


async def test_disconnect_cleanup_lets_waiter_proceed(daemon):
    a = await daemon.client("a", pid=101)
    b = await daemon.client("b", pid=202)
    big = await register(a, "big", FakeWeights(daemon.gpu, 101, resident=20))
    other = await register(b, "other", FakeWeights(daemon.gpu, 202, resident=10))
    await asyncio.to_thread(big.acquire)  # never released: a dies holding it
    attempt = asyncio.ensure_future(asyncio.to_thread(use, other))
    await wait_for(lambda: daemon.arb.queue)
    a.close()
    daemon.clients.remove(a)
    await wait_for(lambda: daemon.state("a/big") == "unloaded")
    assert not attempt.done()  # the process still holds its memory
    daemon.gpu.kill(101)  # ...until it exits
    await daemon.advance(daemon.settings.settle_poll_s)
    assert await attempt == "load"


async def test_queue_is_priority_then_fifo(daemon):
    holder = await daemon.client("holder", pid=100)
    hw = FakeWeights(daemon.gpu, 100, resident=24)
    hog = await register(holder, "hog", hw)
    held = await asyncio.to_thread(hog.acquire)

    order: list[str] = []
    tasks = []
    clients = {}
    for i, (name, prio) in enumerate([("first", 0), ("vip", 5), ("second", 0)]):
        clients[name] = await daemon.client(name, pid=200 + i)
        model = await register(
            clients[name], "m", FakeWeights(daemon.gpu, 200 + i, resident=12), priority=prio
        )

        def run(model=model, name=name):
            with model.lease():
                order.append(name)
            return name

        tasks.append(asyncio.ensure_future(asyncio.to_thread(run)))
        await wait_for(lambda n=i: len(daemon.arb.queue) == n + 1)
    held.release()
    await asyncio.gather(*tasks)
    # 24 GiB card, 12 each: vip and first fit (in that order); second evicts one of them.
    assert order[:2] == ["vip", "first"]
    assert order[2] == "second"


async def test_loads_never_overlap_with_one_load_slot(sock_path):
    d = make_daemon(sock_path, vram="80GiB", host_ram="27GiB", host_headroom=3 * GiB)
    await d.start()
    try:
        active = 0
        peak_active = 0
        lock = threading.Lock()
        clients = [await d.client(f"s{i}", pid=300 + i) for i in range(3)]
        models = []
        for i, c in enumerate(clients):
            w = FakeWeights(d.gpu, 300 + i, resident=10, host=d.host, host_peak=14)

            def load(w=w):
                nonlocal active, peak_active
                with lock:
                    active += 1
                    peak_active = max(peak_active, active)
                time.sleep(0.05)
                w.load()
                with lock:
                    active -= 1

            models.append(
                await asyncio.to_thread(
                    c.register,
                    "m",
                    vram_peak="10GiB",
                    host_peak="14GiB",
                    load=load,
                    unload=w.unload,
                )
            )
        await asyncio.gather(*(asyncio.to_thread(use, m) for m in models))
        # 27 GiB of host RAM, 14 GiB per load: two at once would OOM the host. They queued.
        assert peak_active == 1
    finally:
        for c in d.clients:
            c.close()
        await d.stop()


async def test_host_ram_shortage_fails_with_host_error(sock_path):
    d = make_daemon(sock_path, host_ram="10GiB", host_headroom=3 * GiB)
    await d.start()
    try:
        c = await d.client("s", pid=300)
        m = await asyncio.to_thread(
            c.register, "m", vram_peak="1GiB", host_peak="8GiB", load=lambda: None
        )
        with pytest.raises(HostRamUnavailable, match="host RAM"):
            await asyncio.to_thread(use, m)
    finally:
        for c in d.clients:
            c.close()
        await d.stop()


async def test_working_headroom_reserved_while_busy(daemon):
    """A busy model may grow to its peak: nothing else is admitted into that room."""
    a = await daemon.client("a", pid=101)
    b = await daemon.client("b", pid=202)
    sdxl = await register(a, "sdxl", FakeWeights(daemon.gpu, 101, resident=7), peak=17)
    other = await register(b, "other", FakeWeights(daemon.gpu, 202, resident=8))
    await asyncio.to_thread(use, sdxl)
    lease = await asyncio.to_thread(sdxl.acquire)  # working: may reach 17 GiB
    attempt = asyncio.ensure_future(asyncio.to_thread(use, other))
    await wait_for(lambda: daemon.arb.queue and "waiting for a/sdxl" in daemon.arb.queue[0].reason)
    lease.release()
    assert await attempt == "load"


async def test_standalone_without_daemon(sock_path):
    arb = vrambiter.connect("lonely", socket=sock_path)
    assert isinstance(arb, vrambiter.NullArbiter)
    loads = []
    m = arb.register("m", vram_peak="1GiB", load=lambda: loads.append(1))
    async with m.alease():
        pass
    async with m.alease():
        pass
    assert loads == [1]


async def test_restart_reconnect_then_coordinate_again(sock_path):
    d = make_daemon(sock_path)
    await d.start()
    a = await d.client("a", pid=101)
    b = await d.client("b", pid=202)
    gpu = d.gpu
    wa = FakeWeights(gpu, 101, resident=16)
    wb = FakeWeights(gpu, 202, resident=16)
    ma = await register(a, "m", wa)
    mb = await register(b, "m", wb)
    await asyncio.to_thread(use, ma)
    await d.stop()
    await wait_for(lambda: not a.connected and not b.connected)

    d2 = make_daemon(sock_path)
    d2.gpu = gpu
    await d2.start()
    try:
        await wait_for(lambda: a.connected and b.connected, timeout=10)
        await d2.quiesce()
        assert d2.state("a/m") == "resident"  # believed from the re-registration
        # Coordination works again: b's load evicts a's resident model.
        assert await asyncio.to_thread(use, mb) == "load"
        assert wa.unloads == 1
    finally:
        a.close()
        b.close()
        await d2.stop()


async def test_status_through_the_client(daemon):
    a = await daemon.client("a", pid=101)
    daemon.gpu.allocate(9999, "2GiB")
    m = await register(a, "m", FakeWeights(daemon.gpu, 101, resident=4))
    await asyncio.to_thread(use, m)
    await wait_for(lambda: daemon.state("a/m") == "resident")
    status = await asyncio.to_thread(a.status)
    assert status["models"][0]["state"] == "resident"
    assert status["models"][0]["measured"] == 4 * GiB
    assert status["foreign"] == [{"pid": 9999, "device": 0, "bytes": 2 * GiB}]
    assert {s["name"] for s in status["satellites"]} == {"a"}


async def test_on_wait_reports_why_a_request_waits(daemon):
    a = await daemon.client("a", pid=101)
    b = await daemon.client("b", pid=202)
    big = await register(a, "big", FakeWeights(daemon.gpu, 101, resident=20))
    other = await register(b, "other", FakeWeights(daemon.gpu, 202, resident=10))
    held = await asyncio.to_thread(big.acquire)
    reasons: list[str] = []
    attempt = asyncio.ensure_future(asyncio.to_thread(lambda: use(other, on_wait=reasons.append)))
    await wait_for(lambda: reasons)
    assert "waiting for a/big" in reasons[0]
    held.release()
    assert await attempt == "load"
    assert any(r.startswith("evicting a/big") for r in reasons)
    heard = len(reasons)
    await asyncio.to_thread(use, other, on_wait=reasons.append)  # no wait, no callback
    assert len(reasons) == heard
