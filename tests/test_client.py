"""The client library: standalone mode, leases, evictions, async bridging, reconnects."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

import vrambiter
from tests.helpers import FakeWeights, make_daemon, wait_for
from vrambiter import protocol as p
from vrambiter.client import ArbiterClient, Model, NullArbiter
from vrambiter.errors import ArbiterUnavailable, RegistrationError, UnknownModel, VramUnavailable
from vrambiter.units import GiB

# --------------------------------------------------------------------------- connect / standalone


def test_connect_without_daemon_is_standalone(sock_path):
    arb = vrambiter.connect("tts", socket=sock_path)
    assert isinstance(arb, NullArbiter) and not arb.connected
    with pytest.raises(ArbiterUnavailable):
        vrambiter.connect("tts", socket=sock_path, required=True)
    with pytest.raises(ArbiterUnavailable):
        arb.status()
    with pytest.raises(ArbiterUnavailable):
        arb.pin("x/y")


def test_connect_needs_a_name(sock_path, monkeypatch):
    with pytest.raises(ValueError):
        vrambiter.connect(socket=sock_path)
    monkeypatch.setenv("VRAMBITER_NAME", "from-env")
    assert vrambiter.connect("in-code", socket=sock_path).name == "from-env"


def test_default_socket_path(monkeypatch):
    monkeypatch.setenv("VRAMBITER_SOCKET", "/x/y.sock")
    assert vrambiter.default_socket_path() == "/x/y.sock"
    monkeypatch.delenv("VRAMBITER_SOCKET")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    assert vrambiter.default_socket_path() == "/run/user/1000/vrambiter.sock"
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    assert vrambiter.default_socket_path().startswith("/tmp/vrambiter-")


def test_standalone_loads_on_first_use_and_never_unloads():
    arb = NullArbiter("solo")
    calls = []
    model = arb.register(
        "m",
        vram_peak="9GiB",
        load=lambda: calls.append("load"),
        unload=lambda: calls.append("unload"),
    )
    assert not model.loaded
    with model.lease() as lease:
        assert lease.standalone and lease.action == "standalone"
        assert model.loaded and model.leases == 1
    with model.lease():
        pass
    assert calls == ["load"]
    assert model.leases == 0
    with arb.lease("llama/gemma") as consumer:  # consumer leases are no-ops
        assert consumer.standalone
    with pytest.raises(ValueError):
        arb.register("m", vram_peak=1, load=lambda: None)


def test_standalone_load_failure_propagates_and_retries():
    arb = NullArbiter("solo")
    attempts = []

    def load():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("disk on fire")

    model = arb.register("m", vram_peak=1, load=load)
    with pytest.raises(RuntimeError), model.lease():
        pass
    assert model.leases == 0 and not model.loaded
    with model.lease():
        assert model.loaded


async def test_standalone_alease():
    arb = NullArbiter("solo")
    loaded = []
    model = arb.register("m", vram_peak=1, load=lambda: loaded.append(threading.get_ident()))
    async with model.alease() as lease:
        assert lease.standalone
    assert loaded and loaded[0] != threading.get_ident()  # load ran off the event loop


# --------------------------------------------------------------------------- connected


async def test_register_lease_load_and_release(daemon):
    client = await daemon.client("tts", pid=100)
    assert client.connected and isinstance(client, ArbiterClient)
    weights = FakeWeights(daemon.gpu, 100, resident=8, peak=9)
    model = await asyncio.to_thread(
        client.register,
        "voice",
        vram_peak="9GiB",
        load=weights.load,
        unload=weights.unload,
        measure=lambda: weights.held,
    )
    assert daemon.state("tts/voice") == "unloaded"

    def use():
        with model.lease() as lease:
            assert lease.action == "load" and not lease.standalone
            assert daemon.state("tts/voice") in ("loading", "busy")
            assert weights.held == 8 * GiB
        with model.lease() as lease:
            assert lease.action == "ready"

    await asyncio.to_thread(use)
    await wait_for(lambda: daemon.state("tts/voice") == "resident")  # release is async
    assert weights.loads == 1
    assert daemon.model("tts/voice").reported_bytes == 8 * GiB


async def test_register_errors_surface(daemon):
    client = await daemon.client("tts", pid=100)
    with pytest.raises(RegistrationError):
        await asyncio.to_thread(
            client.register, "bad", vram_peak="1GiB", vram_resident="2GiB", load=lambda: None
        )
    assert "bad" not in client.models


async def test_eviction_unloads_and_next_lease_reloads(daemon):
    a = await daemon.client("a", pid=100)
    b = await daemon.client("b", pid=200)
    wa = FakeWeights(daemon.gpu, 100, resident=16)
    wb = FakeWeights(daemon.gpu, 200, resident=16)
    ma = await asyncio.to_thread(a.register, "m", vram_peak="16GiB", load=wa.load, unload=wa.unload)
    mb = await asyncio.to_thread(b.register, "m", vram_peak="16GiB", load=wb.load, unload=wb.unload)

    def lease_once(model):
        with model.lease():
            pass

    await asyncio.to_thread(lease_once, ma)
    await asyncio.to_thread(lease_once, mb)  # 24 GiB card: a/m must go
    assert wa.unloads == 1 and not ma.loaded
    assert mb.loaded
    await asyncio.to_thread(lease_once, ma)  # and now b/m goes
    assert wb.unloads == 1 and wa.loads == 2


def _evict_target():
    sent = []

    class Recorder(NullArbiter):
        def _notify(self, message, epoch=None):
            sent.append(message)

    arb = Recorder("x")
    unloads = []
    model = arb.register("m", vram_peak=1, load=lambda: None, unload=lambda: unloads.append(1))
    return model, sent, unloads


def test_local_lease_refuses_eviction():
    model, sent, unloads = _evict_target()
    lease = model.acquire()
    model._handle_evict(p.Evict(evict_id="E1", model="m"), epoch=0)
    assert isinstance(sent[-1], p.EvictRefused) and sent[-1].reason == "in use"
    assert unloads == [] and model.loaded
    lease.release()
    model._handle_evict(p.Evict(evict_id="E2", model="m"), epoch=0)
    assert isinstance(sent[-1], p.Evicted) and unloads == [1] and not model.loaded


def test_failed_unload_refuses_eviction():
    model, sent, _ = _evict_target()
    model._unload_fn = lambda: 1 / 0
    with model.lease():
        pass
    model._handle_evict(p.Evict(evict_id="E1", model="m"), epoch=0)
    assert isinstance(sent[-1], p.EvictRefused) and "unload failed" in sent[-1].reason


def test_lease_arriving_during_unload_waits_then_reloads():
    """An eviction never races a lease: the lease blocks on the model lock, then reloads."""
    sent = []

    class Recorder(NullArbiter):
        def _notify(self, message, epoch=None):
            sent.append(message)

    arb = Recorder("x")
    unloading = threading.Event()
    proceed = threading.Event()
    events = []

    def unload():
        unloading.set()
        proceed.wait(5)
        events.append("unloaded")

    model: Model = arb.register(
        "m", vram_peak=1, load=lambda: events.append("loaded"), unload=unload
    )
    with model.lease():
        pass
    evictor = threading.Thread(
        target=model._handle_evict, args=(p.Evict(evict_id="E1", model="m"), 0)
    )
    evictor.start()
    assert unloading.wait(5)
    leaser = threading.Thread(target=lambda: model.acquire().release())
    leaser.start()
    time.sleep(0.05)
    assert events == ["loaded"]  # the lease is blocked behind the unload
    proceed.set()
    evictor.join(5)
    leaser.join(5)
    assert events == ["loaded", "unloaded", "loaded"]


async def test_voluntary_unload(daemon):
    client = await daemon.client("a", pid=100)
    w = FakeWeights(daemon.gpu, 100, resident=4)
    model = await asyncio.to_thread(
        client.register, "m", vram_peak="4GiB", load=w.load, unload=w.unload
    )
    await asyncio.to_thread(lambda: model.acquire().release())
    lease = await asyncio.to_thread(model.acquire)
    assert not model.unload()  # in use
    lease.release()
    assert model.unload()
    await wait_for(lambda: daemon.state("a/m") == "unloaded")


async def test_load_failure_reports_and_releases(daemon):
    client = await daemon.client("a", pid=100)
    w = FakeWeights(daemon.gpu, 100, resident=4)
    w.fail_next = True
    model = await asyncio.to_thread(
        client.register, "m", vram_peak="4GiB", load=w.load, unload=w.unload
    )
    with pytest.raises(RuntimeError, match="corrupted"):
        await asyncio.to_thread(lambda: model.acquire())
    await wait_for(lambda: daemon.state("a/m") == "unloaded" and not daemon.model("a/m").leases)
    assert model.leases == 0
    lease = await asyncio.to_thread(model.acquire)
    assert lease.action == "load"
    lease.release()


async def test_timeout_raises_vram_unavailable(daemon):
    a = await daemon.client("a", pid=100)
    b = await daemon.client("b", pid=200)
    wa = FakeWeights(daemon.gpu, 100, resident=20)
    ma = await asyncio.to_thread(
        a.register, "big", vram_peak="20GiB", load=wa.load, unload=wa.unload
    )
    mb = await asyncio.to_thread(b.register, "other", vram_peak="10GiB", load=lambda: None)
    held = await asyncio.to_thread(ma.acquire)
    attempt = asyncio.ensure_future(asyncio.to_thread(lambda: mb.acquire(timeout=5)))
    await wait_for(lambda: daemon.arb.queue)
    await daemon.advance(5.1)
    with pytest.raises(VramUnavailable) as info:
        await attempt
    assert "timed out" in str(info.value)
    assert info.value.holders[0].name == "a/big"
    assert mb.leases == 0
    held.release()


async def test_alease_and_cancellation_withdraws_the_request(daemon):
    a = await daemon.client("a", pid=100)
    b = await daemon.client("b", pid=200)
    wa = FakeWeights(daemon.gpu, 100, resident=20)
    ma = await asyncio.to_thread(
        a.register, "big", vram_peak="20GiB", load=wa.load, unload=wa.unload
    )
    mb = await asyncio.to_thread(b.register, "other", vram_peak="10GiB", load=lambda: None)
    async with ma.alease() as lease:
        assert lease.action == "load"
        task = asyncio.ensure_future(mb.aacquire())
        await wait_for(lambda: daemon.arb.queue)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_for(lambda: not daemon.arb.queue)
    assert mb.leases == 0


async def test_consumer_lease_on_unknown_model(daemon):
    client = await daemon.client("app", pid=300)
    with pytest.raises(UnknownModel):
        await asyncio.to_thread(lambda: client.acquire("llama/nope"))


async def test_control_operations(daemon):
    client = await daemon.client("a", pid=100)
    w = FakeWeights(daemon.gpu, 100, resident=4)
    await asyncio.to_thread(client.register, "m", vram_peak="4GiB", load=w.load, unload=w.unload)
    status = await asyncio.to_thread(client.status)
    assert status["models"][0]["id"] == "a/m"
    await asyncio.to_thread(client.pin, "a/m")
    assert daemon.model("a/m").pinned
    await asyncio.to_thread(client.unpin, "a/m")
    assert not daemon.model("a/m").pinned


# --------------------------------------------------------------------------- reconnect


async def test_arbiter_restart_client_falls_back_then_reregisters(sock_path):
    d = make_daemon(sock_path)
    await d.start()
    client = await d.client("tts", pid=100)
    w = FakeWeights(d.gpu, 100, resident=8)
    model = await asyncio.to_thread(
        client.register, "voice", vram_peak="8GiB", load=w.load, unload=w.unload
    )
    lease = await asyncio.to_thread(model.acquire)
    assert not lease.standalone
    gpu = d.gpu
    await d.stop()
    await wait_for(lambda: not client.connected)

    # Standalone meanwhile: leases work, nothing reloads.
    def use():
        with model.lease() as l2:
            return l2.standalone

    assert await asyncio.to_thread(use)
    assert w.loads == 1

    # A new arbiter on the same socket: the client re-registers resident, with its held lease.
    d2 = make_daemon(sock_path)
    d2.gpu = gpu
    await d2.start()
    try:
        await wait_for(lambda: client.connected, timeout=10)
        await d2.quiesce()
        record = d2.model("tts/voice")
        assert record.state_label == "busy" and len(record.leases) == 1
        assert lease.arbiter_lease in record.leases
        lease.release()
        await wait_for(lambda: d2.state("tts/voice") == "resident")
        assert w.loads == 1
    finally:
        client.close()
        await d2.stop()
