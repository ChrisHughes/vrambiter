"""The arbiter core, driven message by message through fake connections (no socket, fake clock).

End-to-end tests over a real socket with the real client live in ``test_e2e.py``; these pin down
the daemon's behaviour precisely, including timing, which is easier without threads in the way.
"""

from __future__ import annotations

import itertools

import pytest

from vrambiter import protocol as p
from vrambiter.arbiter import Arbiter, ArbiterSettings
from vrambiter.clock import FakeClock
from vrambiter.gpu import FakeGpu
from vrambiter.host import FakeHost
from vrambiter.units import GiB

_conn_ids = itertools.count(1)


class Sat:
    """A scripted satellite: sends messages, records replies, and holds fake GPU memory."""

    def __init__(self, arb: Arbiter, gpu: FakeGpu, name: str, pid: int, role: str = "satellite"):
        self.arb, self.gpu, self.name, self.pid = arb, gpu, name, pid
        self.id = next(_conn_ids)
        self.sent: list[p.Message] = []
        self.closed = False
        self._ids = itertools.count(1)
        arb.connected(self)
        self.welcome = self.request(p.Hello(name=name, pid=pid, role=role))

    # Connection protocol
    def send(self, message: p.Message) -> None:
        self.sent.append(message)

    def close(self) -> None:
        self.closed = True

    def request(self, message: p.Message) -> p.Message | None:
        message.id = next(self._ids)
        self.arb.handle(self, message)
        return self.reply(message.id)

    def reply(self, request_id: int) -> p.Message | None:
        for m in self.sent:
            if m.id == request_id and not isinstance(m, (p.Evict, p.Notice)):
                return m
        return None

    def evicts(self) -> list[p.Evict]:
        return [m for m in self.sent if isinstance(m, p.Evict)]

    def register(self, model: str, peak: float, resident: float | None = None, **kw) -> p.Message:
        return self.request(
            p.Register(
                model=model,
                vram_peak=int(peak * GiB),
                vram_resident=int(resident * GiB) if resident is not None else None,
                **kw,
            )
        )

    def acquire(self, model: str, **kw) -> int:
        msg = p.Acquire(model=model, **kw)
        msg.id = next(self._ids)
        self.arb.handle(self, msg)
        return msg.id

    def load(self, model: str, nbytes: float) -> None:
        """Pretend to load: allocate on the fake card, then report it."""
        self.gpu.allocate(self.pid, int(nbytes * GiB))
        assert isinstance(self.request(p.Loaded(model=model)), p.Ok)

    def evicted(self, ev: p.Evict, free: float | None = None) -> None:
        self.gpu.free(self.pid, int(free * GiB) if free is not None else None)
        assert isinstance(self.request(p.Evicted(model=ev.model, evict_id=ev.evict_id)), p.Ok)

    def disconnect(self) -> None:
        self.arb.disconnected(self)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def gpu():
    return FakeGpu({0: "24GiB"})


@pytest.fixture
def host():
    return FakeHost("64GiB")


@pytest.fixture
def settings():
    return ArbiterSettings(headroom=0, host_headroom=0, evict_timeout_s=30, settle_timeout_s=5)


@pytest.fixture
async def arb(gpu, host, clock, settings):
    a = Arbiter(gpu, host, settings=settings, clock=clock)
    await a.start()
    yield a
    await a.close()


async def granted(arb: Arbiter, sat: Sat, request_id: int) -> p.Granted:
    await arb.quiesce()
    reply = sat.reply(request_id)
    assert isinstance(reply, p.Granted), reply
    return reply


async def pending(arb: Arbiter, sat: Sat, request_id: int) -> None:
    await arb.quiesce()
    assert sat.reply(request_id) is None, sat.reply(request_id)


async def errored(arb: Arbiter, sat: Sat, request_id: int, code: str) -> p.Error:
    await arb.quiesce()
    reply = sat.reply(request_id)
    assert isinstance(reply, p.Error), reply
    assert reply.code == code, reply
    return reply


async def resident(arb, sat, model, size, peak=None, **kw):
    """Register, acquire, load and release: leaves ``model`` resident and idle."""
    sat.register(model, peak or size, **kw)
    rid = sat.acquire(model)
    g = await granted(arb, sat, rid)
    assert g.action == "load"
    sat.load(model, size)
    sat.request(p.Release(lease=g.lease))
    await arb.quiesce()
    return g


# --------------------------------------------------------------------------- connection basics


async def test_hello_welcome_and_errors(arb, gpu):
    a = Sat(arb, gpu, "tts", 100)
    assert isinstance(a.welcome, p.Welcome) and a.welcome.satellite == "tts"
    dup = Sat(arb, gpu, "tts", 101)
    assert isinstance(dup.welcome, p.Error) and dup.welcome.code == "name_in_use"
    bad = Sat(arb, gpu, "a/b", 102)
    assert isinstance(bad.welcome, p.Error)
    again = a.request(p.Hello(name="tts", pid=100))
    assert isinstance(again, p.Error)


async def test_requests_before_hello_are_rejected(arb, gpu):
    sat = Sat.__new__(Sat)
    sat.arb, sat.gpu, sat.name, sat.pid = arb, gpu, "x", 1
    sat.id, sat.sent, sat.closed, sat._ids = next(_conn_ids), [], False, itertools.count(1)
    arb.connected(sat)
    reply = sat.request(p.StatusRequest())
    assert isinstance(reply, p.Error) and "hello" in reply.message


async def test_name_free_again_after_disconnect(arb, gpu):
    a = Sat(arb, gpu, "tts", 100)
    a.disconnect()
    b = Sat(arb, gpu, "tts", 200)
    assert isinstance(b.welcome, p.Welcome)


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        (dict(model="m", vram_peak=10, vram_resident=11), "exceeds vram_peak"),
        (dict(model="m", vram_peak=10, device=3), "no GPU 3"),
        (dict(model="m", vram_peak=-1), ">= 0"),
        (dict(model="m", vram_peak=1, state="sideways"), "state must be"),
        (dict(model="", vram_peak=1), "non-empty"),
    ],
)
async def test_register_validation(arb, gpu, kwargs, fragment):
    sat = Sat(arb, gpu, "s", 1)
    reply = sat.request(p.Register(**kwargs))
    assert isinstance(reply, p.Error) and reply.code == "registration_error"
    assert fragment in reply.message


async def test_control_connection_cannot_register(arb, gpu):
    ctl = Sat(arb, gpu, "cli", 1, role="control")
    reply = ctl.request(p.Register(model="m", vram_peak=1))
    assert isinstance(reply, p.Error)


# --------------------------------------------------------------------------- admission


async def test_acquire_load_then_concurrent_lease_gets_ready(arb, gpu):
    sat = Sat(arb, gpu, "tts", 100)
    reply = sat.register("voice", 9)
    assert isinstance(reply, p.Ok) and reply.model == "tts/voice"
    first = sat.acquire("voice")
    g1 = await granted(arb, sat, first)
    assert g1.action == "load" and g1.model == "tts/voice"
    second = sat.acquire("voice")
    await pending(arb, sat, second)  # loading: wait for `loaded`
    sat.load("voice", 8)
    g2 = await granted(arb, sat, second)
    assert g2.action == "ready"
    third = sat.acquire("voice")
    assert (await granted(arb, sat, third)).action == "ready"  # busy: immediate


async def test_unknown_model_and_consumer_lease_on_cooperative(arb, gpu):
    a = Sat(arb, gpu, "a", 1)
    b = Sat(arb, gpu, "b", 2)
    a.register("m", 1)
    await errored(arb, b, b.acquire("nope"), "unknown_model")
    reply = await errored(arb, b, b.acquire("a/m"), "arbiter_error")
    assert "consumer" in reply.message


async def test_lru_eviction_round_trip(arb, gpu):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    await resident(arb, a, "old", 8)
    arb.clock.advance(10)
    await resident(arb, a, "new", 8)
    b.register("big", 15)
    rid = b.acquire("big")
    await pending(arb, b, rid)
    # 24 - 16 = 8 free, needs 15: evict the least recently used first, and only that.
    [ev] = a.evicts()
    assert ev.model == "old"
    a.evicted(ev, free=8)
    assert (await granted(arb, b, rid)).action == "load"
    status = arb.status()
    states = {m["id"]: m["state"] for m in status["models"]}
    assert states == {"a/old": "unloaded", "a/new": "resident", "b/big": "loading"}


async def test_busy_model_is_never_evicted_request_waits_then_proceeds(arb, gpu):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    a.register("m", 20)
    g = await granted(arb, a, a.acquire("m"))
    a.load("m", 20)
    b.register("n", 10)
    rid = b.acquire("n")
    await pending(arb, b, rid)
    assert a.evicts() == []
    assert "waiting for a/m" in arb.queue[0].reason
    a.request(p.Release(lease=g.lease))
    await arb.quiesce()
    [ev] = a.evicts()
    a.evicted(ev)
    assert (await granted(arb, b, rid)).action == "load"


async def test_evict_refused_then_lease_wins(arb, gpu, clock):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    await resident(arb, a, "m", 20)
    b.register("n", 10)
    rid = b.acquire("n")
    await arb.quiesce()
    [ev] = a.evicts()
    # A lease arrived on a's side first: it refuses and acquires.
    a.request(p.EvictRefused(model="m", evict_id=ev.evict_id, reason="in use"))
    lease_rid = a.acquire("m")
    g = await granted(arb, a, lease_rid)
    assert g.action == "ready"
    await pending(arb, b, rid)
    assert len(a.evicts()) == 1  # no immediate re-eviction
    a.request(p.Release(lease=g.lease))
    clock.advance(arb.settings.refuse_cooldown_s)
    await arb.quiesce()
    assert len(a.evicts()) == 2


async def test_refusal_cooldown_does_not_fail_the_waiting_request(arb, gpu, clock):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    await resident(arb, a, "m", 20)
    b.register("n", 10)
    rid = b.acquire("n")
    await arb.quiesce()
    [ev] = a.evicts()
    a.request(p.EvictRefused(model="m", evict_id=ev.evict_id))
    await pending(arb, b, rid)  # protected, so neither evicted nor a failure
    clock.advance(arb.settings.refuse_cooldown_s)
    await arb.quiesce()
    [_, ev2] = a.evicts()
    a.evicted(ev2)
    await granted(arb, b, rid)


async def test_evict_timeout_marks_unresponsive_and_replans(arb, gpu, clock):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    await resident(arb, a, "m", 20)
    b.register("n", 10)
    rid = b.acquire("n")
    await arb.quiesce()
    [ev] = a.evicts()
    clock.advance(30)
    err = await errored(arb, b, rid, "vram_unavailable")
    assert "unresponsive" in err.message
    model = arb.registry.models["a/m"]
    assert model.unresponsive
    # A late `evicted` is still accepted, and the satellite is trusted again.
    a.evicted(ev)
    assert not arb.registry.models["a/m"].unresponsive
    assert arb.registry.models["a/m"].phase.value == "unloaded"


async def test_request_timeout_fails_with_holders(arb, gpu, clock):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    a.register("m", 20)
    await granted(arb, a, a.acquire("m"))
    a.load("m", 20)
    b.register("n", 10)
    rid = b.acquire("n", timeout_s=5)
    await pending(arb, b, rid)
    clock.advance(4.9)
    await pending(arb, b, rid)
    clock.advance(0.2)
    err = await errored(arb, b, rid, "vram_unavailable")
    assert "timed out after 5.0s" in err.message
    assert err.detail["holders"][0]["name"] == "a/m"
    assert arb.queue == []


async def test_wait_false_fails_instead_of_queueing(arb, gpu):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    a.register("m", 20)
    await granted(arb, a, a.acquire("m"))
    a.load("m", 20)
    b.register("n", 10)
    err = await errored(arb, b, b.acquire("n", wait=False), "vram_unavailable")
    assert "would wait" in err.message
    err = await errored(arb, b, b.acquire("n", timeout_s=0), "vram_unavailable")


async def test_empty_registry_fails_fast_naming_foreign_process(arb, gpu):
    gpu.allocate(4321, "20GiB")  # someone else's process
    b = Sat(arb, gpu, "b", 200)
    b.register("n", 10)
    err = await errored(arb, b, b.acquire("n"), "vram_unavailable")
    assert "pid 4321" in err.message
    assert err.detail["holders"][0]["kind"] == "foreign"


async def test_never_fits_fails_immediately(arb, gpu):
    b = Sat(arb, gpu, "b", 200)
    b.register("huge", 30)
    err = await errored(arb, b, b.acquire("huge"), "vram_unavailable")
    assert "can ever offer" in err.message


async def test_working_headroom_for_lease_on_idle_resident_model(arb, gpu):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    await resident(arb, a, "other", 6)
    await resident(arb, b, "sdxl", 7, peak=17)  # 7 resident, may grow to 17 while working
    # 24 - 13 = 11 free; a lease on sdxl needs its 10 GiB headroom: fits.
    g = await granted(arb, b, b.acquire("sdxl"))
    assert g.action == "ready"
    b.request(p.Release(lease=g.lease))
    gpu.allocate(4321, "3GiB")  # foreign process: now only 8 free
    rid = b.acquire("sdxl")
    await arb.quiesce()
    [ev] = a.evicts()
    assert ev.model == "other"
    a.evicted(ev)
    await granted(arb, b, rid)


# --------------------------------------------------------------------------- loads and host RAM


async def test_loads_are_serialised(arb, gpu):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    a.register("x", 2)
    b.register("y", 2)
    ra = a.acquire("x")
    rb = b.acquire("y")
    assert (await granted(arb, a, ra)).action == "load"
    await pending(arb, b, rb)
    assert "max_concurrent_loads" in arb.queue[0].reason
    a.load("x", 2)
    assert (await granted(arb, b, rb)).action == "load"


async def test_host_ram_fail_without_load_in_flight(arb, gpu, host):
    host.set_available("4GiB")
    a = Sat(arb, gpu, "a", 100)
    a.register("x", 2, host_peak=6 * GiB)
    err = await errored(arb, a, a.acquire("x"), "host_ram_unavailable")
    assert "host RAM" in err.message


async def test_host_ram_waits_for_load_in_flight(gpu, host, clock):
    settings = ArbiterSettings(headroom=0, host_headroom=0, max_concurrent_loads=2)
    arb = Arbiter(gpu, host, settings=settings, clock=clock)
    await arb.start()
    try:
        host.set_available("10GiB")
        a = Sat(arb, gpu, "a", 100)
        b = Sat(arb, gpu, "b", 200)
        a.register("x", 2, host_peak=8 * GiB)
        b.register("y", 2, host_peak=8 * GiB)
        ra = a.acquire("x")
        rb = b.acquire("y")
        await granted(arb, a, ra)
        await pending(arb, b, rb)  # 10 - 8 reserved for the load in flight < 8
        assert "host RAM" in arb.queue[0].reason
        a.load("x", 2)
        await granted(arb, b, rb)
    finally:
        await arb.close()


async def test_load_failed_releases_reservation_and_lease(arb, gpu):
    a = Sat(arb, gpu, "a", 100)
    a.register("x", 10)
    g = await granted(arb, a, a.acquire("x"))
    assert isinstance(a.request(p.LoadFailed(model="x", error="boom")), p.Ok)
    model = arb.registry.models["a/x"]
    assert model.phase.value == "unloaded" and not model.leases
    assert isinstance(a.request(p.Release(lease=g.lease)), p.Ok)  # idempotent
    assert (await granted(arb, a, a.acquire("x"))).action == "load"


async def test_release_of_loader_lease_without_loaded_aborts_the_load(arb, gpu):
    a = Sat(arb, gpu, "a", 100)
    a.register("x", 10)
    g = await granted(arb, a, a.acquire("x"))
    a.request(p.Release(lease=g.lease))
    assert arb.registry.models["a/x"].phase.value == "unloaded"


# --------------------------------------------------------------------------- queue order


async def test_queue_grants_by_priority_then_fifo(arb, gpu):
    holder = Sat(arb, gpu, "h", 1)
    holder.register("big", 24)
    g = await granted(arb, holder, holder.acquire("big"))
    holder.load("big", 24)
    sats = {}
    order = [("low1", 0), ("high", 5), ("low2", 0)]
    rids = {}
    for name, prio in order:
        sats[name] = Sat(arb, gpu, name, 10 + len(sats))
        sats[name].register("m", 10, priority=prio)
        rids[name] = sats[name].acquire("m")
    await arb.quiesce()
    holder.request(p.Release(lease=g.lease))
    await arb.quiesce()
    [ev] = holder.evicts()
    holder.evicted(ev)
    granted_order = []
    for _ in range(3):
        await arb.quiesce()
        for name in ("low1", "high", "low2"):
            reply = sats[name].reply(rids[name])
            if isinstance(reply, p.Granted) and name not in granted_order:
                granted_order.append(name)
                sats[name].load("m", 10)
    # 24 GiB: high and low1 load (one at a time); low2 waits for room.
    assert granted_order[:2] == ["high", "low1"]


# --------------------------------------------------------------------------- disconnects


async def test_disconnect_ends_leases_and_frees_after_memory_returns(arb, gpu, clock):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    a.register("m", 20)
    await granted(arb, a, a.acquire("m"))
    a.load("m", 20)
    b.register("n", 10)
    rid = b.acquire("n")
    await pending(arb, b, rid)
    a.disconnect()
    assert not arb.registry.leases
    assert arb.registry.models["a/m"].phase.value == "unloaded"
    # The process has not released its memory yet: the request waits, evicting nothing.
    await pending(arb, b, rid)
    gpu.kill(100)
    clock.advance(arb.settings.settle_poll_s)
    await granted(arb, b, rid)


async def test_disconnect_while_loading_holds_reservation_until_settled(arb, gpu, clock):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    a.register("m", 20)
    await granted(arb, a, a.acquire("m"))
    gpu.allocate(100, "5GiB")  # partway through the load
    a.disconnect()
    b.register("n", 10)
    rid = b.acquire("n")
    await pending(arb, b, rid)
    clock.advance(arb.settings.settle_timeout_s)  # deadline: give up waiting for the driver
    await arb.quiesce()
    # 24 - 5 (still held by the stuck process) = 19 free: fits now.
    await granted(arb, b, rid)


async def test_disconnect_drops_queued_requests(arb, gpu):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    a.register("m", 20)
    await granted(arb, a, a.acquire("m"))
    a.load("m", 20)
    b.register("n", 10)
    b.acquire("n")
    await arb.quiesce()
    b.disconnect()
    assert arb.queue == []


async def test_reregister_resident_with_leases_is_believed(arb, gpu):
    gpu.allocate(100, "8GiB")
    a = Sat(arb, gpu, "a", 100)
    reply = a.request(
        p.Register(model="m", vram_peak=10 * GiB, state="resident", vram_bytes=8 * GiB, leases=2)
    )
    assert isinstance(reply, p.Ok) and len(reply.leases) == 2
    model = arb.registry.models["a/m"]
    assert model.state_label == "busy"
    for lease in reply.leases:
        a.request(p.Release(lease=lease))
    assert arb.registry.models["a/m"].state_label == "resident"


# --------------------------------------------------------------------------- settling


async def test_evicted_memory_not_yet_returned_is_waited_for_not_overevicted(arb, gpu, clock):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    await resident(arb, a, "m1", 8)
    clock.advance(1)
    await resident(arb, a, "m2", 8)
    b.register("n", 15)
    rid = b.acquire("n")
    await arb.quiesce()
    [ev] = a.evicts()
    # Report evicted, but the driver has not returned the pages yet.
    a.request(p.Evicted(model=ev.model, evict_id=ev.evict_id))
    await pending(arb, b, rid)
    assert len(a.evicts()) == 1, "must wait for the settle, not evict m2 as well"
    gpu.free(100, "8GiB")
    clock.advance(arb.settings.settle_poll_s)
    await granted(arb, b, rid)


async def test_residue_is_reported_when_memory_never_returns(arb, gpu, clock):
    a = Sat(arb, gpu, "a", 100)
    await resident(arb, a, "m", 8)
    ctl = Sat(arb, gpu, "cli", 1, role="control")
    ctl.request(p.EvictModel(model="a/m"))
    [ev] = a.evicts()
    a.request(p.Evicted(model="m", evict_id=ev.evict_id))  # but frees nothing
    clock.advance(arb.settings.settle_timeout_s)
    await arb.quiesce()
    sat = next(s for s in arb.status()["satellites"] if s["name"] == "a")
    assert sat["residue"] == {"0": 8 * GiB}


# --------------------------------------------------------------------------- operator controls


async def test_pin_protects_and_unpin_releases(arb, gpu):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    ctl = Sat(arb, gpu, "cli", 1, role="control")
    await resident(arb, a, "m", 20)
    assert isinstance(ctl.request(p.Pin(model="a/m")), p.Ok)
    b.register("n", 10)
    err = await errored(arb, b, b.acquire("n"), "vram_unavailable")
    assert "a/m (resident, pinned)" in err.message
    ctl.request(p.Unpin(model="a/m"))
    rid = b.acquire("n")
    await arb.quiesce()
    [ev] = a.evicts()
    a.evicted(ev)
    await granted(arb, b, rid)


async def test_declared_pin_and_operator_unpin(arb, gpu):
    a = Sat(arb, gpu, "a", 100)
    await resident(arb, a, "m", 4, pinned=True)
    assert arb.registry.models["a/m"].pinned
    a.request(p.Unpin(model="m"))  # a satellite may address its own models by short name
    assert not arb.registry.models["a/m"].pinned


async def test_profiles_pin_by_pattern(gpu, host, clock):
    settings = ArbiterSettings(headroom=0, profiles={"party": ["tts/*", "llm/gemma"]})
    arb = Arbiter(gpu, host, settings=settings, clock=clock)
    await arb.start()
    try:
        tts = Sat(arb, gpu, "tts", 1)
        tts.register("voice", 1)
        ctl = Sat(arb, gpu, "cli", 2, role="control")
        assert isinstance(ctl.request(p.Profile(name="party")), p.Ok)
        assert arb.registry.models["tts/voice"].pinned
        tts.register("late", 1)  # registered after activation: pattern still applies
        assert arb.registry.models["tts/late"].pinned
        reply = ctl.request(p.Profile(name="nope"))
        assert isinstance(reply, p.Error) and "party" in reply.message
        ctl.request(p.Profile(name=""))
        assert not arb.registry.models["tts/voice"].pinned
    finally:
        await arb.close()


async def test_operator_evict(arb, gpu):
    a = Sat(arb, gpu, "a", 100)
    ctl = Sat(arb, gpu, "cli", 1, role="control")
    await resident(arb, a, "m", 4)
    g = await granted(arb, a, a.acquire("m"))
    reply = ctl.request(p.EvictModel(model="a/m"))
    assert isinstance(reply, p.Error) and "busy" in reply.message
    a.request(p.Release(lease=g.lease))
    assert isinstance(ctl.request(p.EvictModel(model="a/m")), p.Ok)
    [ev] = a.evicts()
    assert ev.reason == "requested by cli"


async def test_voluntary_unload(arb, gpu, clock):
    a = Sat(arb, gpu, "a", 100)
    await resident(arb, a, "m", 4)
    gpu.free(100)
    assert isinstance(a.request(p.Unloaded(model="m")), p.Ok)
    assert arb.registry.models["a/m"].phase.value == "unloaded"


async def test_cancel_queued_acquire(arb, gpu):
    a = Sat(arb, gpu, "a", 100)
    b = Sat(arb, gpu, "b", 200)
    a.register("m", 20)
    await granted(arb, a, a.acquire("m"))
    a.load("m", 20)
    b.register("n", 10)
    rid = b.acquire("n")
    await arb.quiesce()
    assert isinstance(b.request(p.Cancel(request=rid)), p.Ok)
    err = await errored(arb, b, rid, "arbiter_error")
    assert err.message == "cancelled"
    assert arb.queue == []


async def test_status_reply(arb, gpu, clock):
    a = Sat(arb, gpu, "a", 100)
    await resident(arb, a, "m", 6, peak=8)
    gpu.allocate(4321, "1GiB")
    reply = a.request(p.StatusRequest())
    assert reply is None  # status is measured fresh, off the loop
    await arb.quiesce()
    for _ in range(100):
        reply = next((m for m in a.sent if isinstance(m, p.StatusReply)), None)
        if reply:
            break
        import asyncio

        await asyncio.sleep(0.01)
    assert reply is not None
    assert reply.devices[0]["total"] == 24 * GiB
    assert reply.models[0]["id"] == "a/m" and reply.models[0]["state"] == "resident"
    assert reply.models[0]["measured"] == 6 * GiB
    assert reply.foreign == [{"pid": 4321, "device": 0, "bytes": GiB}]
    assert reply.satellites[0]["usage"] == {"0": 6 * GiB}
    assert reply.arbiter["version"]


async def test_measurement_mismatch_is_logged_not_fatal(arb, gpu, caplog):
    a = Sat(arb, gpu, "a", 100)
    a.register("m", 10, resident=5)
    g = await granted(arb, a, a.acquire("m"))
    a.load("m", 9)
    a.request(p.Release(lease=g.lease))
    await arb.quiesce()
    assert arb.registry.models["a/m"].measured_bytes == 9 * GiB
    assert "mismatch > 20%" in caplog.text


async def test_baseline_learned_after_unload_settles(arb, gpu, clock):
    """What a satellite holds with nothing resident (its CUDA context) is nobody's residency."""
    a = Sat(arb, gpu, "a", 100)
    gpu.allocate(100, "1GiB")  # CUDA context
    await resident(arb, a, "m", 8)
    assert arb.registry.models["a/m"].measured_bytes == 9 * GiB  # no baseline known yet
    ctl = Sat(arb, gpu, "cli", 1, role="control")
    ctl.request(p.EvictModel(model="a/m"))
    [ev] = a.evicts()
    a.evicted(ev, free=8)
    await arb.quiesce()
    assert arb.registry.satellites["a"].baseline == {0: GiB}
    g = await granted(arb, a, a.acquire("m"))
    a.load("m", 8)
    a.request(p.Release(lease=g.lease))
    await arb.quiesce()
    assert arb.registry.models["a/m"].measured_bytes == 8 * GiB


async def test_reconnecting_satellite_usage_is_not_mistaken_for_baseline(arb, gpu):
    gpu.allocate(100, "8GiB")  # holds a model from before the arbiter restarted
    a = Sat(arb, gpu, "a", 100)
    await arb.quiesce()  # a planning pass between hello and register
    a.request(p.Register(model="m", vram_peak=8 * GiB, state="resident", vram_bytes=8 * GiB))
    await arb.quiesce()
    assert arb.registry.models["a/m"].measured_bytes == 8 * GiB


async def test_satellite_invisible_to_nvml_keeps_declared_sizes(arb, gpu, caplog):
    """Zero usage for a process with resident models means NVML cannot see it (containers)."""
    a = Sat(arb, gpu, "a", 100)
    a.register("m", 10, resident=8)
    g = await granted(arb, a, a.acquire("m"))
    a.request(p.Loaded(model="m"))  # loaded, but nothing shows up under pid 100
    a.request(p.Release(lease=g.lease))
    await arb.quiesce()
    model = arb.registry.models["a/m"]
    assert model.measured_bytes is None and model.resident_estimate() == 8 * GiB
    assert "NVML shows no GPU memory for satellite a" in caplog.text


async def test_reconnect_to_same_arbiter_drops_stale_settles(arb, gpu):
    """A satellite whose socket dropped but whose process lives on comes back holding its
    model: nothing is 'returning', so nothing should be waited for."""
    a = Sat(arb, gpu, "a", 100)
    await resident(arb, a, "m1", 8)
    await resident(arb, a, "m2", 8)
    a.disconnect()
    assert len(arb.registry.settling) == 1  # one settle for the whole process
    assert next(iter(arb.registry.settling.values())).expected == 16 * GiB
    again = Sat(arb, gpu, "a", 100)
    for name in ("m1", "m2"):
        again.request(p.Register(model=name, vram_peak=8 * GiB, state="resident"))
    assert arb.registry.settling == {}
    b = Sat(arb, gpu, "b", 200)
    b.register("n", 10)  # 8 free: one of a's models must go
    rid = b.acquire("n")
    await arb.quiesce()
    [ev] = again.evicts()  # evicts straight away instead of waiting on a phantom return
    again.evicted(ev, free=8)
    await granted(arb, b, rid)
