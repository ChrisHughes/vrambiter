from __future__ import annotations

from vrambiter.gpu import GpuSnapshot
from vrambiter.policy import Phase
from vrambiter.state import (
    ModelRecord,
    Registry,
    SatelliteRecord,
    Settle,
    attribute_usage,
    build_snapshot,
    request_for,
    satellite_reservation,
)
from vrambiter.units import GiB


def model(name="m", peak=10, resident=None, phase=Phase.UNLOADED, leases=0, sat="s", **kw):
    m = ModelRecord(
        satellite=sat,
        name=name,
        vram_peak=peak * GiB,
        vram_resident=resident * GiB if resident is not None else None,
        phase=phase,
        **kw,
    )
    m.leases = {f"L{name}{i}" for i in range(leases)}
    return m


# --------------------------------------------------------------------------- estimates and pins


def test_resident_estimate_prefers_measurement_then_report_then_declaration():
    m = model(peak=34, resident=30)
    assert m.resident_estimate() == 30 * GiB
    m.reported_bytes = 29 * GiB
    assert m.resident_estimate() == 29 * GiB
    m.measured_bytes = int(29.7 * GiB)
    assert m.resident_estimate() == int(29.7 * GiB)
    assert model(peak=34).resident_estimate() == 34 * GiB
    assert m.working_headroom() == 34 * GiB - int(29.7 * GiB)


def test_pin_precedence():
    m = model()
    assert not m.pinned
    m.profile_pin = True
    assert m.pinned
    m.manual_pin = False  # operator unpin beats profile and declaration
    assert not m.pinned
    m.manual_pin = None
    m.profile_pin = False
    m.declared_pinned = True
    assert m.pinned


def test_state_label():
    assert model(phase=Phase.RESIDENT, leases=1).state_label == "busy"
    assert model(phase=Phase.RESIDENT).state_label == "resident"
    assert model(phase=Phase.EVICTING).state_label == "evicting"


# --------------------------------------------------------------------------- requests


def test_request_for_each_phase():
    unloaded = model(peak=10, host_peak=3 * GiB)
    r = request_for(unloaded, seq=4)
    assert r.is_load and r.need == 10 * GiB and r.host_peak == 3 * GiB and r.seq == 4

    idle = model(peak=17, resident=7, phase=Phase.RESIDENT)
    r = request_for(idle)
    assert not r.is_load and r.need == 10 * GiB  # the working headroom, like a warm SDXL pass

    busy = model(peak=17, resident=7, phase=Phase.RESIDENT, leases=1)
    assert request_for(busy).need == 0

    assert request_for(model(phase=Phase.LOADING)) is None
    assert request_for(model(phase=Phase.EVICTING)) is None


# --------------------------------------------------------------------------- reservations


def test_reservation_envelope_shrinks_as_a_load_progresses():
    idle = model("a", peak=10, resident=8, phase=Phase.RESIDENT)
    loading = model("b", peak=10, phase=Phase.LOADING, leases=1)
    # Nothing of b allocated yet: 8 (a) in use, envelope 18 -> reserve 10.
    assert satellite_reservation([idle, loading], 8 * GiB) == 10 * GiB
    # 3 GiB of b allocated.
    assert satellite_reservation([idle, loading], 11 * GiB) == 7 * GiB
    # Fully loaded (and then some): nothing left to reserve, never negative.
    assert satellite_reservation([idle, loading], 20 * GiB) == 0


def test_reservation_never_exceeds_outstanding_peaks():
    # A process holding less than its idle models' estimates must not inflate the reservation.
    idle = model("a", peak=10, resident=8, phase=Phase.RESIDENT)
    loading = model("b", peak=10, phase=Phase.LOADING, leases=1)
    assert satellite_reservation([idle, loading], 0) == 10 * GiB


def test_reservation_for_busy_model_is_its_working_headroom():
    busy = model("a", peak=17, resident=7, phase=Phase.RESIDENT, leases=1)
    assert satellite_reservation([busy], 7 * GiB) == 10 * GiB
    assert satellite_reservation([busy], 15 * GiB) == 2 * GiB  # activations allocated
    assert satellite_reservation([busy], None) == 10 * GiB


def test_reservation_without_usage_reserves_full_outstanding():
    loading = model("b", peak=10, phase=Phase.LOADING, leases=1)
    idle = model("a", peak=10, resident=8, phase=Phase.RESIDENT)
    assert satellite_reservation([idle, loading], None) == 10 * GiB
    assert satellite_reservation([idle], None) == 0


# --------------------------------------------------------------------------- attribution


def test_attribution_single_model_takes_usage_minus_baseline():
    m = model("a", phase=Phase.RESIDENT)
    assert attribute_usage([m], 9 * GiB, baseline=GiB) == {"s/a": 8 * GiB}


def test_attribution_splits_by_report_or_declaration():
    a = model("a", peak=30, resident=20, phase=Phase.RESIDENT)
    b = model("b", peak=10, phase=Phase.RESIDENT)
    out = attribute_usage([a, b], 30 * GiB)
    assert out == {"s/a": 20 * GiB, "s/b": 10 * GiB}


def test_attribution_skipped_while_busy_or_loading_or_unknown():
    a = model("a", phase=Phase.RESIDENT, leases=1)
    assert attribute_usage([a], 9 * GiB) == {}
    b = model("b", phase=Phase.LOADING)
    c = model("c", phase=Phase.RESIDENT)
    assert attribute_usage([b, c], 9 * GiB) == {}
    assert attribute_usage([c], None) == {}
    assert attribute_usage([model("d")], 5 * GiB) == {}


# --------------------------------------------------------------------------- settling


def gpu(free, usage=None, total=96):
    return GpuSnapshot(free={0: free * GiB}, total={0: total * GiB}, usage={0: usage})


def test_settle_by_process_usage():
    s = Settle(
        "S1",
        "sat",
        0,
        frozenset({7}),
        expected=10 * GiB,
        usage_before=12 * GiB,
        free_before=0,
        deadline=100,
    )
    assert not s.settled(gpu(0, {7: 12 * GiB}), now=0)
    assert s.outstanding(gpu(0, {7: 12 * GiB})) == 10 * GiB
    assert s.outstanding(gpu(0, {7: 7 * GiB})) == 5 * GiB
    assert s.settled(gpu(0, {7: 4 * GiB}), now=0)  # 8 of 10 back
    assert s.settled(gpu(0, {}), now=0)  # process gone entirely


def test_settle_by_free_memory_when_usage_unknown():
    s = Settle(
        "S1",
        "sat",
        0,
        frozenset({7}),
        expected=10 * GiB,
        usage_before=None,
        free_before=20 * GiB,
        deadline=100,
    )
    assert not s.settled(gpu(25, None), now=0)
    assert s.settled(gpu(28, None), now=0)


def test_reserve_only_settle_waits_for_the_process_memory_to_vanish():
    s = Settle(
        "S1",
        "sat",
        0,
        frozenset({7}),
        expected=0,
        usage_before=0,
        free_before=0,
        deadline=100,
        reserve=10 * GiB,
    )
    assert not s.settled(gpu(0, {7: 5 * GiB}), now=0)
    assert s.settled(gpu(0, {}), now=0)
    assert s.settled(gpu(0, {7: 5 * GiB}), now=100)


def test_settle_deadline():
    s = Settle(
        "S1",
        "sat",
        0,
        frozenset({7}),
        expected=10 * GiB,
        usage_before=12 * GiB,
        free_before=0,
        deadline=100,
    )
    assert s.settled(gpu(0, {7: 12 * GiB}), now=100)


# --------------------------------------------------------------------------- snapshots


def test_build_snapshot_counts_reservations_pending_and_foreign():
    reg = Registry()
    reg.satellites["s"] = SatelliteRecord("s", pid=100, conn_id=1)
    reg.satellites["llama"] = SatelliteRecord("llama", pid=200, pids=frozenset({200, 201}))
    loading = model("load", peak=10, phase=Phase.LOADING, leases=1, host_peak=4 * GiB)
    evict = model("ev", peak=8, resident=6, phase=Phase.EVICTING)
    router_model = model("g", peak=30, phase=Phase.RESIDENT, sat="llama", pids=frozenset({201}))
    router_model.measured_bytes = 29 * GiB
    for m in (loading, evict, router_model):
        reg.models[m.id] = m
    reg.settling["S1"] = Settle(
        "S1",
        "s",
        0,
        frozenset({100}),
        expected=4 * GiB,
        usage_before=None,
        free_before=10 * GiB,
        deadline=1e9,
        reserve=GiB,
    )
    g = GpuSnapshot(
        free={0: 10 * GiB},
        total={0: 96 * GiB},
        usage={0: {100: 9 * GiB, 201: 29 * GiB, 999: 20 * GiB}},
    )
    s = build_snapshot(
        reg, g, host_available=20 * GiB, headroom=2 * GiB, host_headroom=0, max_concurrent_loads=1
    )
    dev = s.devices[0]
    # s: envelope = 10 (loading peak) + 6 (evicting) = 16, usage 9 -> 7, capped by outstanding 10
    # plus the settle's 1 GiB reserve.
    assert dev.reserved == 7 * GiB + GiB
    # pending: evicting 6 + settle outstanding 4 (free has not risen) + its 1 GiB reserve
    assert dev.pending == 6 * GiB + 4 * GiB + GiB
    assert [(f.pid, f.bytes) for f in s.foreign] == [(999, 20 * GiB)]
    assert s.loads_in_flight == 1 and s.host_reserved == 4 * GiB
    assert {m.id for m in s.models} == {"s/load", "s/ev", "llama/g"}
    assert s.model("s/load").resident_bytes == 10 * GiB


def test_registry_resolve_and_leases():
    reg = Registry()
    own = model("voice", sat="tts")
    other = model("gemma", sat="llama")
    reg.models[own.id] = own
    reg.models[other.id] = other
    assert reg.resolve("voice", "tts") is own
    assert reg.resolve("llama/gemma", "tts") is other
    assert reg.resolve("voice", "llama") is None
    lease = reg.add_lease(own, "tts", 1, now=5.0, action="load")
    assert own.leases == {lease.id} and own.last_used == 5.0
    assert reg.end_lease(lease.id, now=9.0) is lease
    assert own.leases == set() and own.last_used == 9.0
    assert reg.end_lease(lease.id, now=10.0) is None
    assert reg.next_evict_id() != reg.next_evict_id()


def test_reservation_includes_the_learned_baseline():
    """The CUDA context is in the process's usage but in no model's estimate."""
    loading = model("x", peak=10, phase=Phase.LOADING, leases=1)
    # 1 GiB context + 3 GiB of x loaded so far: 7 GiB of x still to come.
    assert satellite_reservation([loading], 4 * GiB, baseline=GiB) == 7 * GiB
    assert satellite_reservation([loading], 4 * GiB) == 6 * GiB  # what it used to be: short
