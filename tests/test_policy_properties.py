"""Property tests for the planner: invariants that must hold for *any* snapshot.

* never evict a busy, pinned, unresponsive, higher-priority or non-resident model;
* never admit beyond effective free memory (reservations and headroom included), alone or summed
  over a queue pass;
* victims are the shortest prefix of the (priority, last_used, id) order covering the shortfall;
* equal inputs give equal plans, whatever order the models arrive in;
* Fail only when nothing in flight could help, Wait(BUSY) only when something could.
"""

from __future__ import annotations

from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st

from vrambiter.policy import (
    Admit,
    DeviceView,
    Evict,
    Fail,
    ModelView,
    Phase,
    Request,
    Snapshot,
    Wait,
    WaitKind,
    eligible_victims,
    plan,
    plan_queue,
)
from vrambiter.units import GiB, MiB

SIZE = st.integers(min_value=0, max_value=48 * GiB // MiB).map(lambda n: n * MiB)


@st.composite
def models(draw, device_count: int):
    count = draw(st.integers(min_value=0, max_value=9))
    out = []
    for i in range(count):
        phase = draw(st.sampled_from(list(Phase)))
        leases = draw(st.integers(min_value=0, max_value=2))
        out.append(
            ModelView(
                id=f"s{draw(st.integers(0, 2))}/m{i}",
                device=draw(st.integers(0, device_count - 1)),
                phase=phase,
                leases=leases if phase in (Phase.RESIDENT, Phase.LOADING) else 0,
                pinned=draw(st.booleans()),
                priority=draw(st.integers(0, 3)),
                last_used=float(draw(st.integers(0, 20))),
                resident_bytes=draw(SIZE),
                unresponsive=draw(st.integers(0, 5)) == 0,
                protected=draw(st.integers(0, 5)) == 0,
            )
        )
    return tuple(out)


@st.composite
def snapshots(draw):
    device_count = draw(st.integers(1, 2))
    devices = {}
    for d in range(device_count):
        total = draw(st.integers(8, 128)) * GiB
        devices[d] = DeviceView(
            index=d,
            free=draw(st.integers(0, total // MiB)) * MiB,
            total=total,
            reserved=draw(st.integers(0, 16 * GiB // MiB)) * MiB,
            pending=draw(st.sampled_from([0, 0, 2 * GiB, 9 * GiB])),
        )
    ms = draw(models(device_count))
    loading = sum(1 for m in ms if m.phase is Phase.LOADING)
    return Snapshot(
        devices=devices,
        models=ms,
        headroom=draw(st.sampled_from([0, GiB, 2 * GiB])),
        host_available=draw(st.one_of(st.none(), SIZE)),
        host_headroom=draw(st.sampled_from([0, 3 * GiB])),
        host_reserved=draw(st.sampled_from([0, 4 * GiB])),
        loads_in_flight=loading,
        max_concurrent_loads=draw(st.integers(1, 3)),
    )


@st.composite
def request_for(draw, snapshot: Snapshot, seq: int = 0, new_model: bool = False):
    ids = [m.id for m in snapshot.models]
    if ids and not new_model and draw(st.booleans()):
        mid = draw(st.sampled_from(ids))
    else:
        mid = f"req/r{seq}"
    return Request(
        model_id=mid,
        device=draw(st.sampled_from(sorted(snapshot.devices))),
        need=draw(SIZE),
        priority=draw(st.integers(0, 3)),
        is_load=draw(st.booleans()),
        host_peak=draw(st.sampled_from([0, GiB, 8 * GiB, 40 * GiB])),
        seq=seq,
    )


@st.composite
def cases(draw):
    s = draw(snapshots())
    return s, draw(request_for(s))


def _shortfall(r: Request, s: Snapshot) -> int:
    return r.need - (s.effective_free(r.device) + s.devices[r.device].pending)


def _host_ok(r: Request, s: Snapshot) -> bool:
    if not r.is_load:
        return True
    if s.loads_in_flight >= s.max_concurrent_loads:
        return False
    if s.host_available is None or r.host_peak <= 0:
        return True
    return r.host_peak <= s.host_available - s.host_headroom - s.host_reserved


@settings(max_examples=400, deadline=None)
@given(cases())
def test_victims_are_always_eligible(case):
    s, r = case
    decision = plan(r, s)
    if isinstance(decision, Evict):
        by_id = {m.id: m for m in s.models}
        for vid in decision.victims:
            v = by_id[vid]
            assert v.phase is Phase.RESIDENT and v.leases == 0, "never evict busy work"
            assert not v.pinned, "never evict pinned"
            assert not v.unresponsive and not v.protected
            assert v.device == r.device
            assert v.priority <= r.priority
            assert v.id != r.model_id
        assert len(set(decision.victims)) == len(decision.victims)


@settings(max_examples=400, deadline=None)
@given(cases())
def test_admit_never_exceeds_effective_free(case):
    s, r = case
    if isinstance(plan(r, s), Admit):
        assert r.need <= 0 or r.need <= s.effective_free(r.device)
        assert r.need <= s.devices[r.device].total - s.headroom
        assert _host_ok(r, s)


@settings(max_examples=400, deadline=None)
@given(cases())
def test_victims_are_the_shortest_covering_prefix_in_order(case):
    s, r = case
    decision = plan(r, s)
    if not isinstance(decision, Evict):
        return
    order = [m for m in eligible_victims(r, s)]
    k = len(decision.victims)
    assert list(decision.victims) == [m.id for m in order[:k]]
    sizes = [m.resident_bytes for m in order[:k]]
    shortfall = _shortfall(r, s)
    assert sum(sizes) >= shortfall
    assert sum(sizes[:-1]) < shortfall
    assert decision.expected == sum(sizes)


@settings(max_examples=300, deadline=None)
@given(cases(), st.randoms(use_true_random=False))
def test_plan_is_deterministic_and_order_independent(case, rnd):
    s, r = case
    shuffled = list(s.models)
    rnd.shuffle(shuffled)
    assert plan(r, s) == plan(r, s)
    assert plan(r, s) == plan(r, replace(s, models=tuple(shuffled)))


@settings(max_examples=400, deadline=None)
@given(cases())
def test_never_evicts_when_room_exists_or_is_coming(case):
    s, r = case
    decision = plan(r, s)
    if isinstance(decision, Evict):
        assert _shortfall(r, s) > 0
        assert _host_ok(r, s), "never evict for a load that could not start"


@settings(max_examples=400, deadline=None)
@given(cases())
def test_fail_and_wait_busy_are_justified(case):
    s, r = case
    decision = plan(r, s)
    could_free_later = [
        m
        for m in s.models
        if m.device == r.device
        and m.id != r.model_id
        and (m.busy or m.phase is Phase.LOADING or (m.idle and m.protected))
    ]
    eligible_total = sum(m.resident_bytes for m in eligible_victims(r, s))
    if isinstance(decision, Wait) and decision.kind is WaitKind.BUSY:
        assert could_free_later
        assert eligible_total < _shortfall(r, s)
    if isinstance(decision, Fail) and not decision.host:
        never_fits = r.need > s.devices[r.device].total - s.headroom
        assert never_fits or (not could_free_later and eligible_total < _shortfall(r, s))


@st.composite
def queue_cases(draw):
    s = draw(snapshots())
    n = draw(st.integers(1, 6))
    reqs = [draw(request_for(s, seq=i, new_model=True)) for i in range(n)]
    return s, reqs


@settings(max_examples=300, deadline=None)
@given(queue_cases())
def test_queue_never_admits_beyond_effective_free_in_total(case):
    s, reqs = case
    out = plan_queue(reqs, s)
    for device in s.devices:
        admitted = sum(r.need for r, d in out if isinstance(d, Admit) and r.device == device)
        positive = [r for r, d in out if isinstance(d, Admit) and r.device == device and r.need]
        if positive:
            assert admitted <= s.effective_free(device)
    loads = sum(1 for r, d in out if isinstance(d, Admit) and r.is_load)
    assert s.loads_in_flight + loads <= max(s.max_concurrent_loads, s.loads_in_flight)


@settings(max_examples=300, deadline=None)
@given(queue_cases())
def test_queue_victims_distinct_and_eligible(case):
    s, reqs = case
    out = plan_queue(reqs, s)
    by_id = {m.id: m for m in s.models}
    seen: set[str] = set()
    for r, d in out:
        if isinstance(d, Evict):
            for vid in d.victims:
                assert vid not in seen
                seen.add(vid)
                v = by_id[vid]
                assert v.idle and not v.pinned and not v.unresponsive and not v.protected
                assert v.priority <= r.priority


@settings(max_examples=200, deadline=None)
@given(queue_cases(), st.randoms(use_true_random=False))
def test_queue_is_deterministic_under_input_order(case, rnd):
    s, reqs = case
    shuffled = list(reqs)
    rnd.shuffle(shuffled)
    a = [(r.seq, d) for r, d in plan_queue(reqs, s)]
    b = [(r.seq, d) for r, d in plan_queue(shuffled, s)]
    assert a == b
