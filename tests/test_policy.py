"""Table tests for every admission rule in docs/DESIGN.md ("Admission policy").

Sizes are in GiB throughout to keep the tables readable; ``G(n)`` converts.
"""

from __future__ import annotations

import pytest

from vrambiter.errors import HostRamUnavailable, VramUnavailable
from vrambiter.policy import (
    Admit,
    DeviceView,
    Evict,
    Fail,
    ForeignView,
    ModelView,
    Phase,
    Request,
    Snapshot,
    Wait,
    WaitKind,
    holders,
    plan,
    plan_queue,
    victim_order,
)
from vrambiter.units import GiB


def G(n: float) -> int:
    return int(n * GiB)


def snap(
    *models: ModelView,
    free: float = 24,
    total: float = 96,
    reserved: float = 0,
    pending: float = 0,
    headroom: float = 0,
    devices: dict[int, DeviceView] | None = None,
    **kw,
) -> Snapshot:
    devs = devices or {
        0: DeviceView(0, free=G(free), total=G(total), reserved=G(reserved), pending=G(pending))
    }
    return Snapshot(devices=devs, models=models, headroom=G(headroom), **kw)


def idle(mid: str, size: float, last_used: float = 0, priority: int = 0, device: int = 0, **kw):
    return ModelView(
        id=mid,
        device=device,
        phase=Phase.RESIDENT,
        resident_bytes=G(size),
        last_used=last_used,
        priority=priority,
        **kw,
    )


def busy(mid: str, size: float, device: int = 0, **kw):
    return ModelView(
        id=mid, device=device, phase=Phase.RESIDENT, leases=1, resident_bytes=G(size), **kw
    )


def loading(mid: str, size: float, device: int = 0):
    return ModelView(id=mid, device=device, phase=Phase.LOADING, leases=1, resident_bytes=G(size))


def evicting(mid: str, size: float, device: int = 0):
    return ModelView(id=mid, device=device, phase=Phase.EVICTING, resident_bytes=G(size))


def load(mid: str = "app/new", need: float = 10, priority: int = 0, host_peak: float = 0, **kw):
    return Request(
        model_id=mid,
        device=0,
        need=G(need),
        priority=priority,
        is_load=True,
        host_peak=G(host_peak),
        **kw,
    )


def lease(mid: str = "app/res", need: float = 2, priority: int = 0, **kw):
    return Request(model_id=mid, device=0, need=G(need), priority=priority, is_load=False, **kw)


# --------------------------------------------------------------------------- rule 0: never fits


def test_rule0_request_larger_than_the_card_fails_immediately():
    s = snap(busy("a/busy", 10), total=80, headroom=2, free=70)
    decision = plan(load(need=79), s)
    assert isinstance(decision, Fail)
    assert "can ever offer" in decision.reason
    # Even with a busy model that will finish: waiting cannot help.


def test_rule0_boundary_exactly_capacity_is_not_rule0():
    s = snap(total=80, headroom=2, free=80)
    assert plan(load(need=78), s) == Admit()
    assert isinstance(plan(load(need=78.01), s), Fail)


def test_unknown_device_fails():
    decision = plan(Request(model_id="a/x", device=3, need=1), snap())
    assert isinstance(decision, Fail) and "no GPU 3" in decision.reason


# --------------------------------------------------------------------------- rule 1: host gate


def test_rule1_load_slot_taken_waits():
    s = snap(loading("a/l", 5), free=50, loads_in_flight=1, max_concurrent_loads=1)
    decision = plan(load(need=1), s)
    assert decision == Wait(WaitKind.LOAD_SLOT, decision.reason, ("a/l",))


def test_rule1_load_slot_does_not_gate_leases_on_resident_models():
    s = snap(loading("a/l", 5), free=50, loads_in_flight=1, max_concurrent_loads=1)
    assert plan(lease(need=1), s) == Admit()


def test_rule1_two_slots_allow_second_load():
    s = snap(loading("a/l", 5), free=50, loads_in_flight=1, max_concurrent_loads=2)
    assert plan(load(need=1), s) == Admit()


def test_rule1_host_ram_short_with_load_in_flight_waits():
    s = snap(
        loading("a/l", 5),
        free=50,
        loads_in_flight=1,
        max_concurrent_loads=2,
        host_available=G(20),
        host_headroom=G(3),
        host_reserved=G(8),
    )
    # 20 - 3 - 8 = 9 GiB of host RAM for this load.
    decision = plan(load(need=1, host_peak=10), s)
    assert isinstance(decision, Wait) and decision.kind is WaitKind.HOST_RAM
    assert plan(load(need=1, host_peak=9), s) == Admit()


def test_rule1_host_ram_short_with_nothing_in_flight_fails_as_host_error():
    s = snap(free=50, host_available=G(10), host_headroom=G(3))
    decision = plan(load(need=1, host_peak=8), s)
    assert isinstance(decision, Fail) and decision.host
    err = decision.to_error("app/new", 0)
    assert isinstance(err, HostRamUnavailable) and isinstance(err, VramUnavailable)
    assert err.need == G(8) and err.free == G(7)


def test_rule1_host_check_skipped_when_unknown_or_zero():
    assert plan(load(need=1, host_peak=500), snap(host_available=None)) == Admit()
    assert plan(load(need=1, host_peak=0), snap(host_available=0)) == Admit()


def test_rule1_host_gate_before_eviction():
    # Eviction would make room, but the load could not start anyway: evict nothing yet.
    s = snap(idle("a/idle", 30), loading("b/l", 1), free=5, loads_in_flight=1)
    decision = plan(load(need=20), s)
    assert isinstance(decision, Wait) and decision.kind is WaitKind.LOAD_SLOT


# --------------------------------------------------------------------------- rule 2: admit


@pytest.mark.parametrize(
    ("need", "free", "reserved", "headroom", "admitted"),
    [
        (10, 10, 0, 0, True),
        (10.001, 10, 0, 0, False),
        (10, 14, 2, 2, True),
        (10, 14, 2, 2.5, False),
        (10, 24, 15, 0, False),
    ],
)
def test_rule2_effective_free_includes_reservations_and_headroom(
    need, free, reserved, headroom, admitted
):
    s = snap(free=free, reserved=reserved, headroom=headroom)
    assert (plan(load(need=need), s) == Admit()) is admitted


def test_zero_need_is_always_admitted_even_when_overcommitted():
    s = snap(free=1, reserved=5, headroom=2)
    assert plan(lease(need=0), s) == Admit()


# --------------------------------------------------------------------------- rule 3: pending


def test_rule3_waits_for_evictions_in_flight_instead_of_evicting_more():
    s = snap(evicting("a/e", 8), idle("b/idle", 30), free=4, pending=8)
    decision = plan(load(need=10), s)
    assert decision == Wait(WaitKind.EVICTING, decision.reason, ("a/e",))


def test_rule3_pending_shortfall_is_topped_up_by_eviction():
    s = snap(evicting("a/e", 4), idle("b/idle", 30), free=4, pending=4)
    # 4 free + 4 pending: still 2 short, so one more victim.
    assert plan(load(need=10), s) == Evict(("b/idle",), G(30))


# --------------------------------------------------------------------------- rule 4: victims


def test_rule4_lru_order_and_shortest_prefix():
    s = snap(
        idle("a/newest", 8, last_used=30),
        idle("a/oldest", 4, last_used=10),
        idle("a/middle", 4, last_used=20),
        free=2,
    )
    # need 9, free 2 -> shortfall 7: oldest (4) + middle (4) covers it; newest survives.
    assert plan(load(need=9), s) == Evict(("a/oldest", "a/middle"), G(8))


def test_rule4_lower_priority_goes_first_regardless_of_lru():
    s = snap(
        idle("a/important", 8, last_used=0, priority=5),
        idle("a/cheap", 8, last_used=100, priority=0),
        free=0,
    )
    assert plan(load(need=8, priority=5), s) == Evict(("a/cheap",), G(8))


def test_rule4_never_evicts_higher_priority():
    s = snap(idle("a/vip", 50, priority=9), busy("a/b", 1), free=0)
    decision = plan(load(need=8, priority=1), s)
    assert not isinstance(decision, Evict)
    # ...and since the 1 GiB busy model is all that could ever join, it fails rather than waits.
    assert isinstance(decision, Fail)
    assert isinstance(plan(load(need=1, priority=1), s), Wait)


def test_rule4_equal_priority_is_eligible():
    s = snap(idle("a/peer", 10, priority=3), free=0)
    assert plan(load(need=8, priority=3), s) == Evict(("a/peer",), G(10))


@pytest.mark.parametrize(
    "model",
    [
        busy("x/busy", 50),
        idle("x/pinned", 50, pinned=True),
        idle("x/unresponsive", 50, unresponsive=True),
        idle("x/protected", 50, protected=True),
        idle("x/other-device", 50, device=1),
        loading("x/loading", 50),
        evicting("x/evicting", 50),
        ModelView(id="x/unloaded", device=0, phase=Phase.UNLOADED, resident_bytes=G(50)),
        idle("x/zero", 0),
    ],
    ids=lambda m: m.id,
)
def test_rule4_ineligible_victims(model):
    devices = {0: DeviceView(0, free=0, total=G(96)), 1: DeviceView(1, free=0, total=G(96))}
    s = snap(model, devices=devices)
    decision = plan(load(need=8), s)
    assert not isinstance(decision, Evict)


def test_rule4_never_evicts_the_requested_model_itself():
    s = snap(idle("app/res", 10), free=0)
    decision = plan(lease("app/res", need=4), s)
    assert isinstance(decision, Fail)


def test_victim_order_is_total_and_deterministic():
    a = idle("b/x", 1, last_used=5)
    b = idle("a/x", 1, last_used=5)
    c = idle("c/x", 1, last_used=1, priority=1)
    assert [m.id for m in victim_order([c, a, b])] == ["a/x", "b/x", "c/x"]


# --------------------------------------------------------------------------- rule 5: wait or fail


def test_rule5_not_enough_even_evicting_all_but_busy_model_waits_without_evicting():
    s = snap(idle("a/idle", 4), busy("b/busy", 30), free=2)
    decision = plan(load(need=20), s)
    assert decision == Wait(WaitKind.BUSY, decision.reason, ("b/busy",))


def test_rule5_load_in_flight_counts_as_could_free_later():
    s = snap(loading("b/l", 30), free=2, loads_in_flight=1, max_concurrent_loads=2)
    decision = plan(load(need=20), s)
    assert isinstance(decision, Wait) and decision.blockers == ("b/l",)


def test_rule5_protected_model_counts_as_could_free_later():
    s = snap(idle("a/just-refused", 30, protected=True), free=2)
    decision = plan(load(need=20), s)
    assert isinstance(decision, Wait) and decision.blockers == ("a/just-refused",)


def test_rule5_evicting_model_or_returning_memory_counts_as_in_flight():
    s = snap(evicting("a/e", 4), busy("b/x", 10), free=2, pending=4)
    decision = plan(load(need=12), s)
    assert isinstance(decision, Wait) and "a/e" in decision.blockers
    # A load that was in flight when its satellite vanished: 5 GiB reserved and returning.
    s = snap(free=7, reserved=5, pending=5)
    decision = plan(load(need=8), s)
    assert isinstance(decision, Wait) and decision.kind is WaitKind.EVICTING
    assert isinstance(plan(load(need=20), s), Fail)  # more than could ever come back


def test_queue_request_behind_waiting_requests_waits_instead_of_failing():
    # vip evicts the hog and earmarks half of it, first earmarks the rest; second must queue,
    # not fail, even though nothing is left for it in this pass.
    s = snap(idle("h/hog", 24), free=0)
    out = plan_queue(
        [
            load("a/vip", need=12, priority=5, seq=1),
            load("a/first", need=12, seq=2),
            load("a/second", need=12, seq=3),
        ],
        s,
    )
    assert out[0][1] == Evict(("h/hog",), G(24))
    assert isinstance(out[1][1], Wait)
    assert isinstance(out[2][1], Wait), out[2][1]


def test_queue_never_fits_still_fails_behind_others():
    s = snap(busy("h/hog", 20), free=4, total=24)
    out = plan_queue([load("a/x", need=10, seq=1), load("a/huge", need=30, seq=2)], s)
    assert isinstance(out[1][1], Fail)


def test_rule5_busy_higher_priority_model_cannot_help_so_fail_fast():
    # The vip model is busy and will never be evictable by a priority-0 request; its working
    # headroom (nothing reserved here) would not cover the need either. Waiting would only time out.
    s = snap(busy("a/vip", 20, priority=5), free=2, total=24)
    decision = plan(load(need=10, priority=0), s)
    assert isinstance(decision, Fail) and "may not evict" in decision.reason


def test_rule5_busy_higher_priority_model_whose_headroom_would_cover_means_wait():
    # 8 GiB reserved as the busy vip model's working headroom comes back when its lease ends.
    s = snap(busy("a/vip", 10, priority=5), free=10, reserved=8, total=24)
    decision = plan(load(need=6, priority=0), s)
    assert isinstance(decision, Wait) and decision.blockers == ("a/vip",)


def test_rule5_busy_equal_priority_model_means_wait():
    s = snap(busy("a/peer", 20, priority=0), free=2, total=24)
    assert isinstance(plan(load(need=10, priority=0), s), Wait)


def test_rule5_busy_model_on_another_device_does_not_count():
    devices = {0: DeviceView(0, free=G(2), total=G(96)), 1: DeviceView(1, free=0, total=G(96))}
    s = snap(busy("b/busy", 30, device=1), devices=devices)
    assert isinstance(plan(load(need=20), s), Fail)


def test_rule5_empty_registry_fails_fast_naming_foreign_holders():
    # The production bug: nothing evictable, nothing busy, and the old arbiter waited forever.
    s = snap(
        free=10,
        total=96,
        foreign=(ForeignView(pid=4321, device=0, bytes=G(80)),),
    )
    decision = plan(load(need=20), s)
    assert isinstance(decision, Fail)
    err = decision.to_error("app/new", 0)
    assert isinstance(err, VramUnavailable)
    assert err.holders[0].name == "pid 4321" and err.holders[0].kind == "foreign"
    assert "pid 4321" in str(err)
    # 96 total - 10 free - 80 foreign = 6 GiB nobody claims
    assert any(h.kind == "unattributed" and h.bytes == G(6) for h in err.holders)


def test_rule5_fail_names_pinned_and_higher_priority_models():
    s = snap(idle("a/pinned", 40, pinned=True), idle("a/vip", 40, priority=9), free=4, total=96)
    decision = plan(load(need=20), s)
    assert isinstance(decision, Fail)
    names = {h.name: h for h in decision.holders}
    assert names["a/pinned"].pinned
    assert names["a/vip"].priority == 9


def test_holders_sorted_largest_first_and_skip_unloaded():
    s = snap(
        idle("a/small", 1),
        busy("a/big", 30),
        ModelView(id="a/gone", device=0, phase=Phase.UNLOADED, resident_bytes=G(9)),
        free=60,
        total=96,
    )
    hs = holders(s, 0)
    assert [h.name for h in hs] == ["a/big", "unattributed", "a/small"]
    assert hs[0].state == "busy"


# --------------------------------------------------------------------------- plan_queue


def test_queue_orders_by_priority_then_fifo():
    s = snap(free=100, total=200)
    reqs = [
        lease("a/1", need=1, seq=1),
        lease("a/2", need=1, seq=2, priority=5),
        lease("a/3", need=1, seq=0),
    ]
    assert [r.model_id for r, _ in plan_queue(reqs, s)] == ["a/2", "a/3", "a/1"]


def test_queue_admissions_accumulate_reservations():
    s = snap(free=10)
    out = plan_queue([lease("a/1", need=6, seq=1), lease("a/2", need=6, seq=2)], s)
    assert out[0][1] == Admit()
    assert not isinstance(out[1][1], Admit)


def test_queue_admitted_load_takes_the_slot():
    s = snap(free=50)
    out = plan_queue([load("a/1", need=1, seq=1), load("a/2", need=1, seq=2)], s)
    assert out[0][1] == Admit()
    assert isinstance(out[1][1], Wait) and out[1][1].kind is WaitKind.LOAD_SLOT


def test_queue_waiting_head_earmarks_room_so_later_requests_cannot_steal_it():
    # Head needs 20, 12 free, waits for the busy model. A later 10 GiB request must not take the
    # 12 GiB the head is counting on.
    s = snap(busy("b/busy", 30), free=12)
    out = plan_queue([lease("a/head", need=20, seq=1), lease("a/late", need=10, seq=2)], s)
    assert isinstance(out[0][1], Wait) and out[0][1].kind is WaitKind.BUSY
    assert not isinstance(out[1][1], Admit)


def test_queue_backfills_into_room_the_head_does_not_need():
    s = snap(busy("b/busy", 30), free=30)
    # Head needs 40: waits, earmarking all 30. Nothing left for a backfill.
    out = plan_queue([lease("a/head", need=40, seq=1), lease("a/small", need=2, seq=2)], s)
    assert not isinstance(out[1][1], Admit)
    # A head that can never use all the room leaves the rest: 50 free, head needs 60 - but
    # earmarks only up to its need, so with 70 free the head is admitted outright.
    s2 = snap(busy("b/busy", 30), free=70)
    out = plan_queue([lease("a/head", need=60, seq=1), lease("a/small", need=2, seq=2)], s2)
    assert [d for _, d in out] == [Admit(), Admit()]


def test_queue_host_blocked_load_blocks_later_loads_but_not_leases():
    s = snap(
        loading("x/l", 1),
        free=50,
        host_available=G(10),
        loads_in_flight=1,
        max_concurrent_loads=3,
    )
    out = plan_queue(
        [
            load("a/big", need=1, host_peak=20, seq=1),
            load("a/small", need=1, host_peak=1, seq=2),
            lease("a/res", need=1, seq=3),
        ],
        s,
    )
    assert out[0][1].kind is WaitKind.HOST_RAM
    assert out[1][1].kind is WaitKind.QUEUED
    assert out[2][1] == Admit()


def test_queue_second_request_for_same_unloaded_model_waits_for_the_first():
    s = snap(free=50, max_concurrent_loads=4)
    out = plan_queue([load("a/m", need=5, seq=1), load("a/m", need=5, seq=2)], s)
    assert out[0][1] == Admit()
    assert out[1][1].kind is WaitKind.QUEUED


def test_queue_second_lease_on_busy_model_needs_nothing_more():
    s = snap(idle("a/res", 8), free=3)
    out = plan_queue([lease("a/res", need=2, seq=1), lease("a/res", need=2, seq=2)], s)
    assert [d for _, d in out] == [Admit(), Admit()]


def test_queue_evictions_are_not_chosen_twice():
    s = snap(idle("v/1", 10, last_used=1), idle("v/2", 10, last_used=2), free=0)
    out = plan_queue([load("a/x", need=10, seq=1), lease("a/y", need=10, seq=2)], s)
    assert out[0][1] == Evict(("v/1",), G(10))
    assert out[1][1] == Evict(("v/2",), G(10))


def test_queue_admitted_model_is_not_a_victim_later_in_the_pass():
    s = snap(idle("a/res", 10), free=2, max_concurrent_loads=2)
    out = plan_queue([lease("a/res", need=2, seq=1), load("b/new", need=10, seq=2)], s)
    assert out[0][1] == Admit()
    assert not isinstance(out[1][1], Evict)


def test_queue_never_leases_a_model_chosen_as_victim_in_the_same_pass():
    # X's load evicts Y; a lease request for Y later in the pass must not be told "ready".
    s = snap(idle("drv/Y", 10), free=0, total=10)
    out = plan_queue([load("drv/X", need=10, seq=1), lease("drv/Y", need=0, seq=2)], s)
    assert out[0][1] == Evict(("drv/Y",), G(10))
    assert isinstance(out[1][1], Wait) and out[1][1].kind is WaitKind.QUEUED


def test_queue_two_requests_for_one_unloaded_model_evict_once():
    s = snap(idle("v/y", 10, last_used=1), idle("v/z", 10, last_used=2), free=4, total=24)
    out = plan_queue([load("s/x", need=10, seq=1), load("s/x", need=10, seq=2)], s)
    assert out[0][1] == Evict(("v/y",), G(10))
    assert isinstance(out[1][1], Wait) and out[1][1].kind is WaitKind.QUEUED
