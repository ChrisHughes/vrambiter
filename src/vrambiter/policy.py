"""Admission and victim selection: a pure function of a snapshot.

``plan(request, snapshot)`` answers one question: *can this request be admitted now, and if not,
what should happen?* It returns exactly one of

* :class:`Admit` - there is room; grant it.
* :class:`Evict` - evicting these idle models would make room; evict them, then plan again once
  the evictions are confirmed.
* :class:`Wait` - nothing can be done right now, but something in flight (a busy model, a load, an
  eviction) will change the picture; queue the request.
* :class:`Fail` - nothing in flight could ever make room; fail fast with a structured error.

There is no I/O, no clock and no mutable state here, so every rule is table-tested and
property-tested (see ``tests/test_policy*.py``). The daemon builds a :class:`Snapshot` from NVML,
``/proc/meminfo`` and its own bookkeeping and applies the decision.

THE RULES, in the order they are applied (docs/DESIGN.md, "Admission policy"):

0. A request needing more than ``total - headroom`` can never fit: **Fail** immediately. (The
   in-process predecessor waited forever for 79 GB of a card that could never offer more than 77.)
1. Loads pass the host gate first: if ``loads_in_flight >= max_concurrent_loads``, **Wait**; if
   ``host_peak`` exceeds available host RAM (after ``host_headroom`` and the host peaks of loads
   in flight), **Wait** if a load is in flight, else **Fail**. The gate comes before VRAM so the
   arbiter never evicts models for a load that could not start anyway.
2. ``effective_free = free - reserved - headroom``. If ``need <= effective_free``: **Admit**.
3. If evictions already in flight (``pending``) would cover it: **Wait** for them, evicting nothing
   more. Without this rule a slow driver turns one eviction into several.
4. Victims are idle (resident, no leases), unpinned, responsive (and not momentarily protected)
   models on the same device whose priority is ``<=`` the requester's, ordered by
   ``(priority, last_used, id)``. Take the shortest prefix whose resident sizes cover the
   shortfall: **Evict**.
5. If even every eligible victim is not enough: **Wait** if a busy or loading model on the device
   could free memory later (or one that is only momentarily protected), else **Fail** naming
   the holders.

:func:`plan_queue` applies ``plan`` to the whole wait queue in priority-then-FIFO order, with each
waiting request earmarking the memory it is counting on so later requests can only use what is
left (they may "backfill", never steal).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum

from .errors import Holder, HostRamUnavailable, VramUnavailable
from .units import format_bytes

__all__ = [
    "Phase",
    "WaitKind",
    "ModelView",
    "DeviceView",
    "ForeignView",
    "Snapshot",
    "Request",
    "Admit",
    "Evict",
    "Wait",
    "Fail",
    "Decision",
    "plan",
    "plan_queue",
    "victim_order",
    "holders",
]


class Phase(str, Enum):
    """Where a model is in its lifecycle. *Busy* is not a phase: it is ``leases > 0``."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    RESIDENT = "resident"
    EVICTING = "evicting"


class WaitKind(str, Enum):
    LOAD_SLOT = "load_slot"  # max_concurrent_loads reached
    HOST_RAM = "host_ram"  # a load in flight is using the host RAM this one needs
    EVICTING = "evicting"  # evictions in flight will make room
    BUSY = "busy"  # busy/loading models could free memory later
    QUEUED = "queued"  # behind an earlier request (same model loading, or an earlier load)


@dataclass(frozen=True)
class ModelView:
    """What the planner needs to know about one model."""

    id: str
    device: int
    phase: Phase
    leases: int = 0
    pinned: bool = False
    priority: int = 0
    last_used: float = 0.0
    #: Best estimate of the bytes evicting this model returns (measured, else declared).
    resident_bytes: int = 0
    unresponsive: bool = False
    #: Briefly not evictable: it refused an eviction a moment ago because a lease was arriving.
    protected: bool = False

    @property
    def busy(self) -> bool:
        return self.leases > 0 and self.phase in (Phase.RESIDENT, Phase.LOADING)

    @property
    def idle(self) -> bool:
        return self.phase is Phase.RESIDENT and self.leases == 0


@dataclass(frozen=True)
class DeviceView:
    index: int
    free: int  # what the driver says is free
    total: int
    reserved: int = 0  # promised to loads in flight and to busy models' working headroom
    pending: int = 0  # expected back from evictions in flight or settling


@dataclass(frozen=True)
class ForeignView:
    """A process holding GPU memory that is not a known satellite. Counted, never evicted."""

    pid: int
    device: int
    bytes: int


@dataclass(frozen=True)
class Snapshot:
    devices: Mapping[int, DeviceView]
    models: tuple[ModelView, ...] = ()
    headroom: int = 0
    host_available: int | None = None  # None: unknown, host-RAM check skipped
    host_headroom: int = 0
    host_reserved: int = 0  # host_peak of loads in flight
    loads_in_flight: int = 0
    max_concurrent_loads: int = 1
    foreign: tuple[ForeignView, ...] = ()

    def effective_free(self, device: int) -> int:
        dev = self.devices[device]
        return dev.free - dev.reserved - self.headroom

    def model(self, model_id: str) -> ModelView | None:
        for m in self.models:
            if m.id == model_id:
                return m
        return None


@dataclass(frozen=True)
class Request:
    """One admission question.

    ``need`` is the VRAM to find: ``vram_peak`` for a load, ``vram_peak - resident`` (the working
    headroom) for a lease on an idle resident model, 0 for another lease on a busy one.
    """

    model_id: str
    device: int
    need: int
    priority: int = 0
    is_load: bool = False
    host_peak: int = 0
    seq: int = 0  # FIFO position, used by plan_queue


@dataclass(frozen=True)
class Admit:
    pass


@dataclass(frozen=True)
class Evict:
    victims: tuple[str, ...]
    #: Sum of the victims' resident estimates.
    expected: int


@dataclass(frozen=True)
class Wait:
    kind: WaitKind
    reason: str
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fail:
    """Carries the error's fields rather than the exception so decisions compare by value."""

    need: int
    free: int
    reason: str
    holders: tuple[Holder, ...] = ()
    host: bool = False

    def to_error(self, model_id: str, device: int, *, reason: str | None = None) -> VramUnavailable:
        cls = HostRamUnavailable if self.host else VramUnavailable
        return cls(
            model=model_id,
            device=device,
            need=self.need,
            free=self.free,
            reason=reason if reason is not None else self.reason,
            holders=self.holders,
        )


Decision = Admit | Evict | Wait | Fail


def victim_order(models: Iterable[ModelView]) -> list[ModelView]:
    """Eviction order: lowest priority first, then least recently used, then id (determinism)."""
    return sorted(models, key=lambda m: (m.priority, m.last_used, m.id))


def eligible_victims(request: Request, snapshot: Snapshot) -> list[ModelView]:
    """Idle, unpinned, responsive models on the request's device that it may evict, in order."""
    return victim_order(
        m
        for m in snapshot.models
        if m.device == request.device
        and m.idle
        and not m.pinned
        and not m.unresponsive
        and not m.protected
        and m.priority <= request.priority
        and m.id != request.model_id
        and m.resident_bytes > 0  # evicting something that returns nothing only costs a reload
    )


def holders(snapshot: Snapshot, device: int) -> tuple[Holder, ...]:
    """Everything holding memory on ``device``, largest first, for error messages and status.

    Models report their resident estimate; whatever the driver counts as used that neither models
    nor foreign processes account for is reported as one ``unattributed`` holder, which is where
    CUDA contexts and un-emptied caching allocators show up.
    """
    dev = snapshot.devices.get(device)
    found: list[Holder] = []
    for m in snapshot.models:
        if m.device != device or m.phase is Phase.UNLOADED:
            continue
        state = "busy" if m.busy else m.phase.value
        if m.unresponsive:
            state += ", unresponsive"
        found.append(
            Holder(
                kind="model",
                name=m.id,
                bytes=m.resident_bytes,
                device=device,
                state=state,
                pinned=m.pinned,
                priority=m.priority,
            )
        )
    for f in snapshot.foreign:
        if f.device == device:
            found.append(
                Holder(kind="foreign", name=f"pid {f.pid}", bytes=f.bytes, device=device, pid=f.pid)
            )
    if dev is not None:
        unattributed = dev.total - dev.free - sum(h.bytes for h in found)
        if unattributed > 0:
            found.append(
                Holder(kind="unattributed", name="unattributed", bytes=unattributed, device=device)
            )
    return tuple(sorted(found, key=lambda h: (-h.bytes, h.name)))


def plan(request: Request, snapshot: Snapshot) -> Decision:
    """Decide one request against one snapshot. Pure and deterministic."""
    dev = snapshot.devices.get(request.device)
    if dev is None:
        return Fail(
            need=request.need,
            free=0,
            reason=f"there is no GPU {request.device}",
        )

    # Rule 0: can never fit, whatever is evicted or finishes.
    capacity = dev.total - snapshot.headroom
    if request.need > capacity:
        return Fail(
            need=request.need,
            free=max(0, snapshot.effective_free(request.device)),
            reason=(
                f"more than this GPU can ever offer ({format_bytes(dev.total)} total, "
                f"{format_bytes(snapshot.headroom)} headroom); the declared vram_peak is too large"
            ),
            holders=holders(snapshot, request.device),
        )

    # Rule 1: the host gate, for loads only.
    if request.is_load:
        loading = tuple(sorted(m.id for m in snapshot.models if m.phase is Phase.LOADING))
        if snapshot.loads_in_flight >= snapshot.max_concurrent_loads:
            return Wait(
                WaitKind.LOAD_SLOT,
                f"{snapshot.loads_in_flight} load(s) in flight "
                f"(max_concurrent_loads={snapshot.max_concurrent_loads})",
                loading,
            )
        if snapshot.host_available is not None and request.host_peak > 0:
            host_free = snapshot.host_available - snapshot.host_headroom - snapshot.host_reserved
            if request.host_peak > host_free:
                if snapshot.loads_in_flight > 0:
                    return Wait(
                        WaitKind.HOST_RAM,
                        f"needs {format_bytes(request.host_peak)} host RAM, "
                        f"{format_bytes(max(0, host_free))} available while a load is in flight",
                        loading,
                    )
                return Fail(
                    need=request.host_peak,
                    free=max(0, host_free),
                    reason=(
                        f"not enough host RAM to load: needs {format_bytes(request.host_peak)}, "
                        f"{format_bytes(max(0, host_free))} available after "
                        f"{format_bytes(snapshot.host_headroom)} host headroom, "
                        "and no load is in flight"
                    ),
                    host=True,
                )

    if request.need <= 0:
        return Admit()

    # Rule 2.
    effective_free = snapshot.effective_free(request.device)
    if request.need <= effective_free:
        return Admit()

    # Rule 3.
    if request.need <= effective_free + dev.pending:
        evicting = tuple(
            sorted(
                m.id
                for m in snapshot.models
                if m.device == request.device and m.phase is Phase.EVICTING
            )
        )
        return Wait(
            WaitKind.EVICTING,
            f"waiting for {format_bytes(dev.pending)} expected back from evictions in progress",
            evicting,
        )

    # Rule 4.
    shortfall = request.need - (effective_free + dev.pending)
    chosen: list[ModelView] = []
    covered = 0
    for victim in eligible_victims(request, snapshot):
        chosen.append(victim)
        covered += victim.resident_bytes
        if covered >= shortfall:
            return Evict(tuple(m.id for m in chosen), covered)

    # Rule 5.
    blockers = tuple(
        sorted(
            m.id
            for m in snapshot.models
            if m.device == request.device
            and m.id != request.model_id
            and (m.busy or m.phase is Phase.LOADING or (m.idle and m.protected))
        )
    )
    if blockers:
        return Wait(
            WaitKind.BUSY,
            f"needs {format_bytes(shortfall)} more than idle models can free; "
            f"waiting for {', '.join(blockers)}",
            blockers,
        )
    return Fail(
        need=request.need,
        free=max(0, effective_free),
        reason=(
            f"evicting every eligible idle model would free {format_bytes(covered)} of the "
            f"{format_bytes(shortfall)} missing, and nothing busy or loading could free more"
        ),
        holders=holders(snapshot, request.device),
    )


def plan_queue(requests: Sequence[Request], snapshot: Snapshot) -> list[tuple[Request, Decision]]:
    """Plan every queued request, highest priority first, then FIFO (``seq``).

    Decisions are made against a working copy of the snapshot that each decision updates:

    * an admitted request reserves its ``need`` (and a load slot and its host peak), and its model
      becomes busy so nothing later in the pass picks it as a victim;
    * an eviction marks its victims ``EVICTING`` and counts them as pending;
    * a request left waiting on VRAM *earmarks* the room it is counting on (pending first, then
      free), so a later request can only be admitted into what remains. That allows backfilling a
      small request past a large one stuck behind a busy model, without ever taking the large
      one's room;
    * a load left waiting at the host gate blocks later loads, so loads start in queue order;
    * a second request for a model admitted earlier in the pass waits for that load.
    """
    ordered = sorted(requests, key=lambda r: (-r.priority, r.seq))
    devices = dict(snapshot.devices)
    models = {m.id: m for m in snapshot.models}
    working = snapshot
    blocked_load: str | None = None
    admitted: set[str] = set()
    out: list[tuple[Request, Decision]] = []

    for original in ordered:
        request = original
        current = models.get(request.model_id)
        if not request.is_load and current is not None and current.busy:
            # Another lease on a model that is already busy (perhaps made busy earlier in this
            # pass): its working headroom is already reserved.
            request = replace(request, need=0)
        if request.model_id in admitted:
            decision: Decision = Wait(
                WaitKind.QUEUED, f"{request.model_id} is being admitted", (request.model_id,)
            )
        elif request.is_load and blocked_load is not None:
            decision = Wait(
                WaitKind.QUEUED, f"queued behind the load of {blocked_load}", (blocked_load,)
            )
        else:
            decision = plan(request, working)
        out.append((original, decision))

        dev = devices.get(request.device)
        if isinstance(decision, Admit):
            if request.is_load:
                admitted.add(request.model_id)
            if dev is not None and request.need > 0:
                devices[request.device] = replace(dev, reserved=dev.reserved + request.need)
            if request.model_id in models:
                m = models[request.model_id]
                models[m.id] = replace(
                    m,
                    leases=m.leases + 1,
                    phase=Phase.LOADING if request.is_load else m.phase,
                )
            if request.is_load:
                working = replace(
                    working,
                    loads_in_flight=working.loads_in_flight + 1,
                    host_reserved=working.host_reserved + request.host_peak,
                )
        elif isinstance(decision, Evict) and dev is not None:
            for victim in decision.victims:
                models[victim] = replace(models[victim], phase=Phase.EVICTING)
            dev = replace(dev, pending=dev.pending + decision.expected)
            devices[request.device] = _earmark(dev, request.need, working.headroom)
        elif isinstance(decision, Wait):
            if decision.kind in (WaitKind.LOAD_SLOT, WaitKind.HOST_RAM) and request.is_load:
                blocked_load = request.model_id
            elif decision.kind in (WaitKind.EVICTING, WaitKind.BUSY) and dev is not None:
                devices[request.device] = _earmark(dev, request.need, working.headroom)
        working = replace(working, devices=dict(devices), models=tuple(models.values()))
    return out


def _earmark(dev: DeviceView, need: int, headroom: int) -> DeviceView:
    """Set aside up to ``need`` of the room a waiting request is counting on: pending first."""
    available = max(0, dev.free - dev.reserved - headroom) + dev.pending
    amount = min(need, available)
    from_pending = min(amount, dev.pending)
    return replace(
        dev,
        pending=dev.pending - from_pending,
        reserved=dev.reserved + (amount - from_pending),
    )
