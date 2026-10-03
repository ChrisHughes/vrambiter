"""The arbiter's bookkeeping, and the pure functions that turn it plus a measurement into a
:class:`~vrambiter.policy.Snapshot`.

Nothing here does I/O or reads a clock: times are passed in. ``arbiter.py`` owns one
:class:`Registry`, mutates it in response to messages, and calls :func:`build_snapshot` with a fresh
:class:`~vrambiter.gpu.GpuSnapshot` before every planning pass.

THREE NUMBERS PER MODEL, because a declared size is a peak and not a residency:

* ``vram_peak`` (declared) is what admission reserves for a load or a working lease.
* ``vram_resident`` (declared, optional) is what the model is expected to hold while idle.
* ``measured_bytes`` is what the driver says it actually holds while idle, attributed from NVML
  per-process usage (and refined by the satellite's own report). It wins whenever it exists.

RESERVATIONS ARE AN ENVELOPE, NOT A SUM OF PEAKS. A loading or busy model may still grow to its
peak. Rather than reserving the whole peak on top of what the driver already shows as used (which
double-counts everything the load has allocated so far), the reservation for a satellite process is
``envelope - usage``: the most its models may hold (peak for loading/busy models, resident size for
idle ones) minus what the process holds now. It shrinks as the load progresses and is never
negative. When per-process usage is unknown, the full outstanding peaks are reserved instead.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from .gpu import GpuSnapshot
from .policy import DeviceView, ForeignView, ModelView, Phase, Request, Snapshot

__all__ = [
    "OwnerKind",
    "ModelRecord",
    "Lease",
    "SatelliteRecord",
    "Settle",
    "Registry",
    "SETTLE_FRACTION",
    "request_for",
    "satellite_reservation",
    "attribute_usage",
    "build_snapshot",
]

#: An eviction is considered returned once this fraction of the expected bytes is back. Estimates
#: are estimates; waiting for the last byte would wait for the deadline every time.
SETTLE_FRACTION = 0.8


class OwnerKind(str, Enum):
    COOPERATIVE = "cooperative"  # the satellite loads/unloads through the client library
    ADAPTER = "adapter"  # the arbiter loads/unloads through an adapter (llama-router)
    MANAGED = "managed"  # a black-box process: loading is starting it, unloading is stopping it


@dataclass
class ModelRecord:
    satellite: str
    name: str
    vram_peak: int
    device: int = 0
    vram_resident: int | None = None
    host_peak: int = 0
    priority: int = 0
    declared_pinned: bool = False
    owner_kind: OwnerKind = OwnerKind.COOPERATIVE

    phase: Phase = Phase.UNLOADED
    leases: set[str] = field(default_factory=set)
    last_used: float = 0.0
    reported_bytes: int | None = None  # the satellite's own figure (e.g. torch memory_reserved)
    measured_bytes: int | None = None  # attributed from NVML while idle
    unresponsive: bool = False
    manual_pin: bool | None = None  # operator override: True/False, None = no override
    profile_pin: bool = False
    #: Processes that are exactly this model (llama-router child instances); empty otherwise.
    pids: frozenset[int] = frozenset()
    evict_id: str | None = None
    evict_reason: str = ""
    #: What the owner held, and what was free, when the eviction was sent (for settling).
    evict_usage_before: int | None = None
    evict_free_before: int = 0
    #: Not a victim before this time (it just refused an eviction because a lease was arriving).
    protected_until: float = 0.0
    load_started_at: float | None = None
    mismatch_logged: bool = False
    #: Bumped on every phase change, so results computed before a transition (a driver's poll
    #: of the server's own state, say) can be recognised as stale and dropped.
    version: int = 0

    def __setattr__(self, name: str, value: object) -> None:
        if name == "phase" and getattr(self, "phase", None) is not value:
            object.__setattr__(self, "version", getattr(self, "version", 0) + 1)
        object.__setattr__(self, name, value)

    @property
    def id(self) -> str:
        return f"{self.satellite}/{self.name}"

    @property
    def pinned(self) -> bool:
        if self.manual_pin is not None:
            return self.manual_pin
        return self.declared_pinned or self.profile_pin

    @property
    def busy(self) -> bool:
        return bool(self.leases) and self.phase in (Phase.RESIDENT, Phase.LOADING)

    @property
    def state_label(self) -> str:
        """``unloaded|loading|resident|busy|evicting`` - the design's state names."""
        if self.phase is Phase.RESIDENT and self.leases:
            return "busy"
        return self.phase.value

    def resident_estimate(self) -> int:
        """Bytes this model holds (or will hold) while idle: measured, reported, declared, peak."""
        for value in (self.measured_bytes, self.reported_bytes, self.vram_resident):
            if value is not None:
                return value
        return self.vram_peak

    def working_headroom(self) -> int:
        """How much more than its idle size a lease may make this model hold."""
        return max(0, self.vram_peak - self.resident_estimate())

    def view(self, now: float = 0.0) -> ModelView:
        return ModelView(
            id=self.id,
            device=self.device,
            phase=self.phase,
            leases=len(self.leases),
            pinned=self.pinned,
            priority=self.priority,
            last_used=self.last_used,
            resident_bytes=(
                self.vram_peak if self.phase is Phase.LOADING else self.resident_estimate()
            ),
            unresponsive=self.unresponsive,
            protected=now < self.protected_until,
        )


@dataclass
class Lease:
    id: str
    model_id: str
    holder: str  # satellite name of the connection holding it
    conn_id: int | None
    granted_at: float
    action: str = "ready"


@dataclass
class SatelliteRecord:
    name: str
    pid: int | None = None
    conn_id: int | None = None
    managed: bool = False
    adapter: bool = False
    #: The satellite's process and its descendants, refreshed with every measurement.
    pids: frozenset[int] = frozenset()
    #: Per-device usage with no model resident (CUDA context, allocator floor), learned over time.
    baseline: dict[int, int] = field(default_factory=dict)
    #: Per-device bytes an eviction expected back but the driver never showed returned.
    residue: dict[int, int] = field(default_factory=dict)
    invisible_logged: bool = False

    @property
    def connected(self) -> bool:
        return self.conn_id is not None

    def all_pids(self) -> frozenset[int]:
        return self.pids | ({self.pid} if self.pid else frozenset())


@dataclass
class Settle:
    """Memory the arbiter expects the driver to show returned soon.

    Created when a satellite reports ``evicted`` and when one disconnects. Until it settles, the
    outstanding bytes count as *pending* (so the planner waits instead of evicting more) and
    ``reserve`` bytes stay reserved (a load that was in flight when its process went away).
    """

    key: str
    satellite: str
    device: int
    pids: frozenset[int]
    expected: int
    usage_before: int | None
    free_before: int
    deadline: float
    reserve: int = 0
    model_id: str | None = None

    def returned(self, gpu: GpuSnapshot) -> int:
        usage_now = gpu.pid_usage(self.pids, self.device) if self.pids else None
        if self.usage_before is not None and usage_now is not None:
            return max(0, self.usage_before - usage_now)
        return max(0, gpu.free.get(self.device, 0) - self.free_before)

    def outstanding(self, gpu: GpuSnapshot) -> int:
        return max(0, self.expected - self.returned(gpu))

    def settled(self, gpu: GpuSnapshot, now: float) -> bool:
        if now >= self.deadline:
            return True
        if self.pids and gpu.pid_usage(self.pids, self.device) == 0:
            return True  # the processes hold nothing on this device any more
        if not self.expected:
            # Only a reservation is held (a load was in flight): wait until the process's memory
            # is gone, since "nothing returned yet" is indistinguishable from "nothing to return".
            return False
        return self.returned(gpu) >= SETTLE_FRACTION * self.expected


class Registry:
    """Models, satellites and leases. Mutated only on the event loop thread."""

    def __init__(self) -> None:
        self.models: dict[str, ModelRecord] = {}
        self.satellites: dict[str, SatelliteRecord] = {}
        self.leases: dict[str, Lease] = {}
        self.settling: dict[str, Settle] = {}
        self._lease_ids = itertools.count(1)
        self._evict_ids = itertools.count(1)
        self._settle_ids = itertools.count(1)

    # -- lookup -------------------------------------------------------------------------------

    def models_of(self, satellite: str) -> list[ModelRecord]:
        return [m for m in self.models.values() if m.satellite == satellite]

    def resolve(self, name: str, requester: str | None = None) -> ModelRecord | None:
        """A requester's own model by short name, else a model by full id."""
        if requester is not None:
            own = self.models.get(f"{requester}/{name}")
            if own is not None:
                return own
        return self.models.get(name)

    # -- leases -------------------------------------------------------------------------------

    def add_lease(
        self, model: ModelRecord, holder: str, conn_id: int | None, now: float, action: str
    ) -> Lease:
        lease = Lease(f"L{next(self._lease_ids)}", model.id, holder, conn_id, now, action)
        self.leases[lease.id] = lease
        model.leases.add(lease.id)
        model.last_used = now
        return lease

    def end_lease(self, lease_id: str, now: float) -> Lease | None:
        lease = self.leases.pop(lease_id, None)
        if lease is None:
            return None
        model = self.models.get(lease.model_id)
        if model is not None:
            model.leases.discard(lease_id)
            model.last_used = now
        return lease

    def next_evict_id(self) -> str:
        return f"E{next(self._evict_ids)}"

    def next_settle_key(self) -> str:
        return f"S{next(self._settle_ids)}"

    # -- aggregates -----------------------------------------------------------------------------

    def loading(self) -> list[ModelRecord]:
        return [m for m in self.models.values() if m.phase is Phase.LOADING]

    def known_pids(self) -> frozenset[int]:
        pids: set[int] = set()
        for sat in self.satellites.values():
            pids |= sat.all_pids()
        for model in self.models.values():
            pids |= model.pids
        return frozenset(pids)


def request_for(model: ModelRecord, *, seq: int = 0) -> Request | None:
    """The admission question for an ``acquire`` on ``model`` right now, or ``None`` if the model
    is mid-transition (loading or evicting) and the request must simply wait for that to end."""
    if model.phase is Phase.UNLOADED:
        return Request(
            model_id=model.id,
            device=model.device,
            need=model.vram_peak,
            priority=model.priority,
            is_load=True,
            host_peak=model.host_peak,
            seq=seq,
        )
    if model.phase is Phase.RESIDENT:
        return Request(
            model_id=model.id,
            device=model.device,
            need=0 if model.busy else model.working_headroom(),
            priority=model.priority,
            is_load=False,
            seq=seq,
        )
    return None


def satellite_reservation(
    models: Iterable[ModelRecord], usage: int | None, baseline: int = 0
) -> int:
    """Bytes to reserve for one satellite process on one device (see the module docstring).

    ``models`` are that satellite's models on that device; ``usage`` is what its processes hold
    there now, or ``None`` if the driver cannot say. ``baseline`` is the process's floor (CUDA
    context, allocator pool) when known: it is in ``usage`` but in no model's estimate, so it
    belongs in the envelope too, or every load would be under-reserved by it.
    """
    envelope = baseline
    outstanding = 0  # the most that could still be needed: full peaks of loads, busy headroom
    for m in models:
        if m.phase is Phase.LOADING:
            envelope += m.vram_peak
            outstanding += m.vram_peak
        elif m.phase in (Phase.RESIDENT, Phase.EVICTING):
            if m.leases:
                envelope += max(m.vram_peak, m.resident_estimate())
                outstanding += m.working_headroom()
            else:
                envelope += m.resident_estimate()
    if usage is None:
        return outstanding
    return max(0, min(outstanding, envelope - usage))


def attribute_usage(
    models: Sequence[ModelRecord], usage: int | None, baseline: int = 0
) -> dict[str, int]:
    """Split one satellite's measured usage on one device among its idle resident models.

    Returns ``{model_id: bytes}`` for the models it could attribute, and nothing while any model
    of the satellite on that device is loading or busy (the process total then includes transient
    working memory, which is not anyone's residency). One resident model takes the whole figure
    minus the satellite's baseline; several split it in proportion to their own reports or
    declarations.
    """
    if usage is None:
        return {}
    if any(m.phase is Phase.LOADING or m.leases for m in models):
        return {}
    resident = [m for m in models if m.phase is Phase.RESIDENT]
    if not resident:
        return {}
    attributable = max(0, usage - baseline)
    if len(resident) == 1:
        return {resident[0].id: attributable}
    weights = [m.reported_bytes or m.vram_resident or m.vram_peak or 1 for m in resident]
    total_weight = sum(weights)
    return {m.id: attributable * w // total_weight for m, w in zip(resident, weights, strict=True)}


def build_snapshot(
    registry: Registry,
    gpu: GpuSnapshot,
    *,
    host_available: int | None,
    headroom: int,
    host_headroom: int,
    max_concurrent_loads: int,
    now: float = 0.0,
) -> Snapshot:
    """Combine bookkeeping and a measurement into the planner's input."""
    devices: dict[int, DeviceView] = {}
    by_sat_dev: dict[tuple[str, int], list[ModelRecord]] = {}
    for m in registry.models.values():
        by_sat_dev.setdefault((m.satellite, m.device), []).append(m)

    for device in gpu.devices:
        reserved = 0
        pending = 0
        for (sat_name, dev), models in by_sat_dev.items():
            if dev != device:
                continue
            sat = registry.satellites.get(sat_name)
            pids = sat.all_pids() if sat else frozenset()
            # Models with their own processes (router instances) are reserved individually.
            own = [m for m in models if m.pids]
            shared = [m for m in models if not m.pids]
            for m in own:
                reserved += satellite_reservation([m], gpu.pid_usage(m.pids, device))
            if shared:
                own_pids = frozenset().union(*(m.pids for m in own)) if own else frozenset()
                usage = gpu.pid_usage(pids - own_pids, device) if pids else None
                baseline = sat.baseline.get(device, 0) if sat else 0
                reserved += satellite_reservation(shared, usage, baseline)
            pending += sum(m.resident_estimate() for m in models if m.phase is Phase.EVICTING)
        for settle in registry.settling.values():
            if settle.device == device:
                # A held reservation comes back when the settle ends, so it is pending too: the
                # planner waits for it rather than failing or evicting around it.
                reserved += settle.reserve
                pending += settle.outstanding(gpu) + settle.reserve
        devices[device] = DeviceView(
            index=device,
            free=gpu.free[device],
            total=gpu.total[device],
            reserved=reserved,
            pending=pending,
        )

    known = registry.known_pids()
    foreign = tuple(
        ForeignView(pid=pid, device=device, bytes=nbytes)
        for device, per_pid in sorted(gpu.usage.items())
        if per_pid
        for pid, nbytes in sorted(per_pid.items())
        if pid not in known and nbytes > 0
    )
    loading = registry.loading()
    return Snapshot(
        devices=devices,
        models=tuple(m.view(now) for m in sorted(registry.models.values(), key=lambda m: m.id)),
        headroom=headroom,
        host_available=host_available,
        host_headroom=host_headroom,
        host_reserved=sum(m.host_peak for m in loading),
        loads_in_flight=len(loading),
        max_concurrent_loads=max_concurrent_loads,
        foreign=foreign,
    )


def usage_by_satellite(registry: Registry, gpu: GpuSnapshot) -> Mapping[str, dict[int, int | None]]:
    """``{satellite: {device: bytes or None}}`` for status output and baseline learning."""
    out: dict[str, dict[int, int | None]] = {}
    for sat in registry.satellites.values():
        pids = sat.all_pids()
        out[sat.name] = {
            device: (gpu.pid_usage(pids, device) if pids else None) for device in gpu.devices
        }
    return out
