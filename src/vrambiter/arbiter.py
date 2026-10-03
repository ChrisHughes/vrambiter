"""The daemon core: applies plans, drives evictions, enforces timeouts, reacts to events.

``Arbiter`` owns the :class:`~vrambiter.state.Registry` and the wait queue. It is driven from three
directions, all on one asyncio event loop:

* **messages** from satellites (``server.py`` calls :meth:`Arbiter.handle` for each decoded line and
  :meth:`Arbiter.disconnected` when a socket closes). Handlers mutate bookkeeping synchronously and
  never await, so messages from one connection are applied in order.
* **timers** from the injected :class:`~vrambiter.clock.Clock`: request deadlines, eviction
  deadlines, the NVML poll, settle re-checks.
* **drivers** for models the arbiter loads itself (adapter-backed and black-box managed models),
  whose blocking work runs in worker threads and reports back on the loop.

Every event that could change a decision calls :meth:`kick`, which wakes a single *reconcile* task.
Reconcile measures (NVML, host RAM, process trees: blocking, so in a worker thread), learns
measurements, retires settled evictions, builds a snapshot and plans the whole queue with
:func:`~vrambiter.policy.plan_queue`. Because only that one task grants, two admissions can never
both count the same free gigabytes, and waiting is purely event-driven: there is no sleep loop
anywhere, only the poll timer that catches memory moved by processes the arbiter does not know.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import itertools
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import __version__
from . import protocol as p
from .clock import Clock, LoopClock, TimerHandle
from .errors import (
    ArbiterError,
    NameInUse,
    ProtocolError,
    RegistrationError,
    UnknownModel,
    VrambiterError,
    VramUnavailable,
)
from .gpu import GpuBackend, GpuSnapshot, take_snapshot
from .host import HostBackend, process_tree
from .policy import Admit, Evict, Fail, Phase, Snapshot, Wait, WaitKind, holders, plan_queue
from .state import (
    SETTLE_FRACTION,
    ModelRecord,
    OwnerKind,
    Registry,
    SatelliteRecord,
    Settle,
    attribute_usage,
    build_snapshot,
    request_for,
)
from .units import GiB, format_bytes

__all__ = [
    "Arbiter",
    "ArbiterSettings",
    "Connection",
    "LoadResult",
    "ModelDriver",
    "ModelSpec",
]

log = logging.getLogger("vrambiter.arbiter")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass
class ArbiterSettings:
    """Tunables, normally from the ``[arbiter]`` table of ``vrambiter.toml``."""

    headroom: int = 2 * GiB
    host_headroom: int = 3 * GiB
    max_concurrent_loads: int = 1
    poll_interval_s: float = 1.0
    evict_timeout_s: float = 30.0
    #: How long to wait for the driver to show an eviction's memory returned.
    settle_timeout_s: float = 5.0
    settle_poll_s: float = 0.25
    #: After ``evict_refused``, the model is not chosen as a victim again for this long.
    refuse_cooldown_s: float = 2.0
    #: Black-box managed processes: SIGTERM -> SIGKILL grace.
    kill_grace_s: float = 10.0
    #: Kill a managed process on evict timeout even if it has other models.
    kill_on_evict_timeout: bool = False
    #: ``{profile name: [glob patterns of model ids to pin]}``.
    profiles: dict[str, list[str]] = field(default_factory=dict)


class Connection(Protocol):
    """What the arbiter needs from a client connection (``server.SocketConnection``)."""

    id: int

    def send(self, message: p.Message) -> None: ...

    def close(self) -> None: ...


@dataclass
class ModelSpec:
    """A model declared in config for an adapter-backed or managed satellite."""

    name: str
    vram_peak: int
    vram_resident: int | None = None
    host_peak: int = 0
    priority: int = 0
    pinned: bool = False
    device: int = 0


@dataclass
class LoadResult:
    """What a driver learned while loading: the model's own processes and/or its footprint."""

    vram_bytes: int | None = None
    pids: frozenset[int] = frozenset()


class ModelDriver(Protocol):
    """Loads and unloads models for a satellite that does not speak the protocol itself.

    Implemented by ``managed.BlackBoxDriver`` (start/stop a process) and
    ``adapters.llama_router.LlamaRouterDriver`` (HTTP calls to llama-server). Methods are awaited
    on the event loop and must push blocking work to threads themselves.
    """

    kind: OwnerKind

    async def load(self, model: str) -> LoadResult: ...

    async def unload(self, model: str) -> None: ...

    def root_pids(self) -> frozenset[int]:
        """Processes whose trees belong to this satellite (for attributing GPU memory)."""
        ...

    async def force_stop(self) -> None:
        """Last resort after an eviction timed out: stop the whole process."""
        ...


@dataclass
class _ConnState:
    conn: Connection
    name: str | None = None
    role: str = "satellite"


@dataclass
class _Queued:
    seq: int
    conn: Connection | None  # None: internal request (autostart, warm-up)
    request_id: int | None
    requester: str
    model_id: str
    wait: bool
    enqueued_at: float
    deadline: float | None = None
    timer: TimerHandle | None = None
    reason: str = "queued"
    done: bool = False
    on_done: Callable[[str | None, VrambiterError | None], None] | None = None


@dataclass
class _Measurement:
    gpu: GpuSnapshot
    host_available: int | None
    host_total: int | None
    trees: dict[str, frozenset[int]]


class Arbiter:
    def __init__(
        self,
        gpu: GpuBackend,
        host: HostBackend,
        *,
        settings: ArbiterSettings | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.gpu = gpu
        self.host = host
        self.settings = settings or ArbiterSettings()
        self.clock: Clock = clock or LoopClock()
        self.registry = Registry()
        self.queue: list[_Queued] = []
        self.active_profile: str = ""
        self._conns: dict[int, _ConnState] = {}
        self._drivers: dict[str, ModelDriver] = {}
        #: Managed satellites' last resort after an eviction times out (SIGTERM, then SIGKILL).
        self._force_stop: dict[str, Callable[[], Awaitable[None]]] = {}
        self._evict_timers: dict[str, TimerHandle] = {}
        self._seq = itertools.count(1)
        self._wake = asyncio.Event()
        self._reconcile_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._poll_timer: TimerHandle | None = None
        self._settle_timer: TimerHandle | None = None
        self._release_gen = 0
        self._enqueued_seq = 0
        self._last: _Measurement | None = None
        self._last_snapshot: Snapshot | None = None
        self._started_at = 0.0
        self._closed = False
        self._planning = False
        #: Incremented after every planning pass (observability; tests use :meth:`quiesce`).
        self.passes = 0

    # ------------------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Take a first measurement (so devices are known), then start reconciling and polling."""
        self._started_at = self.clock.now()
        self._last = await self._measure()
        self._reconcile_task = asyncio.create_task(self._reconcile_loop(), name="vrambiter-plan")
        self._schedule_poll()
        self.kick()

    async def close(self) -> None:
        self._closed = True
        for timer in (self._poll_timer, self._settle_timer, *self._evict_timers.values()):
            if timer is not None:
                timer.cancel()
        for q in self.queue:
            if q.timer:
                q.timer.cancel()
        self._wake.set()
        tasks = [t for t in (self._reconcile_task, *self._tasks) if t is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def quiesce(self, timeout: float = 5.0) -> None:
        """Wait until no planning pass is pending or running. For tests and tools: production
        code never needs to wait for the planner."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self._wake.is_set() or self._planning:
            if loop.time() > deadline:
                raise TimeoutError("arbiter did not quiesce")
            await asyncio.sleep(0.001)

    def kick(self) -> None:
        """Something changed: re-plan soon. Cheap and idempotent; calls coalesce."""
        self._wake.set()

    def _spawn(self, coro: Any, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # ------------------------------------------------------------------------------ satellites
    # configured from vrambiter.toml (managed / adapter-backed)

    def add_driven_satellite(
        self, name: str, driver: ModelDriver, models: list[ModelSpec], *, managed: bool
    ) -> None:
        """Declare a satellite whose models the arbiter loads itself through ``driver``."""
        if not _NAME_RE.match(name):
            raise ValueError(f"invalid satellite name {name!r}")
        sat = self.registry.satellites.setdefault(name, SatelliteRecord(name))
        sat.managed = managed
        sat.adapter = driver.kind is OwnerKind.ADAPTER
        self._drivers[name] = driver
        if managed:
            self._force_stop[name] = driver.force_stop
        for spec in models:
            record = ModelRecord(
                satellite=name,
                name=spec.name,
                vram_peak=spec.vram_peak,
                device=spec.device,
                vram_resident=spec.vram_resident,
                host_peak=spec.host_peak,
                priority=spec.priority,
                declared_pinned=spec.pinned,
                owner_kind=driver.kind,
            )
            record.profile_pin = self._profile_matches(record.id)
            self.registry.models[record.id] = record

    def add_cooperative_satellite(
        self, name: str, force_stop: Callable[[], Awaitable[None]] | None = None
    ) -> None:
        """Declare a managed satellite that will connect and register its own models."""
        if not _NAME_RE.match(name):
            raise ValueError(f"invalid satellite name {name!r}")
        sat = self.registry.satellites.setdefault(name, SatelliteRecord(name))
        sat.managed = True
        if force_stop is not None:
            self._force_stop[name] = force_stop

    def satellite_process_exited(self, name: str) -> None:
        """A driven satellite's process died (or was stopped): its models are gone."""
        now = self.clock.now()
        for model in self.registry.models_of(name):
            if model.phase is not Phase.UNLOADED:
                self._model_gone(model, now, reason=f"{name} exited")
        self._bump()
        self.kick()

    def external_model_state(self, satellite: str, model_name: str, loaded: bool) -> None:
        """A driver noticed a model change state behind the arbiter's back (e.g. llama-server
        autoloaded it for a request, or unloaded it on its own idle timer)."""
        model = self.registry.models.get(f"{satellite}/{model_name}")
        if model is None or model.phase in (Phase.LOADING, Phase.EVICTING):
            return
        now = self.clock.now()
        if loaded and model.phase is Phase.UNLOADED:
            log.warning("%s was loaded outside the arbiter; tracking it as resident", model.id)
            model.phase = Phase.RESIDENT
            model.last_used = now
            self.kick()
        elif not loaded and model.phase is Phase.RESIDENT and not model.leases:
            log.info("%s was unloaded outside the arbiter", model.id)
            self._model_gone(model, now, reason="unloaded externally")
            self._bump()
            self.kick()

    # ------------------------------------------------------------------------------ messages

    def connected(self, conn: Connection) -> None:
        self._conns[conn.id] = _ConnState(conn)

    def handle(self, conn: Connection, msg: p.Message) -> None:
        """Apply one message. Never awaits; replies via ``conn.send``."""
        state = self._conns.get(conn.id)
        if state is None:
            state = self._conns[conn.id] = _ConnState(conn)
        try:
            if isinstance(msg, p.Hello):
                self._on_hello(state, msg)
                return
            if state.name is None:
                raise ProtocolError("send 'hello' first")
            handler = self._HANDLERS.get(type(msg))
            if handler is None:
                raise ProtocolError(f"{msg.type!r} is not a request")
            handler(self, state, msg)
        except VrambiterError as exc:
            conn.send(p.Error(id=msg.id, **exc.to_wire()))
        except Exception as exc:  # a bug here must not take the daemon down
            log.exception("error handling %s from %s", msg.type, state.name)
            conn.send(p.Error(id=msg.id, code="internal_error", message=str(exc)))

    def disconnected(self, conn: Connection) -> None:
        state = self._conns.pop(conn.id, None)
        if state is None or state.name is None:
            return
        now = self.clock.now()
        for q in [q for q in self.queue if q.conn is conn]:
            self._dequeue(q)
        for lease in [lease for lease in self.registry.leases.values() if lease.conn_id == conn.id]:
            self._end_lease(lease.id, now)
        if state.role != "satellite":
            self._bump()
            self.kick()
            return
        sat = self.registry.satellites.get(state.name)
        if sat is None or sat.conn_id != conn.id:
            return
        log.info("satellite %s disconnected", sat.name)
        sat.conn_id = None
        self._release_satellite_memory(sat, now)
        self._bump()
        self.kick()

    def _on_hello(self, state: _ConnState, msg: p.Hello) -> None:
        if state.name is not None:
            raise ProtocolError("already said hello")
        if msg.role not in ("satellite", "control"):
            raise ProtocolError(f"unknown role {msg.role!r}")
        if msg.role == "control":
            state.name, state.role = msg.name or "control", "control"
            state.conn.send(p.Welcome(id=msg.id, satellite=state.name, arbiter_version=__version__))
            return
        if not _NAME_RE.match(msg.name):
            raise ProtocolError(
                f"invalid satellite name {msg.name!r}: letters, digits, '.', '_', '-'; no '/'"
            )
        sat = self.registry.satellites.get(msg.name)
        if sat is not None and sat.connected:
            raise NameInUse(f"satellite name {msg.name!r} is already connected")
        if sat is not None and msg.name in self._drivers:
            raise NameInUse(f"{msg.name!r} is driven by the arbiter (adapter or black box)")
        if sat is None:
            sat = self.registry.satellites[msg.name] = SatelliteRecord(msg.name)
        sat.conn_id = state.conn.id
        sat.pid = msg.pid
        state.name, state.role = msg.name, "satellite"
        log.info("satellite %s connected (pid %s, client %s)", msg.name, msg.pid, msg.version)
        state.conn.send(p.Welcome(id=msg.id, satellite=msg.name, arbiter_version=__version__))

    def _on_register(self, state: _ConnState, msg: p.Register) -> None:
        if state.role != "satellite":
            raise RegistrationError("control connections cannot register models")
        assert state.name is not None
        if not msg.model or "\n" in msg.model:
            raise RegistrationError("model name must be a non-empty single line")
        for fname in ("vram_peak", "host_peak", "leases"):
            if getattr(msg, fname) < 0:
                raise RegistrationError(f"{fname} must be >= 0")
        if msg.vram_resident is not None and msg.vram_resident > msg.vram_peak:
            raise RegistrationError(
                f"vram_resident ({format_bytes(msg.vram_resident)}) exceeds vram_peak "
                f"({format_bytes(msg.vram_peak)}): a peak is the most a model ever holds"
            )
        if self._last is not None and msg.device not in self._last.gpu.total:
            raise RegistrationError(
                f"no GPU {msg.device} (devices: {sorted(self._last.gpu.total)})"
            )
        if msg.state not in ("unloaded", "resident"):
            raise RegistrationError(f"state must be 'unloaded' or 'resident', not {msg.state!r}")

        now = self.clock.now()
        model_id = f"{state.name}/{msg.model}"
        model = self.registry.models.get(model_id)
        if model is not None and model.owner_kind is not OwnerKind.COOPERATIVE:
            raise RegistrationError(f"{model_id} is declared in the arbiter config")
        fresh = model is None or model.phase is Phase.UNLOADED
        if model is None:
            model = ModelRecord(satellite=state.name, name=msg.model, vram_peak=msg.vram_peak)
            model.last_used = now
            self.registry.models[model_id] = model
        model.vram_peak = msg.vram_peak
        model.vram_resident = msg.vram_resident
        model.host_peak = msg.host_peak
        model.priority = msg.priority
        model.declared_pinned = msg.pinned
        model.device = msg.device
        model.profile_pin = self._profile_matches(model_id)

        adopted: list[str] = []
        if fresh and msg.state == "resident":
            # Re-registration after a reconnect or an arbiter restart: believe the satellite.
            model.phase = Phase.RESIDENT
            model.reported_bytes = msg.vram_bytes
            model.measured_bytes = None
            for _ in range(msg.leases):
                lease = self.registry.add_lease(model, state.name, state.conn.id, now, "ready")
                adopted.append(lease.id)
            log.info("%s re-registered as resident with %d lease(s)", model_id, len(adopted))
        state.conn.send(p.Ok(id=msg.id, model=model_id, leases=adopted or None))
        self.kick()

    def _resolve(self, state: _ConnState, name: str) -> ModelRecord:
        model = self.registry.resolve(name, state.name if state.role == "satellite" else None)
        if model is None:
            raise UnknownModel(f"unknown model {name!r}", detail={"model": name})
        return model

    def _on_acquire(self, state: _ConnState, msg: p.Acquire) -> None:
        assert state.name is not None
        model = self._resolve(state, msg.model)
        if model.satellite != state.name:
            if model.owner_kind is OwnerKind.COOPERATIVE:
                raise ArbiterError(
                    f"{model.id} belongs to cooperative satellite {model.satellite}; consumer "
                    "leases are supported for adapter-backed and managed models only"
                )
        elif model.owner_kind is not OwnerKind.COOPERATIVE:
            raise ArbiterError(f"{model.id} is driven by the arbiter")
        if model.owner_kind is OwnerKind.COOPERATIVE:
            owner = self.registry.satellites.get(model.satellite)
            if owner is None or not owner.connected:
                raise ArbiterError(f"{model.id}: its satellite is not connected")
            if model.unresponsive and model.satellite == state.name:
                model.unresponsive = False  # it is talking to us, so it is alive
        if msg.timeout_s is not None and msg.timeout_s < 0:
            raise ProtocolError("timeout_s must be >= 0")
        # A zero timeout means "do not wait", not "fail at the next loop iteration".
        no_wait = not msg.wait or msg.timeout_s == 0
        self._enqueue(
            conn=state.conn,
            request_id=msg.id,
            requester=state.name,
            model=model,
            wait=not no_wait,
            timeout_s=None if no_wait else msg.timeout_s,
        )

    def _enqueue(
        self,
        *,
        conn: Connection | None,
        request_id: int | None,
        requester: str,
        model: ModelRecord,
        wait: bool,
        timeout_s: float | None,
        on_done: Callable[[str | None, VrambiterError | None], None] | None = None,
    ) -> _Queued:
        now = self.clock.now()
        q = _Queued(
            seq=next(self._seq),
            conn=conn,
            request_id=request_id,
            requester=requester,
            model_id=model.id,
            wait=wait,
            enqueued_at=now,
            on_done=on_done,
        )
        self._enqueued_seq = q.seq
        if timeout_s is not None:
            q.deadline = now + timeout_s
            q.timer = self.clock.call_later(timeout_s, lambda: self._timed_out(q))
        # A lease on a model that is already busy needs nothing: grant without queueing.
        if model.phase is Phase.RESIDENT and model.busy:
            self._grant(q, model, "ready")
            return q
        self.queue.append(q)
        self.kick()
        return q

    def _on_cancel(self, state: _ConnState, msg: p.Cancel) -> None:
        for q in self.queue:
            if q.conn is state.conn and q.request_id == msg.request:
                self._dequeue(q)
                self._reply_error(q, ArbiterError("cancelled", detail={"code": "cancelled"}))
                break
        state.conn.send(p.Ok(id=msg.id))

    def _owned(self, state: _ConnState, name: str) -> ModelRecord:
        model = self.registry.models.get(f"{state.name}/{name}")
        if model is None:
            raise UnknownModel(f"{state.name} has no model {name!r}", detail={"model": name})
        return model

    def _on_loaded(self, state: _ConnState, msg: p.Loaded) -> None:
        model = self._owned(state, msg.model)
        if model.phase is not Phase.LOADING:
            log.warning("%s reported loaded while %s; trusting it", model.id, model.phase.value)
        if model.phase is Phase.EVICTING:
            self._cancel_evict_timer(model)
        model.phase = Phase.RESIDENT
        model.reported_bytes = msg.vram_bytes
        model.measured_bytes = None
        model.unresponsive = False
        model.load_started_at = None
        model.last_used = self.clock.now()
        state.conn.send(p.Ok(id=msg.id))
        self._bump()
        self.kick()

    def _on_load_failed(self, state: _ConnState, msg: p.LoadFailed) -> None:
        model = self._owned(state, msg.model)
        log.warning("%s failed to load: %s", model.id, msg.error or "(no detail)")
        if model.phase is Phase.LOADING:
            self._load_aborted(model)
        state.conn.send(p.Ok(id=msg.id))
        self._bump()
        self.kick()

    def _load_aborted(self, model: ModelRecord) -> None:
        """A load ended without ``loaded``: the reservation goes, the loader's lease goes."""
        now = self.clock.now()
        model.phase = Phase.UNLOADED
        model.load_started_at = None
        for lease_id in list(model.leases):
            lease = self.registry.leases.get(lease_id)
            if lease is not None and lease.action == "load":
                self._end_lease(lease_id, now)

    def _on_release(self, state: _ConnState, msg: p.Release) -> None:
        lease = self.registry.leases.get(msg.lease)
        if lease is not None and lease.conn_id == state.conn.id:
            model = self.registry.models.get(lease.model_id)
            if model is not None and model.phase is Phase.LOADING and lease.action == "load":
                # Released without loaded/load_failed: the load did not happen.
                self._load_aborted(model)
            else:
                self._end_lease(msg.lease, self.clock.now())
        # Releasing an unknown lease is not an error: after a reconnect or a load_failed the
        # client cannot know which of its leases the arbiter still holds.
        state.conn.send(p.Ok(id=msg.id))
        self._bump()
        self.kick()

    def _on_unloaded(self, state: _ConnState, msg: p.Unloaded) -> None:
        model = self._owned(state, msg.model)
        if model.phase in (Phase.RESIDENT, Phase.EVICTING):
            self._cancel_evict_timer(model)
            self._model_gone(model, self.clock.now(), reason="unloaded")
        state.conn.send(p.Ok(id=msg.id))
        self._bump()
        self.kick()

    def _on_evicted(self, state: _ConnState, msg: p.Evicted) -> None:
        model = self._owned(state, msg.model)
        if model.evict_id == msg.evict_id and model.phase in (Phase.EVICTING, Phase.RESIDENT):
            # RESIDENT here means the eviction had timed out (unresponsive): late, but welcome.
            self._finish_evict(model)
        state.conn.send(p.Ok(id=msg.id))
        self._bump()
        self.kick()

    def _on_evict_refused(self, state: _ConnState, msg: p.EvictRefused) -> None:
        model = self._owned(state, msg.model)
        if model.evict_id == msg.evict_id and model.phase is Phase.EVICTING:
            log.info("%s refused eviction: %s", model.id, msg.reason or "in use")
            self._cancel_evict_timer(model)
            model.phase = Phase.RESIDENT
            model.evict_id = None
            model.protected_until = self.clock.now() + self.settings.refuse_cooldown_s
            self.clock.call_later(self.settings.refuse_cooldown_s, self.kick)
        state.conn.send(p.Ok(id=msg.id))
        self.kick()

    def _on_pin(self, state: _ConnState, msg: p.Pin | p.Unpin) -> None:
        model = self._resolve(state, msg.model)
        model.manual_pin = isinstance(msg, p.Pin)
        log.info("%s %s", model.id, "pinned" if model.manual_pin else "unpinned")
        state.conn.send(p.Ok(id=msg.id, model=model.id))
        self.kick()

    def _on_evict_model(self, state: _ConnState, msg: p.EvictModel) -> None:
        model = self._resolve(state, msg.model)
        if model.phase is Phase.RESIDENT and model.leases:
            raise ArbiterError(f"{model.id} is busy ({len(model.leases)} lease(s)); not evicting")
        if model.phase is Phase.LOADING:
            raise ArbiterError(f"{model.id} is loading; not evicting")
        if model.phase is Phase.RESIDENT:
            self._start_evict(model, reason=f"requested by {state.name}")
        state.conn.send(p.Ok(id=msg.id, model=model.id))

    def _on_profile(self, state: _ConnState, msg: p.Profile) -> None:
        if msg.name and msg.name not in self.settings.profiles:
            known = ", ".join(sorted(self.settings.profiles)) or "none configured"
            raise ArbiterError(f"unknown profile {msg.name!r} ({known})")
        self.active_profile = msg.name
        for model in self.registry.models.values():
            model.profile_pin = self._profile_matches(model.id)
        log.info("profile %s", f"{msg.name!r} active" if msg.name else "cleared")
        state.conn.send(p.Ok(id=msg.id))
        self.kick()

    def _on_status(self, state: _ConnState, msg: p.StatusRequest) -> None:
        self._spawn(self._reply_status(state.conn, msg.id), "vrambiter-status")

    _HANDLERS: Mapping[type[p.Message], Callable[[Arbiter, _ConnState, Any], None]] = {
        p.Register: _on_register,
        p.Acquire: _on_acquire,
        p.Cancel: _on_cancel,
        p.Loaded: _on_loaded,
        p.LoadFailed: _on_load_failed,
        p.Release: _on_release,
        p.Unloaded: _on_unloaded,
        p.Evicted: _on_evicted,
        p.EvictRefused: _on_evict_refused,
        p.Pin: _on_pin,
        p.Unpin: _on_pin,
        p.EvictModel: _on_evict_model,
        p.Profile: _on_profile,
        p.StatusRequest: _on_status,
    }

    def _profile_matches(self, model_id: str) -> bool:
        patterns = (
            self.settings.profiles.get(self.active_profile, []) if self.active_profile else []
        )
        return any(fnmatch.fnmatchcase(model_id, pat) for pat in patterns)

    # ------------------------------------------------------------------------------ internal API
    # used by the CLI-less paths: autostart, warm-up, tests

    def acquire_internal(
        self, model_id: str, *, timeout_s: float | None = None
    ) -> asyncio.Future[str]:
        """Acquire a lease on behalf of the arbiter itself; resolves to the lease id."""
        model = self.registry.models.get(model_id)
        if model is None:
            raise UnknownModel(f"unknown model {model_id!r}")
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        def done(lease: str | None, error: VrambiterError | None) -> None:
            if future.done():
                return
            if error is not None:
                future.set_exception(error)
            else:
                assert lease is not None
                future.set_result(lease)

        self._enqueue(
            conn=None,
            request_id=None,
            requester="vrambiter",
            model=model,
            wait=True,
            timeout_s=timeout_s,
            on_done=done,
        )
        return future

    def release_internal(self, lease_id: str) -> None:
        self._end_lease(lease_id, self.clock.now())
        self._bump()
        self.kick()

    # ------------------------------------------------------------------------------ grants

    def _grant(self, q: _Queued, model: ModelRecord, action: str) -> None:
        self._dequeue(q)
        now = self.clock.now()
        conn_id = q.conn.id if q.conn is not None else None
        lease = self.registry.add_lease(model, q.requester, conn_id, now, action)
        if action == "load":
            model.phase = Phase.LOADING
            model.load_started_at = now
            if model.owner_kind is not OwnerKind.COOPERATIVE:
                # The arbiter loads it; the consumer hears "ready" once it is resident.
                lease.action = "drive"
                self._spawn(self._drive_load(model, q, lease.id), f"vrambiter-load-{model.id}")
                return
        log.debug("granted %s to %s (%s)", model.id, q.requester, action)
        self._reply_granted(q, lease.id, action, model)

    def _reply_granted(self, q: _Queued, lease_id: str, action: str, model: ModelRecord) -> None:
        q.done = True
        if q.on_done is not None:
            q.on_done(lease_id, None)
        if q.conn is not None:
            if q.conn.id not in self._conns:
                # Gone while we worked: the lease was already ended by the disconnect.
                return
            q.conn.send(p.Granted(id=q.request_id, lease=lease_id, action=action, model=model.id))

    def _reply_error(self, q: _Queued, error: VrambiterError) -> None:
        q.done = True
        if q.on_done is not None:
            q.on_done(None, error)
        if q.conn is not None and q.conn.id in self._conns:
            q.conn.send(p.Error(id=q.request_id, **error.to_wire()))

    def _dequeue(self, q: _Queued) -> None:
        if q.timer is not None:
            q.timer.cancel()
            q.timer = None
        with contextlib.suppress(ValueError):
            self.queue.remove(q)

    def _timed_out(self, q: _Queued) -> None:
        if q.done or q not in self.queue:
            return
        self._dequeue(q)
        model = self.registry.models.get(q.model_id)
        waited = self.clock.now() - q.enqueued_at
        self._reply_error(q, self._unavailable(model, f"timed out after {waited:.1f}s: {q.reason}"))
        log.info("acquire of %s by %s timed out (%s)", q.model_id, q.requester, q.reason)

    def _unavailable(self, model: ModelRecord | None, reason: str) -> VramUnavailable:
        snap = self._last_snapshot
        if model is None:
            return VramUnavailable(reason=reason)
        req = request_for(model)
        need = req.need if req is not None else model.vram_peak
        free = (
            max(0, snap.effective_free(model.device))
            if snap and model.device in snap.devices
            else 0
        )
        return VramUnavailable(
            model=model.id,
            device=model.device,
            need=need,
            free=free,
            reason=reason,
            holders=holders(snap, model.device) if snap else (),
        )

    def _end_lease(self, lease_id: str, now: float) -> None:
        self.registry.end_lease(lease_id, now)

    # ------------------------------------------------------------------------------ evictions

    def _start_evict(self, model: ModelRecord, reason: str) -> None:
        sat = self.registry.satellites.get(model.satellite)
        model.phase = Phase.EVICTING
        model.evict_id = self.registry.next_evict_id()
        model.evict_reason = reason
        if self._last is not None:
            model.evict_usage_before = self._last.gpu.pid_usage(
                self._model_pids(model, sat), model.device
            )
            model.evict_free_before = self._last.gpu.free.get(model.device, 0)
        evict_id = model.evict_id
        self._evict_timers[model.id] = self.clock.call_later(
            self.settings.evict_timeout_s, lambda: self._evict_timed_out(model.id, evict_id)
        )
        log.info("evicting %s (%s): %s", model.id, format_bytes(model.resident_estimate()), reason)
        if model.owner_kind is OwnerKind.COOPERATIVE:
            if sat is None or sat.conn_id is None or sat.conn_id not in self._conns:
                self._finish_evict(model)  # owner already gone; disconnect handling frees it
                return
            self._conns[sat.conn_id].conn.send(
                p.Evict(evict_id=evict_id, model=model.name, reason=reason)
            )
        else:
            self._spawn(self._drive_unload(model, evict_id), f"vrambiter-unload-{model.id}")

    def _model_pids(self, model: ModelRecord, sat: SatelliteRecord | None) -> frozenset[int]:
        if model.pids:
            return model.pids
        return sat.all_pids() if sat is not None else frozenset()

    def _cancel_evict_timer(self, model: ModelRecord) -> None:
        timer = self._evict_timers.pop(model.id, None)
        if timer is not None:
            timer.cancel()

    def _finish_evict(self, model: ModelRecord) -> None:
        self._cancel_evict_timer(model)
        now = self.clock.now()
        expected = model.resident_estimate()
        sat = self.registry.satellites.get(model.satellite)
        self._add_settle(
            sat_name=model.satellite,
            device=model.device,
            pids=self._model_pids(model, sat),
            expected=expected,
            usage_before=model.evict_usage_before,
            free_before=model.evict_free_before,
            model_id=model.id,
        )
        model.phase = Phase.UNLOADED
        model.unresponsive = False
        model.evict_id = None
        model.pids = frozenset()
        model.last_used = now
        log.info("%s evicted", model.id)

    def _evict_timed_out(self, model_id: str, evict_id: str) -> None:
        model = self.registry.models.get(model_id)
        self._evict_timers.pop(model_id, None)
        if model is None or model.evict_id != evict_id or model.phase is not Phase.EVICTING:
            return
        sat = self.registry.satellites.get(model.satellite)
        stop = self._force_stop.get(model.satellite)
        only_model = (
            sum(
                1 for m in self.registry.models_of(model.satellite) if m.phase is not Phase.UNLOADED
            )
            == 1
        )
        if (
            stop is not None
            and sat is not None
            and sat.managed
            and (only_model or self.settings.kill_on_evict_timeout)
        ):
            log.warning("evicting %s timed out; stopping its process", model.id)
            self._spawn(stop(), f"vrambiter-kill-{model.satellite}")
            return
        log.warning(
            "evicting %s timed out after %.0fs; marking it unresponsive",
            model.id,
            self.settings.evict_timeout_s,
        )
        model.phase = Phase.RESIDENT
        model.unresponsive = True  # keep evict_id: a late `evicted` is still accepted
        self.kick()

    def _model_gone(self, model: ModelRecord, now: float, reason: str) -> None:
        """A model's memory is going away without an eviction round trip (unloaded voluntarily,
        owner disconnected, process exited). Ends its leases and expects its memory back."""
        sat = self.registry.satellites.get(model.satellite)
        self._cancel_evict_timer(model)
        reserve = 0
        if model.phase is Phase.LOADING:
            reserve = model.vram_peak  # held until the driver shows the process's memory gone
        expected = model.resident_estimate() if model.phase is not Phase.LOADING else 0
        usage = None
        free = 0
        pids = self._model_pids(model, sat)
        if self._last is not None:
            usage = self._last.gpu.pid_usage(pids, model.device) if pids else None
            free = self._last.gpu.free.get(model.device, 0)
        if expected or reserve:
            self._add_settle(
                sat_name=model.satellite,
                device=model.device,
                pids=pids,
                expected=expected,
                usage_before=usage,
                free_before=free,
                model_id=model.id,
                reserve=reserve,
            )
        for lease_id in list(model.leases):
            lease = self.registry.leases.get(lease_id)
            if (
                lease is not None
                and lease.conn_id in self._conns
                and lease.holder != model.satellite
            ):
                self._conns[lease.conn_id].conn.send(
                    p.Notice(
                        kind="lease_lost",
                        message=f"{model.id}: {reason}",
                        data={"lease": lease_id, "model": model.id},
                    )
                )
            self._end_lease(lease_id, now)
        model.phase = Phase.UNLOADED
        model.evict_id = None
        model.load_started_at = None
        model.pids = frozenset()
        # Queued requests for a cooperative model whose owner left cannot be served.
        if model.owner_kind is OwnerKind.COOPERATIVE and (sat is None or not sat.connected):
            for q in [q for q in self.queue if q.model_id == model.id]:
                self._dequeue(q)
                self._reply_error(q, ArbiterError(f"{model.id}: its satellite went away"))

    def _release_satellite_memory(self, sat: SatelliteRecord, now: float) -> None:
        for model in self.registry.models_of(sat.name):
            if model.phase is not Phase.UNLOADED:
                self._model_gone(model, now, reason=f"{sat.name} disconnected")
        for q in [q for q in self.queue if q.model_id.startswith(sat.name + "/")]:
            model = self.registry.models.get(q.model_id)
            if model is not None and model.owner_kind is OwnerKind.COOPERATIVE:
                self._dequeue(q)
                self._reply_error(q, ArbiterError(f"{q.model_id}: its satellite went away"))

    def _add_settle(
        self,
        *,
        sat_name: str,
        device: int,
        pids: frozenset[int],
        expected: int,
        usage_before: int | None,
        free_before: int,
        model_id: str | None,
        reserve: int = 0,
    ) -> None:
        key = self.registry.next_settle_key()
        self.registry.settling[key] = Settle(
            key=key,
            satellite=sat_name,
            device=device,
            pids=pids,
            expected=expected,
            usage_before=usage_before,
            free_before=free_before,
            deadline=self.clock.now() + self.settings.settle_timeout_s,
            reserve=reserve,
            model_id=model_id,
        )
        self._schedule_settle_check()

    def _schedule_settle_check(self) -> None:
        if self._settle_timer is None and not self._closed:

            def fire() -> None:
                self._settle_timer = None
                self.kick()

            self._settle_timer = self.clock.call_later(self.settings.settle_poll_s, fire)

    # ------------------------------------------------------------------------------ drivers

    async def _drive_load(self, model: ModelRecord, q: _Queued, lease_id: str) -> None:
        driver = self._drivers[model.satellite]
        try:
            result = await driver.load(model.name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("loading %s failed: %s", model.id, exc)
            if model.phase is Phase.LOADING:
                model.phase = Phase.UNLOADED
                model.load_started_at = None
            self._end_lease(lease_id, self.clock.now())
            self._reply_error(
                q, ArbiterError(f"loading {model.id} failed: {exc}", detail={"code": "load_failed"})
            )
            self._bump()
            self.kick()
            return
        if model.phase is not Phase.LOADING:
            return  # the process died meanwhile; satellite_process_exited handled it
        model.phase = Phase.RESIDENT
        model.load_started_at = None
        model.reported_bytes = result.vram_bytes
        model.measured_bytes = None
        model.pids = result.pids
        model.last_used = self.clock.now()
        lease = self.registry.leases.get(lease_id)
        if lease is not None:
            lease.action = "ready"
            self._reply_granted(q, lease_id, "ready", model)
        self._bump()
        self.kick()

    async def _drive_unload(self, model: ModelRecord, evict_id: str) -> None:
        driver = self._drivers[model.satellite]
        try:
            await driver.unload(model.name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("unloading %s failed: %s", model.id, exc)
            return  # the evict timeout decides what happens next
        if model.evict_id == evict_id and model.phase in (Phase.EVICTING, Phase.RESIDENT):
            self._finish_evict(model)
            self._bump()
            self.kick()

    # ------------------------------------------------------------------------------ reconcile

    def _bump(self) -> None:
        """An event lowered reservations or pending returns; measurements in flight are stale."""
        self._release_gen += 1

    def _schedule_poll(self) -> None:
        if self._closed:
            return

        def fire() -> None:
            self._poll_timer = None
            self.kick()
            self._schedule_poll()

        self._poll_timer = self.clock.call_later(self.settings.poll_interval_s, fire)

    def _roots(self) -> dict[str, frozenset[int]]:
        roots: dict[str, frozenset[int]] = {}
        for sat in self.registry.satellites.values():
            pids = frozenset({sat.pid}) if sat.pid and sat.connected else frozenset()
            driver = self._drivers.get(sat.name)
            if driver is not None:
                pids |= driver.root_pids()
            roots[sat.name] = pids
        return roots

    def _measure_blocking(self, roots: dict[str, frozenset[int]]) -> _Measurement:
        gpu = take_snapshot(self.gpu)
        trees: dict[str, frozenset[int]] = {}
        for name, pids in roots.items():
            tree: set[int] = set()
            for pid in pids:
                tree |= process_tree(pid)
            trees[name] = frozenset(tree)
        return _Measurement(gpu, self.host.mem_available(), self.host.mem_total(), trees)

    async def _measure(self) -> _Measurement:
        return await asyncio.to_thread(self._measure_blocking, self._roots())

    async def _reconcile_loop(self) -> None:
        while not self._closed:
            await self._wake.wait()
            self._planning = True
            self._wake.clear()
            if self._closed:
                return
            try:
                await self._reconcile()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("planning pass failed; retrying at the next poll")
            finally:
                self._planning = False
            self.passes += 1

    async def _reconcile(self) -> None:
        measurement: _Measurement | None = None
        # Only requests that arrived before the measurement started are planned in this pass, so
        # every decision is made on numbers taken after the request was made.
        seq_limit = self._enqueued_seq
        for _ in range(5):
            gen = self._release_gen
            measurement = await self._measure()
            if gen == self._release_gen:
                break
        else:
            # Events keep arriving faster than we can measure; plan on the next wake-up.
            self.kick()
            return
        assert measurement is not None
        if self._closed:
            return
        self._last = measurement
        now = self.clock.now()
        for name, tree in measurement.trees.items():
            sat = self.registry.satellites.get(name)
            if sat is not None:
                sat.pids = tree
        self._learn(measurement.gpu)
        self._settle(measurement.gpu, now)
        snapshot = build_snapshot(
            self.registry,
            measurement.gpu,
            host_available=measurement.host_available,
            headroom=self.settings.headroom,
            host_headroom=self.settings.host_headroom,
            max_concurrent_loads=self.settings.max_concurrent_loads,
            now=now,
        )
        self._last_snapshot = snapshot
        self._plan(snapshot, seq_limit)

    def _learn(self, gpu: GpuSnapshot) -> None:
        """Attribute measured usage to idle models; learn each satellite's baseline."""
        for sat in self.registry.satellites.values():
            pids = sat.all_pids()
            if not pids:
                continue
            models = self.registry.models_of(sat.name)
            for device in gpu.devices:
                on_dev = [m for m in models if m.device == device]
                own = [m for m in on_dev if m.pids]
                shared = [m for m in on_dev if not m.pids]
                for m in own:
                    if m.phase is Phase.RESIDENT and not m.leases:
                        usage = gpu.pid_usage(m.pids, device)
                        if usage is not None:
                            self._set_measured(m, usage)
                own_pids = frozenset().union(*(m.pids for m in own)) if own else frozenset()
                usage = gpu.pid_usage(pids - own_pids, device)
                if usage is None:
                    continue
                if all(m.phase is Phase.UNLOADED for m in shared):
                    if not any(s.satellite == sat.name for s in self.registry.settling.values()):
                        sat.baseline[device] = usage
                    continue
                for model_id, nbytes in attribute_usage(
                    shared, usage, sat.baseline.get(device, 0)
                ).items():
                    self._set_measured(self.registry.models[model_id], nbytes)

    def _set_measured(self, model: ModelRecord, nbytes: int) -> None:
        model.measured_bytes = nbytes
        declared = model.vram_resident or model.reported_bytes
        if declared and not model.mismatch_logged and abs(nbytes - declared) > 0.2 * declared:
            model.mismatch_logged = True
            log.warning(
                "%s holds %s while idle; declared/reported %s (mismatch > 20%%, using the "
                "measurement)",
                model.id,
                format_bytes(nbytes),
                format_bytes(declared),
            )

    def _settle(self, gpu: GpuSnapshot, now: float) -> None:
        for key, settle in list(self.registry.settling.items()):
            if not settle.settled(gpu, now):
                continue
            del self.registry.settling[key]
            outstanding = settle.outstanding(gpu)
            if settle.expected and outstanding > (1 - SETTLE_FRACTION) * settle.expected:
                sat = self.registry.satellites.get(settle.satellite)
                if sat is not None:
                    sat.residue[settle.device] = sat.residue.get(settle.device, 0) + outstanding
                log.warning(
                    "%s: expected %s back on GPU %d, %s did not come back within %.0fs",
                    settle.model_id or settle.satellite,
                    format_bytes(settle.expected),
                    settle.device,
                    format_bytes(outstanding),
                    self.settings.settle_timeout_s,
                )
        if self.registry.settling:
            self._schedule_settle_check()

    def _plan(self, snapshot: Snapshot, seq_limit: int) -> None:
        entries: dict[int, _Queued] = {}
        requests = []
        for q in list(self.queue):
            if q.seq > seq_limit:
                self.kick()  # arrived during the measurement: next pass
                continue
            model = self.registry.models.get(q.model_id)
            if model is None:
                self._dequeue(q)
                self._reply_error(q, UnknownModel(f"{q.model_id} is gone"))
                continue
            if model.phase is Phase.RESIDENT and model.busy:
                self._grant(q, model, "ready")
                continue
            req = request_for(model, seq=q.seq)
            if req is None:
                q.reason = f"{model.id} is {model.phase.value}"
                continue
            entries[q.seq] = q
            requests.append(req)

        for req, decision in plan_queue(requests, snapshot):
            q = entries[req.seq]
            model = self.registry.models[q.model_id]
            if isinstance(decision, Admit):
                self._grant(q, model, "load" if req.is_load else "ready")
            elif isinstance(decision, Evict):
                q.reason = f"evicting {', '.join(decision.victims)}"
                for victim_id in decision.victims:
                    victim = self.registry.models[victim_id]
                    if victim.phase is Phase.RESIDENT and not victim.leases:
                        self._start_evict(victim, reason=f"making room for {model.id}")
            elif isinstance(decision, Wait):
                if q.reason != decision.reason:
                    log.info("%s waits: %s", model.id, decision.reason)
                q.reason = decision.reason
                if not q.wait and decision.kind not in (WaitKind.EVICTING, WaitKind.QUEUED):
                    self._dequeue(q)
                    self._reply_error(q, self._unavailable(model, f"would wait: {decision.reason}"))
            elif isinstance(decision, Fail):
                self._dequeue(q)
                error = decision.to_error(model.id, model.device)
                log.warning("cannot admit %s: %s", model.id, error)
                self._reply_error(q, error)

    # ------------------------------------------------------------------------------ status

    async def _reply_status(self, conn: Connection, request_id: int | None) -> None:
        try:
            measurement = await self._measure()
        except Exception as exc:  # pragma: no cover - backend failure
            conn.send(p.Error(id=request_id, code="internal_error", message=str(exc)))
            return
        if conn.id in self._conns:
            conn.send(p.StatusReply(id=request_id, **self.status(measurement)))

    def status(self, measurement: _Measurement | None = None) -> dict[str, Any]:
        """The ``status`` reply body. Sizes are bytes; times are seconds."""
        m = measurement or self._last
        now = self.clock.now()
        snap = None
        if m is not None:
            snap = build_snapshot(
                self.registry,
                m.gpu,
                host_available=m.host_available,
                headroom=self.settings.headroom,
                host_headroom=self.settings.host_headroom,
                max_concurrent_loads=self.settings.max_concurrent_loads,
                now=now,
            )
        devices = []
        if snap is not None:
            for d in snap.devices.values():
                devices.append(
                    {
                        "index": d.index,
                        "total": d.total,
                        "free": d.free,
                        "reserved": d.reserved,
                        "pending": d.pending,
                        "headroom": self.settings.headroom,
                        "effective_free": d.free - d.reserved - self.settings.headroom,
                    }
                )
        models = []
        for model in sorted(self.registry.models.values(), key=lambda x: x.id):
            models.append(
                {
                    "id": model.id,
                    "satellite": model.satellite,
                    "name": model.name,
                    "device": model.device,
                    "state": model.state_label,
                    "kind": model.owner_kind.value,
                    "leases": len(model.leases),
                    "pinned": model.pinned,
                    "priority": model.priority,
                    "vram_peak": model.vram_peak,
                    "vram_resident": model.vram_resident,
                    "reported": model.reported_bytes,
                    "measured": model.measured_bytes,
                    "estimate": model.resident_estimate(),
                    "host_peak": model.host_peak,
                    "idle_s": round(now - model.last_used, 1) if not model.leases else 0.0,
                    "unresponsive": model.unresponsive,
                }
            )
        satellites = []
        usage = m.gpu.usage if m is not None else {}
        for sat in sorted(self.registry.satellites.values(), key=lambda s: s.name):
            pids = sat.all_pids()
            satellites.append(
                {
                    "name": sat.name,
                    "pid": sat.pid,
                    "connected": sat.connected or sat.name in self._drivers,
                    "managed": sat.managed,
                    "adapter": sat.adapter,
                    "pids": sorted(pids),
                    "usage": {
                        str(dev): sum((per or {}).get(pid, 0) for pid in pids)
                        for dev, per in usage.items()
                    },
                    "residue": {str(k): v for k, v in sat.residue.items()},
                }
            )
        queue = [
            {
                "model": q.model_id,
                "requester": q.requester,
                "priority": self.registry.models[q.model_id].priority
                if q.model_id in self.registry.models
                else 0,
                "waiting_s": round(now - q.enqueued_at, 1),
                "timeout_s": round(q.deadline - now, 1) if q.deadline is not None else None,
                "reason": q.reason,
            }
            for q in sorted(self.queue, key=lambda q: q.seq)
        ]
        foreign = (
            [{"pid": f.pid, "device": f.device, "bytes": f.bytes} for f in snap.foreign]
            if snap
            else []
        )
        return {
            "arbiter": {
                "version": __version__,
                "pid": os.getpid(),
                "uptime_s": round(now - self._started_at, 1),
                "profile": self.active_profile,
                "profiles": sorted(self.settings.profiles),
                "max_concurrent_loads": self.settings.max_concurrent_loads,
            },
            "devices": devices,
            "host": {
                "available": m.host_available if m else None,
                "total": m.host_total if m else None,
                "headroom": self.settings.host_headroom,
                "loads_in_flight": len(self.registry.loading()),
            },
            "models": models,
            "satellites": satellites,
            "queue": queue,
            "foreign": foreign,
        }
