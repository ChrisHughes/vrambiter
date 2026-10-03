"""The satellite side: register models, take leases, unload on request. Standard library only.

    import vrambiter

    arb = vrambiter.connect("my-tts-server")      # NullArbiter (standalone) if no daemon is running
    tts = arb.register("voxcpm2", vram_peak="9GiB", load=load_weights, unload=free_weights)

    with tts.lease():                             # admitted, loaded if needed, busy until exit
        audio = synthesize(text)

HOW IT IS BUILT, AND WHY

* **One background thread, one event loop, one socket per client.** Sync and async callers both
  bridge into it, so a Flask app, a FastAPI app and a plain script use the same code path. Every
  message is written from that loop in the order it was submitted (submission is a thread-safe
  ``call_soon_threadsafe``, never a task that might run later), which is what makes "``loaded``
  before ``release``" and "``evicted`` before the next ``acquire``" hold.

* **Leases are counted locally, and that is the safety net.** A lease increments the model's local
  count *before* its ``acquire`` is sent. An ``evict`` that arrives while the count is non-zero is
  refused on the spot. So even when the arbiter's picture is stale (a reconnect, an eviction that
  timed out and was retried), a model in use in this process is never unloaded under a lease.

* **``load`` and ``unload`` run under a per-model lock**, and the lease path takes the same lock
  to check residency, so an eviction never races a lease: a lease that arrives during an unload
  waits for it to finish and then loads again.

* **Standalone is a mode, not an error.** Without an arbiter, ``lease()`` loads on first use and
  never unloads: exactly what a plain server does. If the arbiter goes away mid-run, the client
  falls back to that behaviour and reconnects in the background, re-registering each model with
  its current state and its live lease count so the new arbiter neither reloads nor evicts it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import contextlib
import itertools
import logging
import os
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Any

from . import protocol as p
from .errors import ArbiterUnavailable, VrambiterError, error_from_wire
from .units import parse_bytes, parse_optional_bytes

__all__ = [
    "connect",
    "default_socket_path",
    "ArbiterClient",
    "NullArbiter",
    "Model",
    "Lease",
]

log = logging.getLogger("vrambiter.client")

LoadFn = Callable[[], Any]
MeasureFn = Callable[[], "int | None"]


def default_socket_path() -> str:
    """Where satellites and the daemon meet, unless configured otherwise.

    ``$VRAMBITER_SOCKET``, else ``$XDG_RUNTIME_DIR/vrambiter.sock`` (systemd sets that to
    ``/run/user/<uid>``), else ``/tmp/vrambiter-<uid>.sock``.
    """
    explicit = os.environ.get("VRAMBITER_SOCKET")
    if explicit:
        return explicit
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return os.path.join(runtime, "vrambiter.sock")
    return f"/tmp/vrambiter-{os.getuid()}.sock"


def connect(
    name: str | None = None,
    *,
    socket: str | None = None,
    required: bool = False,
    wait_s: float = 0.0,
    pid: int | None = None,
    reconnect: bool = True,
) -> ArbiterClient | NullArbiter:
    """Connect to the arbiter, or return a :class:`NullArbiter` if there is none.

    ``name`` identifies this satellite; ``$VRAMBITER_NAME`` overrides it so whoever launches the
    process (``vrambiter run --name``, a managed satellite in the config) decides its identity.
    ``wait_s`` waits that long for a daemon that is still starting. ``required=True`` raises
    :class:`ArbiterUnavailable` instead of falling back to standalone mode. ``pid`` is the process
    whose GPU memory is this satellite's (default: this one; set it when a launcher connects on
    behalf of a child).
    """
    resolved = os.environ.get("VRAMBITER_NAME") or name
    if not resolved:
        raise ValueError("pass a satellite name to connect() or set VRAMBITER_NAME")
    path = socket or default_socket_path()
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        try:
            return ArbiterClient(resolved, socket=path, pid=pid, reconnect=reconnect)
        except ArbiterUnavailable as exc:
            if time.monotonic() >= deadline:
                if required:
                    raise
                log.info("no arbiter at %s (%s); running standalone", path, exc.message)
                return NullArbiter(resolved)
            time.sleep(0.1)


# --------------------------------------------------------------------------- leases and models


class _ConnectionLost(Exception):
    """Internal: the request could not complete because the link to the arbiter dropped."""


@dataclass(frozen=True)
class _Grant:
    lease_id: str | None  # None: standalone (no arbiter involved)
    action: str  # "load" | "ready" | "standalone"
    epoch: int


_STANDALONE = _Grant(None, "standalone", -1)


class Lease:
    """A claim that a model is in use. Release it exactly once (context managers do).

    ``model`` is the short name for a model this satellite owns, or the full id for a consumer
    lease. ``standalone`` is true when no arbiter was involved.
    """

    def __init__(self, owner: _LeaseOwner, model: str, grant: _Grant) -> None:
        self._owner = owner
        self.model = model
        self._grant = grant
        self.arbiter_lease: str | None = grant.lease_id
        self.epoch = grant.epoch
        self.released = False
        self._lock = threading.Lock()

    @property
    def standalone(self) -> bool:
        return self.arbiter_lease is None

    @property
    def action(self) -> str:
        """``"load"`` if this lease made the model resident, ``"ready"``, or ``"standalone"``."""
        return self._grant.action

    def release(self) -> None:
        """End the lease. Idempotent, non-blocking, safe from any thread or event loop."""
        with self._lock:
            if self.released:
                return
            self.released = True
        self._owner._lease_ended(self)

    def __enter__(self) -> Lease:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def __repr__(self) -> str:
        state = "released" if self.released else "held"
        return f"<Lease {self.model} {self.arbiter_lease or 'standalone'} {state}>"


class _LeaseOwner:
    def _lease_ended(self, lease: Lease) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class _CancelToken:
    """Lets an async caller abandon a sync acquire that is blocked in another thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._future: cf.Future[Any] | None = None
        self.cancelled = False

    def attach(self, future: cf.Future[Any]) -> None:
        with self._lock:
            self._future = future
            if self.cancelled:
                future.cancel()

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True
            if self._future is not None:
                self._future.cancel()


class Model(_LeaseOwner):
    """A model this satellite can load. Created by :meth:`ArbiterClient.register`."""

    def __init__(
        self,
        arbiter: ArbiterClient | NullArbiter,
        name: str,
        *,
        vram_peak: int,
        vram_resident: int | None,
        host_peak: int,
        priority: int,
        pinned: bool,
        device: int,
        load: LoadFn,
        unload: LoadFn | None,
        measure: MeasureFn | None,
        cleanup: bool,
    ) -> None:
        self._arbiter = arbiter
        self.name = name
        self.vram_peak = vram_peak
        self.vram_resident = vram_resident
        self.host_peak = host_peak
        self.priority = priority
        self.pinned = pinned
        self.device = device
        self._load_fn = load
        self._unload_fn = unload
        self._measure_fn = measure
        self._cleanup = cleanup
        # The heavy lock: held for the whole of load() and unload().
        self._lock = threading.RLock()
        # The light lock: held for nanoseconds, guards the lease count and set.
        self._count_lock = threading.Lock()
        self._active: list[Lease] = []
        self._resident = False
        self.vram_bytes: int | None = None
        self.last_used = 0.0
        #: The connection epoch on which the arbiter accepted this model's registration.
        self._registered_epoch = -1

    # -- public ---------------------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._resident

    @property
    def leases(self) -> int:
        with self._count_lock:
            return len(self._active)

    @property
    def id(self) -> str:
        return f"{self._arbiter.name}/{self.name}"

    def acquire(self, *, timeout: float | None = None, wait: bool = True) -> Lease:
        """Block until admitted, load if needed, and return a held :class:`Lease`."""
        return self._begin(timeout=timeout, wait=wait, token=None)

    @contextlib.contextmanager
    def lease(self, *, timeout: float | None = None, wait: bool = True) -> Iterator[Lease]:
        """``with model.lease():`` - the model is resident and busy inside the block.

        ``timeout`` bounds the wait for admission (the arbiter fails it with
        :class:`~vrambiter.errors.VramUnavailable`); ``wait=False`` fails instead of queueing.
        """
        lease = self.acquire(timeout=timeout, wait=wait)
        try:
            yield lease
        finally:
            lease.release()

    async def aacquire(self, *, timeout: float | None = None, wait: bool = True) -> Lease:
        """Async :meth:`acquire`. The wait never blocks the caller's loop; ``load`` runs in a
        worker thread. Cancelling the awaiting task withdraws the request."""
        token = _CancelToken()
        loop = asyncio.get_running_loop()
        job = loop.run_in_executor(
            None, lambda: self._begin(timeout=timeout, wait=wait, token=token)
        )
        try:
            return await asyncio.shield(job)
        except asyncio.CancelledError:
            token.cancel()

            def _release_late(done: asyncio.Future[Lease]) -> None:
                if not done.cancelled() and done.exception() is None:
                    done.result().release()

            job.add_done_callback(_release_late)
            raise

    @contextlib.asynccontextmanager
    async def alease(
        self, *, timeout: float | None = None, wait: bool = True
    ) -> AsyncIterator[Lease]:
        """``async with model.alease():`` - asyncio variant of :meth:`lease`."""
        lease = await self.aacquire(timeout=timeout, wait=wait)
        try:
            yield lease
        finally:
            lease.release()

    def unload(self) -> bool:
        """Unload now if no lease is held, and tell the arbiter. Returns whether it unloaded."""
        with self._lock:
            with self._count_lock:
                if self._active:
                    return False
            if not self._resident:
                return False
            self._do_unload()
            self._arbiter._notify(p.Unloaded(model=self.name))
        return True

    # -- lease machinery ---------------------------------------------------------------------------

    def _begin(self, *, timeout: float | None, wait: bool, token: _CancelToken | None) -> Lease:
        placeholder = Lease(self, self.name, _STANDALONE)
        with self._count_lock:
            # Counted before the acquire is sent: from here on an evict for this model is refused.
            self._active.append(placeholder)
        try:
            grant = self._arbiter._acquire(self.name, wait=wait, timeout=timeout, token=token)
            self._ensure_resident(grant)
        except BaseException:
            with self._count_lock:
                self._active.remove(placeholder)
            raise
        placeholder._grant = grant
        if grant.lease_id is not None:
            placeholder.arbiter_lease = grant.lease_id
            placeholder.epoch = grant.epoch
        # else: standalone, unless a reconnect adopted this lease meanwhile (then keep that id).
        self.last_used = time.monotonic()
        return placeholder

    def _ensure_resident(self, grant: _Grant) -> None:
        with self._lock:
            if self._resident:
                if grant.action == "load":
                    # The arbiter thought it unloaded (a reconnect); it is here, say so.
                    self._arbiter._notify(p.Loaded(model=self.name, vram_bytes=self.vram_bytes))
                return
            try:
                self.vram_bytes = self._do_load()
            except BaseException as exc:
                if grant.lease_id is not None:
                    self._arbiter._notify(
                        p.LoadFailed(model=self.name, error=f"{type(exc).__name__}: {exc}"),
                        epoch=grant.epoch,
                    )
                    self._arbiter._notify(p.Release(lease=grant.lease_id), epoch=grant.epoch)
                raise
            if grant.lease_id is not None:
                if grant.action != "load":
                    log.warning("%s: granted 'ready' but not resident here; loaded it", self.name)
                self._arbiter._notify(
                    p.Loaded(model=self.name, vram_bytes=self.vram_bytes), epoch=grant.epoch
                )

    def _do_load(self) -> int | None:
        before = None if self._measure_fn else _torch_reserved(self.device)
        self._load_fn()
        self._resident = True
        if self._measure_fn is not None:
            try:
                return self._measure_fn()
            except Exception:
                log.warning("%s: measure() failed", self.name, exc_info=True)
                return None
        after = _torch_reserved(self.device)
        if before is not None and after is not None:
            return max(0, after - before)
        return None

    def _do_unload(self) -> None:
        if self._unload_fn is not None:
            self._unload_fn()
        self._resident = False
        self.vram_bytes = None
        if self._cleanup:
            _release_cached_memory()

    def _lease_ended(self, lease: Lease) -> None:
        with self._count_lock, contextlib.suppress(ValueError):
            self._active.remove(lease)
        self.last_used = time.monotonic()
        if lease.arbiter_lease is not None:
            self._arbiter._notify(p.Release(lease=lease.arbiter_lease), epoch=lease.epoch)

    def _handle_evict(self, evict: p.Evict, epoch: int) -> None:
        """Called on a worker thread for an ``evict`` from the arbiter (on connection ``epoch``)."""
        with self._lock:
            with self._count_lock:
                busy = bool(self._active)
            if busy:
                self._arbiter._notify(
                    p.EvictRefused(model=self.name, evict_id=evict.evict_id, reason="in use"),
                    epoch=epoch,
                )
                return
            if self._resident:
                try:
                    self._do_unload()
                except Exception as exc:
                    log.exception("%s: unload failed during eviction", self.name)
                    self._arbiter._notify(
                        p.EvictRefused(
                            model=self.name, evict_id=evict.evict_id, reason=f"unload failed: {exc}"
                        ),
                        epoch=epoch,
                    )
                    return
                log.info("%s evicted (%s)", self.name, evict.reason or "no reason given")
            # Sent while still holding the lock, so it precedes any acquire a waiting lease sends.
            self._arbiter._notify(p.Evicted(model=self.name, evict_id=evict.evict_id), epoch=epoch)

    # -- re-registration ---------------------------------------------------------------------------

    def _register_message(self) -> tuple[p.Register, list[Lease]]:
        """The ``register`` describing this model as it is now, plus the leases to adopt."""
        with self._count_lock:
            held = [lease for lease in self._active if not lease.released]
        return (
            p.Register(
                model=self.name,
                vram_peak=self.vram_peak,
                vram_resident=self.vram_resident,
                host_peak=self.host_peak,
                priority=self.priority,
                pinned=self.pinned,
                device=self.device,
                state="resident" if self._resident else "unloaded",
                vram_bytes=self.vram_bytes,
                leases=len(held) if self._resident else 0,
            ),
            held if self._resident else [],
        )

    def __repr__(self) -> str:
        return (
            f"<Model {self.id} {'resident' if self._resident else 'unloaded'} leases={self.leases}>"
        )


class _ConsumerLeases(_LeaseOwner):
    """Leases on models owned by someone else (adapter-backed, managed), by full id."""

    def __init__(self, arbiter: ArbiterClient | NullArbiter) -> None:
        self._arbiter = arbiter

    def begin(
        self, model_id: str, timeout: float | None, wait: bool, token: _CancelToken | None
    ) -> Lease:
        grant = self._arbiter._acquire(model_id, wait=wait, timeout=timeout, token=token)
        return Lease(self, model_id, grant)

    def _lease_ended(self, lease: Lease) -> None:
        if lease.arbiter_lease is not None:
            self._arbiter._notify(p.Release(lease=lease.arbiter_lease), epoch=lease.epoch)


# --------------------------------------------------------------------------- the arbiters


class _Base:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self._models: dict[str, Model] = {}
        self._models_lock = threading.Lock()
        self._consumer = _ConsumerLeases(self)  # type: ignore[arg-type]

    @property
    def connected(self) -> bool:
        return False

    @property
    def models(self) -> dict[str, Model]:
        with self._models_lock:
            return dict(self._models)

    def register(
        self,
        name: str,
        *,
        vram_peak: int | str,
        load: LoadFn,
        unload: LoadFn | None = None,
        vram_resident: int | str | None = None,
        host_peak: int | str = 0,
        priority: int = 0,
        pinned: bool = False,
        device: int = 0,
        measure: MeasureFn | None = None,
        cleanup: bool = True,
    ) -> Model:
        """Declare a model this satellite can load.

        ``vram_peak`` is the most VRAM the model needs while loading *or working* (admission
        reserves it); ``vram_resident`` what it holds while idle (optional, measured anyway);
        ``host_peak`` the host RAM a load needs. Sizes accept ints (bytes) or strings like
        ``"9GiB"``. ``load()`` makes the model resident; ``unload()`` drops it (after which, with
        ``cleanup=True``, the library runs ``gc.collect()`` and ``torch.cuda.empty_cache()`` if
        torch is imported). ``measure()`` may return the bytes the model holds; by default the
        growth of ``torch.cuda.memory_reserved()`` across ``load()`` is reported when torch is
        in use.
        """
        if not name or "\n" in name:
            raise ValueError(f"invalid model name {name!r}")
        model = Model(
            self,  # type: ignore[arg-type]
            name,
            vram_peak=parse_bytes(vram_peak),
            vram_resident=parse_optional_bytes(vram_resident),
            host_peak=parse_bytes(host_peak),
            priority=int(priority),
            pinned=bool(pinned),
            device=int(device),
            load=load,
            unload=unload,
            measure=measure,
            cleanup=cleanup,
        )
        with self._models_lock:
            if name in self._models:
                raise ValueError(f"model {name!r} is already registered")
            self._models[name] = model
        try:
            self._announce(model)
        except BaseException:
            with self._models_lock:
                self._models.pop(name, None)
            raise
        return model

    def _announce(self, model: Model) -> None:
        pass

    def acquire(self, model_id: str, *, timeout: float | None = None, wait: bool = True) -> Lease:
        """A consumer lease on another satellite's model, by full id (``"llama/gemma-4-26b"``)."""
        return self._consumer.begin(model_id, timeout, wait, None)

    @contextlib.contextmanager
    def lease(
        self, model_id: str, *, timeout: float | None = None, wait: bool = True
    ) -> Iterator[Lease]:
        """``with arb.lease("llama/gemma-4-26b"):`` - the model is resident and busy inside."""
        lease = self.acquire(model_id, timeout=timeout, wait=wait)
        try:
            yield lease
        finally:
            lease.release()

    async def aacquire(
        self, model_id: str, *, timeout: float | None = None, wait: bool = True
    ) -> Lease:
        token = _CancelToken()
        loop = asyncio.get_running_loop()
        job = loop.run_in_executor(
            None, lambda: self._consumer.begin(model_id, timeout, wait, token)
        )
        try:
            return await asyncio.shield(job)
        except asyncio.CancelledError:
            token.cancel()
            job.add_done_callback(
                lambda d: (
                    d.result().release() if not d.cancelled() and d.exception() is None else None
                )
            )
            raise

    @contextlib.asynccontextmanager
    async def alease(
        self, model_id: str, *, timeout: float | None = None, wait: bool = True
    ) -> AsyncIterator[Lease]:
        lease = await self.aacquire(model_id, timeout=timeout, wait=wait)
        try:
            yield lease
        finally:
            lease.release()

    # Hooks the models call.
    def _acquire(
        self, model: str, *, wait: bool, timeout: float | None, token: _CancelToken | None
    ) -> _Grant:
        return _STANDALONE

    def _notify(self, message: p.Message, epoch: int | None = None) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullArbiter(_Base):
    """Standalone mode: the same API with no arbiter behind it.

    ``lease()`` loads a model on first use and keeps it resident; nothing is ever evicted;
    consumer leases are no-ops. A satellite written against vrambiter therefore behaves exactly
    like a plain server when the daemon is not running.
    """

    def _unavailable(self, *args: object, **kwargs: object) -> Any:
        raise ArbiterUnavailable("standalone mode: no arbiter is connected")

    status = pin = unpin = evict = profile = _unavailable

    def __repr__(self) -> str:
        return f"<NullArbiter {self.name} standalone>"


class ArbiterClient(_Base):
    """A live connection to the arbiter. Use :func:`connect` rather than constructing it."""

    def __init__(
        self,
        name: str,
        *,
        socket: str | None = None,
        pid: int | None = None,
        role: str = "satellite",
        reconnect: bool = True,
        connect_timeout: float = 5.0,
    ) -> None:
        super().__init__(name)
        self.socket_path = socket or default_socket_path()
        self.pid = pid if pid is not None else os.getpid()
        self.role = role
        self._reconnect_enabled = reconnect and role == "satellite"
        self._ids = itertools.count(1)
        self._state_lock = threading.Lock()
        self._pending: dict[int, cf.Future[p.Message]] = {}
        self._abandoned: set[int] = set()
        self._link_up = False  # the socket is usable
        self._ready = False  # ...and models are registered: user requests may use it
        self._epoch = 0
        self._closed = False
        self._writer: asyncio.StreamWriter | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self.arbiter_version = ""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name=f"vrambiter-client-{name}", daemon=True
        )
        self._thread.start()
        try:
            asyncio.run_coroutine_threadsafe(self._open(), self._loop).result(connect_timeout)
        except BaseException as exc:
            self._shutdown_loop()
            if isinstance(exc, ArbiterUnavailable):
                raise
            if isinstance(exc, VrambiterError):
                raise
            raise ArbiterUnavailable(f"cannot connect to {self.socket_path}: {exc}") from exc

    # -- loop thread ---------------------------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        with contextlib.suppress(Exception):
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self._loop.close()

    def _shutdown_loop(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=5)

    async def _open(self) -> None:
        """Connect, say hello, (re-)register models, then accept user requests."""
        try:
            reader, writer = await asyncio.open_unix_connection(
                self.socket_path, limit=p.MAX_LINE_BYTES + 1
            )
        except (OSError, ValueError) as exc:
            raise ArbiterUnavailable(f"cannot connect to {self.socket_path}: {exc}") from None
        with self._state_lock:
            self._writer = writer
            self._link_up = True
            self._epoch += 1
        self._reader_task = asyncio.create_task(self._read_loop(reader, self._epoch))
        try:
            welcome = await self._request_async(
                p.Hello(name=self.name, pid=self.pid, version=_version(), role=self.role)
            )
            self.arbiter_version = getattr(welcome, "arbiter_version", "")
            for model in self.models.values():
                try:
                    await self._register_async(model)
                except ArbiterUnavailable:
                    raise
                except VrambiterError as exc:
                    # One model the new arbiter rejects (a device it does not have, say) must
                    # not keep the others from coordinating. It stays usable standalone.
                    log.error("re-registering %s failed: %s", model.name, exc)
        except BaseException:
            writer.close()
            with self._state_lock:
                self._link_up = False
            raise
        with self._state_lock:
            self._ready = True
        log.info(
            "connected to arbiter %s at %s as %s", self.arbiter_version, self.socket_path, self.name
        )

    async def _register_async(self, model: Model) -> None:
        message, held = model._register_message()
        reply = await self._request_async(message)
        model._registered_epoch = self._epoch
        adopted = list(getattr(reply, "leases", None) or [])
        with model._count_lock:
            still_held = [lease for lease in held if not lease.released]
        for lease, lease_id in zip(still_held, adopted, strict=False):
            lease.arbiter_lease = lease_id
            lease.epoch = self._epoch
        for lease_id in adopted[len(still_held) :]:
            self._notify(p.Release(lease=lease_id))  # released while we were re-registering

    async def _read_loop(self, reader: asyncio.StreamReader, epoch: int) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    message = p.decode_reply(line)
                except p.UnknownMessageType as exc:
                    log.debug("ignoring unknown message type %s", exc.type_name)
                    continue
                except VrambiterError as exc:
                    log.warning("bad message from arbiter: %s", exc)
                    continue
                self._dispatch(message)
        except (OSError, ValueError, asyncio.IncompleteReadError):
            pass
        finally:
            self._link_lost(epoch)

    def _dispatch(self, message: p.Message) -> None:
        if isinstance(message, p.Evict):
            model = self.models.get(message.model)
            if model is None:
                self._notify(p.Evicted(model=message.model, evict_id=message.evict_id))
                return
            threading.Thread(
                target=model._handle_evict,
                args=(message, self._epoch),
                name=f"vrambiter-evict-{model.name}",
                daemon=True,
            ).start()
            return
        if isinstance(message, p.Notice):
            log.info("arbiter notice %s: %s", message.kind, message.message)
            return
        if message.id is None:
            return
        with self._state_lock:
            future = self._pending.pop(message.id, None)
            abandoned = message.id in self._abandoned
            self._abandoned.discard(message.id)
        if abandoned and isinstance(message, p.Granted):
            self._notify(p.Release(lease=message.lease))  # its acquirer gave up
            return
        if future is not None and not future.done():
            future.set_result(message)

    def _link_lost(self, epoch: int) -> None:
        with self._state_lock:
            if epoch != self._epoch or not self._link_up:
                return
            self._link_up = False
            self._ready = False
            self._writer = None
            pending, self._pending = self._pending, {}
            self._abandoned.clear()
        for future in pending.values():
            if not future.done():
                future.set_exception(_ConnectionLost())
        if self._closed:
            return
        log.warning("lost the arbiter; running standalone until it is back")
        if self._reconnect_enabled:
            self._reconnect_task = asyncio.ensure_future(self._reconnect())

    async def _reconnect(self) -> None:
        delay = 0.2
        while not self._closed:
            await asyncio.sleep(delay)
            try:
                await self._open()
                log.info("reconnected to the arbiter; models re-registered")
                return
            except VrambiterError as exc:
                log.debug("reconnect failed: %s", exc)
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("reconnect failed: %r", exc)
            delay = min(delay * 2, 5.0)

    # -- sending ----------------------------------------------------------------------------------

    def _submit(self, message: p.Message, *, internal: bool = False) -> cf.Future[p.Message]:
        """Assign an id, register for the reply, and queue the write - all in submission order."""
        future: cf.Future[p.Message] = cf.Future()
        with self._state_lock:
            usable = self._link_up if internal else self._ready
            if not usable or self._closed:
                future.set_exception(_ConnectionLost())
                return future
            message.id = next(self._ids)
            self._pending[message.id] = future
            writer, epoch = self._writer, self._epoch
        self._loop.call_soon_threadsafe(self._write, writer, epoch, message)
        return future

    def _write(self, writer: asyncio.StreamWriter | None, epoch: int, message: p.Message) -> None:
        if writer is None or epoch != self._epoch or writer.is_closing():
            return
        with contextlib.suppress(OSError, RuntimeError):
            writer.write(p.encode(message))

    async def _request_async(self, message: p.Message) -> p.Message:
        """Send from the loop thread during (re)connection and await the reply."""
        try:
            reply = await asyncio.wrap_future(self._submit(message, internal=True))
        except _ConnectionLost:
            raise ArbiterUnavailable("connection to the arbiter lost") from None
        if isinstance(reply, p.Error):
            raise error_from_wire(reply.code, reply.message, reply.detail)
        return reply

    def _request(self, message: p.Message, timeout: float | None = 30.0) -> p.Message:
        """Send from a user thread and wait for the reply. Raises on ``error`` replies."""
        future = self._submit(message)
        try:
            reply = future.result(timeout)
        except _ConnectionLost:
            raise ArbiterUnavailable("not connected to the arbiter") from None
        if isinstance(reply, p.Error):
            raise error_from_wire(reply.code, reply.message, reply.detail)
        return reply

    def _notify(self, message: p.Message, epoch: int | None = None) -> None:
        """Fire-and-forget. With ``epoch``, only if still on that connection (lease ids and evict
        ids do not survive a reconnect)."""
        if epoch is not None and epoch != self._epoch:
            return
        future = self._submit(message, internal=True)
        future.add_done_callback(_log_error_reply)

    # -- hooks ------------------------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._ready

    def _announce(self, model: Model) -> None:
        if not self._ready:
            return  # registered when the link comes back
        try:
            message, _ = model._register_message()
            epoch = self._epoch
            self._request(message)
            model._registered_epoch = epoch
        except ArbiterUnavailable:
            pass  # lost the link meanwhile; the reconnect re-registers it

    def _acquire(
        self, model: str, *, wait: bool, timeout: float | None, token: _CancelToken | None
    ) -> _Grant:
        own = self.models.get(model)
        if own is not None and own._registered_epoch != self._epoch:
            return _STANDALONE  # this arbiter does not know the model (rejected or not yet sent)
        message = p.Acquire(model=model, wait=wait, timeout_s=timeout)
        future = self._submit(message)
        if future.done() and isinstance(future.exception(), _ConnectionLost):
            return _STANDALONE
        request_id = message.id  # assigned by _submit
        epoch = self._epoch
        if token is not None:
            token.attach(future)
        try:
            reply = future.result(None if timeout is None else timeout + 30.0)
        except _ConnectionLost:
            return _STANDALONE  # the arbiter went away: proceed as a plain server would
        except BaseException:
            # Interrupted (Ctrl-C, task cancelled, local timeout): withdraw the request, and make
            # sure a grant that is already on its way gets released.
            future.cancel()
            self._abandon(request_id)
            raise
        if isinstance(reply, p.Error):
            raise error_from_wire(reply.code, reply.message, reply.detail)
        assert isinstance(reply, p.Granted)
        return _Grant(reply.lease, reply.action, epoch)

    def _abandon(self, request_id: int | None) -> None:
        if request_id is None:
            return
        with self._state_lock:
            if request_id in self._pending:
                self._pending.pop(request_id, None)
                self._abandoned.add(request_id)
        self._notify(p.Cancel(request=request_id))

    # -- control operations -----------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """The arbiter's view of every device, model, satellite and queued request."""
        reply = self._request(p.StatusRequest())
        out = reply.to_wire()
        out.pop("type", None)
        out.pop("id", None)
        return out

    def pin(self, model: str) -> None:
        self._request(p.Pin(model=model))

    def unpin(self, model: str) -> None:
        self._request(p.Unpin(model=model))

    def evict(self, model: str) -> None:
        """Ask the arbiter to evict ``model`` now if it is idle."""
        self._request(p.EvictModel(model=model))

    def profile(self, name: str) -> None:
        """Activate a configured profile ('' clears it)."""
        self._request(p.Profile(name=name))

    def close(self) -> None:
        """Disconnect. Models stay as they are; later leases run standalone."""
        if self._closed:
            return
        self._closed = True

        async def _close() -> None:
            if self._reconnect_task is not None:
                self._reconnect_task.cancel()
            with self._state_lock:
                writer = self._writer
                self._ready = self._link_up = False
                pending, self._pending = self._pending, {}
            # Callers blocked on a reply carry on standalone rather than hang forever.
            for future in pending.values():
                if not future.done():
                    future.set_exception(_ConnectionLost())
            if writer is not None:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()

        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(_close(), self._loop).result(5)
        self._shutdown_loop()

    def __repr__(self) -> str:
        state = "connected" if self._ready else "standalone (reconnecting)"
        return f"<ArbiterClient {self.name} {state} {self.socket_path}>"


# --------------------------------------------------------------------------- helpers


def _log_error_reply(future: cf.Future[p.Message]) -> None:
    if future.cancelled() or future.exception() is not None:
        return
    reply = future.result()
    if isinstance(reply, p.Error):
        log.warning("arbiter rejected a notification: %s (%s)", reply.message, reply.code)


def _version() -> str:
    from . import __version__

    return __version__


def _torch_reserved(device: int) -> int | None:
    """``torch.cuda.memory_reserved(device)`` if torch is already imported; never imports it."""
    if "torch" not in sys.modules:
        return None
    from .integrations.torch import reserved_bytes

    return reserved_bytes(device)


def _release_cached_memory() -> None:
    """``gc.collect()`` and, if torch is imported, ``torch.cuda.empty_cache()``."""
    from .integrations.torch import release_cached_memory

    release_cached_memory()
