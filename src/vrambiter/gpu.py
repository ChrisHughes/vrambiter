"""GPU memory measurement: the one number admission is allowed to trust, and the seam tests use.

WHY THE DRIVER'S NUMBER. Admission compares against what the *driver* says is free (NVML, the same
figure ``torch.cuda.mem_get_info`` returns), never against a sum of declarations or any process's
allocator statistics. The driver sees every process on the card, the CUDA contexts, and the
freed-but-cached blocks PyTorch keeps in its pool, and those are exactly the things that made an
in-process arbiter's accounting drift from reality. Per-process usage (also from NVML) is what lets
the arbiter attribute memory to satellites and call the rest "foreign".

Backends:

* :class:`NvmlBackend` uses ``nvidia-ml-py`` (the ``daemon`` extra), imported lazily so that this
  module can be imported on a machine without an NVIDIA driver, such as a laptop running the tests.
* :class:`FakeGpu` is in-memory and programmable. With ``path=`` its state lives in a JSON file
  under ``flock``, so separate processes (satellite scripts in tests) can "allocate" on the same
  fake card the daemon is measuring.

Device indices are NVML indices. CUDA enumerates devices fastest-first unless
``CUDA_DEVICE_ORDER=PCI_BUS_ID`` is set; set it in satellites on multi-GPU machines so the two
agree.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .units import GiB, format_bytes, parse_bytes

__all__ = [
    "GpuBackend",
    "GpuBackendError",
    "GpuSnapshot",
    "NvmlBackend",
    "FakeGpu",
    "FakeOutOfMemory",
    "take_snapshot",
    "default_gpu_backend",
]


class GpuBackendError(RuntimeError):
    """The backend could not be initialised or a query failed."""


@runtime_checkable
class GpuBackend(Protocol):
    """What the arbiter needs from a GPU: a list of devices, free/total, and per-process usage."""

    def devices(self) -> list[int]: ...

    def mem_info(self, device: int) -> tuple[int, int]:
        """``(free, total)`` in bytes, as the driver reports them."""
        ...

    def process_usage(self, device: int) -> dict[int, int] | None:
        """``{pid: bytes}`` for processes using ``device``, or ``None`` when the driver cannot say
        (some containers, missing permissions). ``{}`` means "known: nobody"."""
        ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class GpuSnapshot:
    """One consistent-enough reading of every device, taken off the event loop.

    ``usage[device]`` is ``None`` when per-process numbers are unavailable, so callers can tell
    "this process holds nothing" from "nobody knows what this process holds".
    """

    free: Mapping[int, int]
    total: Mapping[int, int]
    usage: Mapping[int, Mapping[int, int] | None] = field(default_factory=dict)

    @property
    def devices(self) -> list[int]:
        return sorted(self.total)

    def pid_usage(self, pids: set[int] | frozenset[int], device: int) -> int | None:
        """Total bytes ``pids`` hold on ``device``, or ``None`` if per-process usage is unknown."""
        per_pid = self.usage.get(device)
        if per_pid is None:
            return None
        return sum(per_pid.get(pid, 0) for pid in pids)


def take_snapshot(backend: GpuBackend) -> GpuSnapshot:
    """Read every device once. Blocking: call it from a worker thread, never the event loop.

    Free memory and per-process usage cannot be read atomically, and an allocation landing
    between the two reads would count as free *and* (by shrinking a loading process's
    reservation) as already used: over-admission by its size. So usage is read before and after
    the free reading and each process gets the smaller figure, which is conservative whichever
    way memory moved in between.
    """
    free: dict[int, int] = {}
    total: dict[int, int] = {}
    usage: dict[int, dict[int, int] | None] = {}
    for device in backend.devices():
        before = backend.process_usage(device)
        free[device], total[device] = backend.mem_info(device)
        after = backend.process_usage(device)
        if before is None or after is None:
            usage[device] = None
        else:
            usage[device] = {
                pid: min(before.get(pid, 0), after.get(pid, 0)) for pid in before.keys() | after
            }
    return GpuSnapshot(free=free, total=total, usage=usage)


class NvmlBackend:
    """Real measurements through NVML (``pip install vrambiter[daemon]``)."""

    def __init__(self) -> None:
        try:
            import pynvml  # provided by the nvidia-ml-py distribution
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise GpuBackendError(
                "NVML is not available: install the daemon extra (pip install 'vrambiter[daemon]')"
            ) from exc
        self._nvml = pynvml
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        except pynvml.NVMLError as exc:  # pragma: no cover - needs a broken driver
            raise GpuBackendError(f"NVML initialisation failed: {exc}") from exc
        self._closed = False

    def devices(self) -> list[int]:
        return list(range(len(self._handles)))

    def mem_info(self, device: int) -> tuple[int, int]:
        try:
            info = self._nvml.nvmlDeviceGetMemoryInfo(self._handles[device])
        except self._nvml.NVMLError as exc:  # pragma: no cover
            raise GpuBackendError(f"nvmlDeviceGetMemoryInfo({device}) failed: {exc}") from exc
        return int(info.free), int(info.total)

    def process_usage(self, device: int) -> dict[int, int] | None:
        handle = self._handles[device]
        usage: dict[int, int] = {}
        answered = False
        queries = (
            self._nvml.nvmlDeviceGetComputeRunningProcesses,
            self._nvml.nvmlDeviceGetGraphicsRunningProcesses,
        )
        for query in queries:
            try:
                processes = query(handle)
            except self._nvml.NVMLError:  # pragma: no cover - e.g. not supported on this GPU
                continue
            if not processes:
                answered = True
            for proc in processes:
                used = getattr(proc, "usedGpuMemory", None)
                if used is None:  # NVML_VALUE_NOT_AVAILABLE: no permission, or MIG/vGPU
                    continue
                answered = True
                # A process doing both compute and graphics appears in both lists with the same
                # total; take the max rather than double-counting it.
                usage[int(proc.pid)] = max(usage.get(int(proc.pid), 0), int(used))
        return usage if answered else None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            with contextlib.suppress(Exception):
                self._nvml.nvmlShutdown()


class FakeOutOfMemory(MemoryError):
    """Raised by :meth:`FakeGpu.allocate` when the fake card is full, like a CUDA OOM.

    Tests rely on it: if the arbiter ever admits more than fits, a satellite's ``load`` fails loudly
    instead of the test passing on arithmetic nobody checked.
    """


class FakeGpu:
    """A programmable in-memory GPU for tests and demos.

    ``devices`` maps device index to total bytes (an int or size string means one device ``0``).
    Memory is held per pid; ``unattributed`` memory belongs to no process (a CUDA context, a driver
    that has not yet reclaimed a dead process's pages).

    ``path`` makes the state shared between processes through a JSON file guarded by ``flock``.
    ``reap_dead=True`` drops allocations of pids that no longer exist, which is what the real
    driver does when a process exits. ``report_process_usage=False`` simulates a driver that cannot
    report per-process usage.
    """

    def __init__(
        self,
        devices: Mapping[int, int | str] | int | str = 24 * GiB,
        *,
        path: str | os.PathLike[str] | None = None,
        reap_dead: bool = False,
        report_process_usage: bool = True,
    ) -> None:
        if isinstance(devices, (int, str)):
            devices = {0: devices}
        self._lock = threading.RLock()
        self._path = Path(path) if path is not None else None
        self.reap_dead = reap_dead
        self.report_process_usage = report_process_usage
        totals = {int(k): parse_bytes(v) for k, v in devices.items()}
        self._state: dict[str, Any] = {
            "total": {str(k): v for k, v in totals.items()},
            "usage": {str(k): {} for k in totals},
            "unattributed": {str(k): 0 for k in totals},
        }
        if self._path is not None:
            self._path.write_text(json.dumps(self._state))

    @classmethod
    def attach(
        cls, path: str | os.PathLike[str], *, reap_dead: bool = False, **kwargs: Any
    ) -> FakeGpu:
        """Open a shared fake card that another process created with ``FakeGpu(..., path=)``.

        Used by satellite scripts in tests: the daemon owns the card, satellites allocate on it.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"no shared fake GPU at {path}")
        gpu = cls.__new__(cls)
        gpu._lock = threading.RLock()
        gpu._path = Path(path)
        gpu.reap_dead = reap_dead
        gpu.report_process_usage = kwargs.get("report_process_usage", True)
        gpu._state = {}
        return gpu

    # -- state access -------------------------------------------------------------------------

    @contextlib.contextmanager
    def _locked(self, write: bool) -> Iterator[dict[str, Any]]:
        with self._lock:
            if self._path is None:
                yield self._state
                return
            with open(self._path, "r+") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
                try:
                    state = json.loads(fh.read() or "{}")
                    yield state
                    if write:
                        fh.seek(0)
                        fh.truncate()
                        fh.write(json.dumps(state))
                        fh.flush()
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)

    def _reap(self, state: dict[str, Any]) -> None:
        if not self.reap_dead:
            return
        for per_pid in state["usage"].values():
            for pid in [p for p in per_pid if not _pid_alive(int(p))]:
                del per_pid[pid]

    @staticmethod
    def _used(state: dict[str, Any], dev: str) -> int:
        return sum(state["usage"][dev].values()) + state["unattributed"].get(dev, 0)

    # -- GpuBackend -----------------------------------------------------------------------------

    def devices(self) -> list[int]:
        with self._locked(write=False) as state:
            return sorted(int(k) for k in state["total"])

    def mem_info(self, device: int) -> tuple[int, int]:
        with self._locked(write=self.reap_dead) as state:
            self._reap(state)
            dev = str(device)
            total = state["total"][dev]
            return max(0, total - self._used(state, dev)), total

    def process_usage(self, device: int) -> dict[int, int] | None:
        if not self.report_process_usage:
            return None
        with self._locked(write=self.reap_dead) as state:
            self._reap(state)
            return {int(pid): n for pid, n in state["usage"][str(device)].items() if n > 0}

    def close(self) -> None:
        pass

    # -- programming the fake ---------------------------------------------------------------------

    def allocate(self, pid: int, nbytes: int | str, device: int = 0) -> None:
        """Give ``pid`` another ``nbytes`` on ``device``; :class:`FakeOutOfMemory` if full."""
        n = parse_bytes(nbytes)
        with self._locked(write=True) as state:
            self._reap(state)
            dev = str(device)
            free = state["total"][dev] - self._used(state, dev)
            if n > free:
                raise FakeOutOfMemory(
                    f"fake GPU {device}: pid {pid} asked for {format_bytes(n)}, "
                    f"{format_bytes(free)} free"
                )
            per_pid = state["usage"][dev]
            per_pid[str(pid)] = per_pid.get(str(pid), 0) + n

    def free(self, pid: int, nbytes: int | str | None = None, device: int = 0) -> None:
        """Return ``nbytes`` (default: everything) that ``pid`` holds on ``device``."""
        with self._locked(write=True) as state:
            per_pid = state["usage"][str(device)]
            held = per_pid.get(str(pid), 0)
            remaining = 0 if nbytes is None else max(0, held - parse_bytes(nbytes))
            if remaining:
                per_pid[str(pid)] = remaining
            else:
                per_pid.pop(str(pid), None)

    def set_usage(self, pid: int, nbytes: int | str, device: int = 0) -> None:
        """Set what ``pid`` holds outright (no capacity check: use it to model a misbehaving
        process or a driver that is slow to reclaim)."""
        n = parse_bytes(nbytes)
        with self._locked(write=True) as state:
            per_pid = state["usage"][str(device)]
            if n:
                per_pid[str(pid)] = n
            else:
                per_pid.pop(str(pid), None)

    def kill(self, pid: int) -> None:
        """Drop everything ``pid`` holds on every device, as process exit would."""
        with self._locked(write=True) as state:
            for per_pid in state["usage"].values():
                per_pid.pop(str(pid), None)

    def set_unattributed(self, nbytes: int | str, device: int = 0) -> None:
        """Memory used on ``device`` that no process owns."""
        with self._locked(write=True) as state:
            state["unattributed"][str(device)] = parse_bytes(nbytes)

    def usage(self, pid: int, device: int = 0) -> int:
        with self._locked(write=False) as state:
            return int(state["usage"][str(device)].get(str(pid), 0))

    def free_bytes(self, device: int = 0) -> int:
        return self.mem_info(device)[0]


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` is a live process. A zombie is not: the driver frees a process's memory
    when it exits, not when its parent gets round to reaping it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:  # pragma: no cover
        return exc.errno != errno.ESRCH
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is in the daemon extra
        return True
    try:
        return bool(psutil.Process(pid).status() != psutil.STATUS_ZOMBIE)
    except psutil.Error:
        return False


def default_gpu_backend() -> GpuBackend:
    """``NvmlBackend``, or :class:`GpuBackendError` with an actionable message."""
    return NvmlBackend()
