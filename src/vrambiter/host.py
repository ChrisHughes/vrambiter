"""Host-side measurements: available RAM, and which processes belong to which satellite.

WHY HOST RAM IS ARBITRATED AT ALL. Loading a model can need far more host memory than the model
finally occupies on the GPU: safetensors are staged through host RAM, some loaders materialise fp32
copies before casting. On the workstation this project was built for (97 GB of VRAM, 27 GB of RAM)
two loads at once is how the machine gets OOM-killed, long before the GPU is full. So the arbiter
checks ``MemAvailable`` before admitting a load and serialises loads.

``MemAvailable`` (not ``MemFree``) is the kernel's estimate of what can be allocated without
swapping, counting reclaimable page cache. It is the number ``free -h`` shows as "available".
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from .units import parse_bytes

__all__ = [
    "HostBackend",
    "ProcMeminfo",
    "PsutilHost",
    "FakeHost",
    "UnknownHost",
    "default_host_backend",
    "process_tree",
]


@runtime_checkable
class HostBackend(Protocol):
    def mem_available(self) -> int | None:
        """Bytes the kernel says can be allocated now, or ``None`` if unknown."""
        ...

    def mem_total(self) -> int | None: ...


class ProcMeminfo:
    """Linux: parse ``/proc/meminfo``. No dependencies, and the authoritative source."""

    def __init__(self, path: str | os.PathLike[str] = "/proc/meminfo") -> None:
        self._path = Path(path)

    def _read(self) -> dict[str, int]:
        values: dict[str, int] = {}
        for line in self._path.read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if not parts:
                continue
            try:
                number = int(parts[0])
            except ValueError:
                continue
            # The kernel writes "kB" and means KiB.
            values[key.strip()] = number * 1024 if len(parts) > 1 and parts[1] == "kB" else number
        return values

    def mem_available(self) -> int | None:
        values = self._read()
        if "MemAvailable" in values:
            return values["MemAvailable"]
        # Kernels before 3.14 lack MemAvailable; this approximation is what `free` used then.
        if "MemFree" in values:
            return values["MemFree"] + values.get("Buffers", 0) + values.get("Cached", 0)
        return None

    def mem_total(self) -> int | None:
        return self._read().get("MemTotal")


class PsutilHost:
    """Any platform with ``psutil`` installed (the ``daemon`` extra)."""

    def __init__(self) -> None:
        import psutil

        self._psutil = psutil

    def mem_available(self) -> int | None:
        return int(self._psutil.virtual_memory().available)

    def mem_total(self) -> int | None:
        return int(self._psutil.virtual_memory().total)


class UnknownHost:
    """No way to measure. Host-RAM admission is skipped; load serialisation still applies."""

    def mem_available(self) -> int | None:
        return None

    def mem_total(self) -> int | None:
        return None


class FakeHost:
    """Programmable host memory for tests."""

    def __init__(self, available: int | str = "64GiB", total: int | str | None = None) -> None:
        self._lock = threading.Lock()
        self._available = parse_bytes(available)
        self._total = parse_bytes(total) if total is not None else self._available

    def set_available(self, nbytes: int | str) -> None:
        with self._lock:
            self._available = parse_bytes(nbytes)

    def take(self, nbytes: int | str) -> None:
        """Model a load staging ``nbytes`` through host RAM."""
        with self._lock:
            self._available = max(0, self._available - parse_bytes(nbytes))

    def give(self, nbytes: int | str) -> None:
        with self._lock:
            self._available += parse_bytes(nbytes)

    def mem_available(self) -> int | None:
        with self._lock:
            return self._available

    def mem_total(self) -> int | None:
        return self._total


def default_host_backend() -> HostBackend:
    """``/proc/meminfo`` on Linux, else ``psutil`` if installed, else :class:`UnknownHost`."""
    if Path("/proc/meminfo").exists():
        return ProcMeminfo()
    try:
        return PsutilHost()
    except ImportError:
        return UnknownHost()


def process_tree(pid: int) -> set[int]:
    """``pid`` and all its descendants.

    Used to attribute GPU memory to a satellite whose work happens in child processes, such as
    ``llama-server`` in router mode, which runs each model in its own child instance. Uses
    ``psutil`` when available, else scans ``/proc`` on Linux, else returns just ``{pid}``.
    Blocking (it reads the process table): call it off the event loop.
    """
    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            return {pid} | {child.pid for child in proc.children(recursive=True)}
        except psutil.Error:
            return {pid}
    proc_dir = Path("/proc")
    if not proc_dir.is_dir():
        return {pid}
    children: dict[int, list[int]] = {}
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # The command name is parenthesised and may contain spaces; the ppid follows the ')'.
        fields = stat.rpartition(")")[2].split()
        if len(fields) >= 2:
            children.setdefault(int(fields[1]), []).append(int(entry.name))
    tree, stack = {pid}, [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in tree:
                tree.add(child)
                stack.append(child)
    return tree
