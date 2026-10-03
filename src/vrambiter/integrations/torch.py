"""PyTorch helpers: actually give memory back, and measure what a model holds.

WHY UNLOADING NEEDS HELP. Dropping the last reference to a model's tensors returns their blocks to
PyTorch's *caching allocator*, not to the driver. ``nvidia-smi``, NVML and
``torch.cuda.mem_get_info`` (the numbers the arbiter admits against) keep counting that memory as
used until ``torch.cuda.empty_cache()`` hands the cached blocks back. And reference cycles (a
pipeline that points at its modules that point back at it) keep tensors alive until the cyclic
garbage collector runs. So an unload is: drop references, ``gc.collect()``, ``empty_cache()``. The
client does the last two after every ``unload`` callback; :func:`unload` does all three.

NOTHING HERE IMPORTS TORCH AT MODULE IMPORT TIME. Importing torch costs seconds and, in a process
that has not chosen a CUDA device yet, can initialise a context. Every function checks
``sys.modules`` and does nothing when the satellite has not imported torch itself.
"""

from __future__ import annotations

import gc
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = [
    "torch_loaded",
    "release_cached_memory",
    "unload",
    "reserved_bytes",
    "allocated_bytes",
    "measure_reserved",
]

log = logging.getLogger("vrambiter.integrations.torch")


def _torch() -> Any:
    """The already-imported torch module, or ``None``. Never imports it."""
    return sys.modules.get("torch")


def torch_loaded() -> bool:
    """Whether the process has imported torch *and* initialised CUDA."""
    torch = _torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    except Exception:
        return False


def release_cached_memory() -> None:
    """``gc.collect()``, then ``torch.cuda.empty_cache()`` if torch has initialised CUDA.

    Safe to call anywhere, any number of times; costs a GC pass.
    """
    gc.collect()
    if not torch_loaded():
        return
    torch = _torch()
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Memory shared with other processes through CUDA IPC is only freed once collected.
        torch.cuda.ipc_collect()
    except Exception:
        log.warning("empty_cache failed", exc_info=True)


def unload(owner: Any, *attrs: str) -> None:
    """Set ``owner.<attr> = None`` for each attribute, then :func:`release_cached_memory`.

    ``owner`` may also be a dict, in which case the keys are deleted. Typical use in an unload
    callback::

        def free_weights():
            vrambiter.integrations.torch.unload(state, "pipe", "vae")
    """
    for attr in attrs:
        if isinstance(owner, dict):
            owner.pop(attr, None)
        else:
            setattr(owner, attr, None)
    release_cached_memory()


def reserved_bytes(device: int | None = None) -> int | None:
    """``torch.cuda.memory_reserved(device)``: this process's allocator pool, or ``None``.

    This is what the process holds from the driver's point of view (minus the CUDA context), so it
    is the right figure to report as a model's footprint. ``memory_allocated`` would undercount by
    whatever the pool keeps cached.
    """
    if not torch_loaded():
        return None
    try:
        return int(_torch().cuda.memory_reserved(device))
    except Exception:
        return None


def allocated_bytes(device: int | None = None) -> int | None:
    """``torch.cuda.memory_allocated(device)``: live tensors only, or ``None``."""
    if not torch_loaded():
        return None
    try:
        return int(_torch().cuda.memory_allocated(device))
    except Exception:
        return None


class _Measurement:
    def __init__(self) -> None:
        self.before: int | None = None
        self.after: int | None = None

    @property
    def delta(self) -> int | None:
        if self.before is None or self.after is None:
            return None
        return max(0, self.after - self.before)


@contextmanager
def measure_reserved(device: int | None = None) -> Iterator[_Measurement]:
    """``with measure_reserved(0) as m: load(); m.delta`` - pool growth across the block.

    Useful in a multi-model satellite, where ``memory_reserved`` is the whole process and the
    per-model figure is the growth during that model's load.
    """
    m = _Measurement()
    m.before = reserved_bytes(device)
    try:
        yield m
    finally:
        m.after = reserved_bytes(device)
