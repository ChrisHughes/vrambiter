"""Exceptions shared by the daemon and the client, and their wire representation.

Every error the arbiter can report to a satellite has a stable string ``code``. The arbiter sends
``{"type": "error", "code": ..., "message": ..., "detail": {...}}`` and the client turns it back
into the matching exception class with :func:`error_from_wire`, so a satellite can ``except
VramUnavailable`` without caring that the decision was made in another process.

Stdlib only: this module is imported by the client library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .units import format_bytes

__all__ = [
    "VrambiterError",
    "ProtocolError",
    "ArbiterUnavailable",
    "ArbiterError",
    "UnknownModel",
    "RegistrationError",
    "NameInUse",
    "Holder",
    "VramUnavailable",
    "HostRamUnavailable",
    "error_from_wire",
]


class VrambiterError(Exception):
    """Base class for every exception this library raises on purpose."""

    code = "error"

    def __init__(self, message: str = "", *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = dict(detail or {})

    def to_wire(self) -> dict[str, Any]:
        """The ``code``/``message``/``detail`` triple carried by an ``error`` reply."""
        return {"code": self.code, "message": self.message, "detail": self.detail}


class ProtocolError(VrambiterError):
    """A peer sent something that is not valid vrambiter protocol (bad JSON, missing field, ...)."""

    code = "protocol_error"


class ArbiterUnavailable(VrambiterError):
    """No arbiter is reachable: raised by ``connect(required=True)`` and by control commands."""

    code = "arbiter_unavailable"


class ArbiterError(VrambiterError):
    """The arbiter rejected a request for a reason without a more specific class."""

    code = "arbiter_error"


class UnknownModel(VrambiterError):
    """The model id names nothing the arbiter knows about."""

    code = "unknown_model"


class RegistrationError(VrambiterError):
    """A ``register`` was rejected: inconsistent sizes, unknown device, wrong owner, ..."""

    code = "registration_error"


class NameInUse(VrambiterError):
    """Another live connection already uses this satellite name."""

    code = "name_in_use"


@dataclass(frozen=True)
class Holder:
    """One thing holding GPU memory, as named in a :class:`VramUnavailable`.

    ``kind`` is ``"model"`` for a registered model, ``"foreign"`` for a process the arbiter does
    not know, ``"reservation"`` for a load in flight and ``"unattributed"`` for memory nobody claims
    (the CUDA contexts, a caching allocator that was never emptied, a driver that is still
    returning a dead process's pages).
    """

    kind: str
    name: str
    bytes: int
    device: int = 0
    state: str = ""
    pinned: bool = False
    priority: int = 0
    pid: int | None = None

    def describe(self) -> str:
        bits = [self.state] if self.state else []
        if self.pinned:
            bits.append("pinned")
        if self.kind == "model" and self.priority:
            bits.append(f"priority {self.priority}")
        if self.kind != "model":
            bits.append(self.kind)
        extra = f" ({', '.join(bits)})" if bits else ""
        return f"{self.name}{extra} {format_bytes(self.bytes)}"


@dataclass
class _VramDetail:
    model: str
    device: int
    need: int
    free: int
    reason: str
    holders: list[Holder] = field(default_factory=list)


class VramUnavailable(VrambiterError):
    """The request cannot be satisfied, and the arbiter says why and who holds the memory.

    Raised when evicting every eligible idle model would still not make room *and* nothing busy or
    loading could free memory later, when a request's timeout expires while it is queued, and when
    a ``wait=False`` request would have to wait. It exists because the in-process arbiter this
    project grew out of had no terminal state: with an empty registry it slept and re-checked
    forever. A structured error that names the holders (including foreign processes) turns "the
    server hangs" into "pid 4321 holds 20 GiB".
    """

    code = "vram_unavailable"

    def __init__(
        self,
        message: str = "",
        *,
        model: str = "",
        device: int = 0,
        need: int = 0,
        free: int = 0,
        reason: str = "",
        holders: list[Holder] | tuple[Holder, ...] = (),
    ) -> None:
        self.model = model
        self.device = device
        self.need = need
        self.free = free
        self.reason = reason
        self.holders: list[Holder] = list(holders)
        if not message:
            message = self._default_message()
        detail = asdict(_VramDetail(model, device, need, free, reason, self.holders))
        super().__init__(message, detail=detail)

    def _default_message(self) -> str:
        head = (
            f"{self.model or 'request'} needs {format_bytes(self.need)} on GPU {self.device}; "
            f"{format_bytes(max(self.free, 0))} is available"
        )
        if self.reason:
            head += f": {self.reason}"
        if self.holders:
            named = ", ".join(h.describe() for h in self.holders[:8])
            more = f" and {len(self.holders) - 8} more" if len(self.holders) > 8 else ""
            head += f". Held by: {named}{more}"
        return head

    @classmethod
    def from_detail(cls, message: str, detail: dict[str, Any]) -> VramUnavailable:
        """Rebuild from an ``error`` reply's ``detail`` (works for subclasses too)."""
        holders = []
        for raw in detail.get("holders") or []:
            try:
                holders.append(
                    Holder(**{k: raw[k] for k in Holder.__dataclass_fields__ if k in raw})
                )
            except TypeError:
                continue
        return cls(
            message,
            model=str(detail.get("model", "")),
            device=int(detail.get("device", 0)),
            need=int(detail.get("need", 0)),
            free=int(detail.get("free", 0)),
            reason=str(detail.get("reason", "")),
            holders=holders,
        )


class HostRamUnavailable(VramUnavailable):
    """Like :class:`VramUnavailable`, but the missing resource is host RAM for a load.

    A subclass on purpose: a satellite that handles "the arbiter cannot make room" catches both.
    ``need`` and ``free`` are host bytes here, and ``holders`` is empty (the arbiter does not
    attribute host memory).
    """

    code = "host_ram_unavailable"

    def _default_message(self) -> str:
        head = (
            f"{self.model or 'request'} needs {format_bytes(self.need)} of host RAM to load; "
            f"{format_bytes(max(self.free, 0))} is available"
        )
        return f"{head}: {self.reason}" if self.reason else head


_BY_CODE: dict[str, type[VrambiterError]] = {
    cls.code: cls
    for cls in (
        VrambiterError,
        ProtocolError,
        ArbiterUnavailable,
        ArbiterError,
        UnknownModel,
        RegistrationError,
        NameInUse,
    )
}


def error_from_wire(
    code: str, message: str, detail: dict[str, Any] | None = None
) -> VrambiterError:
    """Rebuild the exception an ``error`` reply describes. Unknown codes become ``ArbiterError``."""
    detail = detail or {}
    if code == VramUnavailable.code:
        return VramUnavailable.from_detail(message, detail)
    if code == HostRamUnavailable.code:
        return HostRamUnavailable.from_detail(message, detail)
    cls = _BY_CODE.get(code, ArbiterError)
    err = cls(message, detail=detail)
    if cls is ArbiterError and code != ArbiterError.code:
        err.detail.setdefault("code", code)
    return err
