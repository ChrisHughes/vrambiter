"""vrambiter: cooperative VRAM arbitration for model-serving processes sharing a GPU.

Satellites use :func:`connect`, :meth:`ArbiterClient.register` and leases; see the README for a
quick start and ``docs/DESIGN.md`` for how the arbiter decides. Importing this package imports only
the standard library.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vrambiter")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0+unknown"

from .client import ArbiterClient, Lease, Model, NullArbiter, connect, default_socket_path
from .errors import (
    ArbiterError,
    ArbiterUnavailable,
    HostRamUnavailable,
    NameInUse,
    ProtocolError,
    RegistrationError,
    UnknownModel,
    VrambiterError,
    VramUnavailable,
)
from .units import format_bytes, parse_bytes

__all__ = [
    "__version__",
    "connect",
    "default_socket_path",
    "ArbiterClient",
    "NullArbiter",
    "Model",
    "Lease",
    "VrambiterError",
    "VramUnavailable",
    "HostRamUnavailable",
    "ArbiterUnavailable",
    "ArbiterError",
    "UnknownModel",
    "RegistrationError",
    "NameInUse",
    "ProtocolError",
    "parse_bytes",
    "format_bytes",
]
