"""Byte quantities: ``"17GiB"``, ``"9GB"``, ``"512 MiB"`` and plain ints, to and from bytes.

Sizes appear in three places (code passed to ``register()``, the TOML config, and CLI output), and
every one of them is written by a human who is thinking in gigabytes. The wire protocol carries
plain integers, so parsing happens once, at the edge, and the arbiter only ever sees bytes.

Both families are accepted and they are *not* interchangeable: ``GB`` is 10**9 bytes and ``GiB`` is
2**30. NVML and ``torch.cuda.mem_get_info`` report bytes, and the "96 GB" on a datasheet is
usually GiB-ish marketing, so the parser refuses to guess. A bare number is bytes.

Stdlib only: this module is imported by the client library.
"""

from __future__ import annotations

import math
import re

__all__ = ["KiB", "MiB", "GiB", "TiB", "parse_bytes", "parse_optional_bytes", "format_bytes"]

KiB = 1024
MiB = 1024**2
GiB = 1024**3
TiB = 1024**4

# Keys are lower-cased suffixes. Single letters follow the Kubernetes convention ("G" is SI, "Gi" is
# IEC) because that is the only widely used convention that is not ambiguous.
_SUFFIXES: dict[str, int] = {
    "": 1,
    "b": 1,
    "k": 10**3,
    "kb": 10**3,
    "m": 10**6,
    "mb": 10**6,
    "g": 10**9,
    "gb": 10**9,
    "t": 10**12,
    "tb": 10**12,
    "ki": KiB,
    "kib": KiB,
    "mi": MiB,
    "mib": MiB,
    "gi": GiB,
    "gib": GiB,
    "ti": TiB,
    "tib": TiB,
}

_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*([A-Za-z]*)\s*$")


def parse_bytes(value: int | float | str) -> int:
    """Return ``value`` as a non-negative number of bytes.

    Accepts ints (bytes), finite non-negative floats (bytes, rounded up) and strings such as
    ``"17GiB"``, ``"9 GB"``, ``"1.5gib"`` or ``"4096"``. Fractions round *up*: a size is a claim on
    memory, and under-claiming by a byte is the wrong direction to err in.

    Raises ``ValueError`` for anything else, including negative sizes and ``bool`` (which is an
    ``int`` subclass and almost certainly a bug at the call site).
    """
    if isinstance(value, bool):
        raise ValueError(f"not a byte size: {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"byte size must be >= 0, got {value}")
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"byte size must be a finite number >= 0, got {value}")
        return math.ceil(value)
    if isinstance(value, str):
        match = _PATTERN.match(value)
        if not match:
            raise ValueError(f"not a byte size: {value!r} (expected e.g. '9GiB', '9GB' or 4096)")
        number, suffix = match.groups()
        try:
            multiplier = _SUFFIXES[suffix.lower()]
        except KeyError:
            raise ValueError(
                f"unknown size unit {suffix!r} in {value!r}; use B, KB/KiB, MB/MiB, GB/GiB, TB/TiB"
            ) from None
        if "." in number:
            # Decimal arithmetic through float is exact enough for sizes up to petabytes; the
            # ceil keeps the rounding direction conservative.
            return math.ceil(float(number) * multiplier)
        return int(number) * multiplier
    raise ValueError(f"not a byte size: {value!r}")


def parse_optional_bytes(value: int | float | str | None) -> int | None:
    """``parse_bytes``, passing ``None`` through, for optional sizes like ``vram_resident``."""
    return None if value is None else parse_bytes(value)


def format_bytes(n: int | float | None, *, precision: int = 1) -> str:
    """Render bytes for people: ``format_bytes(9 * GiB) == "9.0 GiB"``.

    Always IEC units, because that is what NVML-derived numbers naturally are. Negative values are
    allowed (deltas, residues) and ``None`` renders as ``"?"`` so status output never crashes on an
    unknown measurement.
    """
    if n is None:
        return "?"
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit, size in (("TiB", TiB), ("GiB", GiB), ("MiB", MiB), ("KiB", KiB)):
        if n >= size:
            return f"{sign}{n / size:.{precision}f} {unit}"
    return f"{sign}{int(n)} B"
