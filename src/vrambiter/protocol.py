"""The wire protocol: newline-delimited JSON over a Unix domain socket.

One JSON object per line, UTF-8, ``\\n``-terminated. Every object has a ``type``. Requests from a
satellite carry an integer ``id`` and the arbiter's reply echoes it; the arbiter may also send
unsolicited messages (``evict``, ``notice``) at any time, which carry their own identifiers.

Compatibility rules, chosen so that an old satellite keeps working against a newer arbiter and the
other way round:

* unknown *fields* are ignored on decode;
* an unknown message *type* is answered with an ``error`` reply, never a disconnect;
* ``protocol`` in ``hello``/``welcome`` is an integer that only changes on an incompatible change.

Message classes are plain dataclasses so both sides construct them with keyword arguments and get
validation for free; the arbiter and client never pass raw dicts around.

Stdlib only: this module is imported by the client library.
"""

from __future__ import annotations

import builtins
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from .errors import ProtocolError

__all__ = [
    "PROTOCOL_VERSION",
    "MAX_LINE_BYTES",
    "Message",
    "UnknownMessageType",
    "encode",
    "decode",
    "decode_request",
    "decode_reply",
    "split_model_id",
    # client -> arbiter
    "Hello",
    "Register",
    "Acquire",
    "Cancel",
    "Loaded",
    "LoadFailed",
    "Release",
    "Unloaded",
    "Evicted",
    "EvictRefused",
    "Pin",
    "Unpin",
    "EvictModel",
    "Profile",
    "StatusRequest",
    # arbiter -> client
    "Welcome",
    "Ok",
    "Error",
    "Granted",
    "StatusReply",
    "Evict",
    "Notice",
]

#: Bumped only on an incompatible change. Additive changes (new fields, new message types) do not.
PROTOCOL_VERSION = 1

#: A line longer than this is a protocol error. Status replies for hundreds of models fit easily.
MAX_LINE_BYTES = 1 << 20

M = TypeVar("M", bound="Message")


@dataclass(kw_only=True)
class Message:
    """Base class. ``type`` is the wire name; ``id`` correlates a request with its reply."""

    type: ClassVar[str] = ""
    id: int | None = None

    def to_wire(self) -> dict[str, Any]:
        """A JSON-ready dict. ``None`` fields are omitted to keep lines short."""
        out: dict[str, Any] = {"type": self.type}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if value is not None:
                out[f.name] = value
        return out

    @classmethod
    def from_wire(cls: builtins.type[M], data: dict[str, Any]) -> M:
        """Build from a decoded dict, ignoring unknown fields and type-checking known ones."""
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name in data and data[f.name] is not None:
                value = data[f.name]
                if not _type_ok(value, str(f.type)):
                    raise ProtocolError(
                        f"{cls.type}.{f.name}: expected {f.type}, got {type(value).__name__}",
                        detail={"field": f.name},
                    )
                kwargs[f.name] = float(value) if str(f.type).startswith("float") else value
            elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise ProtocolError(
                    f"{cls.type}: missing required field {f.name!r}", detail={"field": f.name}
                )
        return cls(**kwargs)


def _type_ok(value: Any, annotation: str) -> bool:
    """Minimal runtime check for the handful of annotation shapes messages use.

    Annotations are strings here (``from __future__ import annotations``). A full validator would be
    a dependency; this only has to stop a malformed peer from putting a string where the arbiter
    will do arithmetic.
    """
    for option in (part.strip() for part in annotation.split("|")):
        if option == "None" and value is None:
            return True
        if option == "Any":
            return True
        if option == "int" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if option == "float" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if option == "str" and isinstance(value, str):
            return True
        if option == "bool" and isinstance(value, bool):
            return True
        if option.startswith("dict") and isinstance(value, dict):
            return True
        if option.startswith("list") and isinstance(value, list):
            return True
    return False


# --------------------------------------------------------------------------- client -> arbiter


@dataclass(kw_only=True)
class Hello(Message):
    """First message on every connection. ``role`` is ``"satellite"`` or ``"control"`` (the CLI)."""

    type: ClassVar[str] = "hello"
    name: str
    pid: int
    version: str = ""
    protocol: int = PROTOCOL_VERSION
    role: str = "satellite"


@dataclass(kw_only=True)
class Register(Message):
    """Declare a model this satellite can load. Sizes are bytes.

    ``state``/``vram_bytes``/``leases`` describe the model as the satellite currently holds it, so a
    satellite re-registering after an arbiter restart is believed rather than reset: a model that
    is resident and leased stays resident and busy.
    """

    type: ClassVar[str] = "register"
    model: str
    vram_peak: int
    device: int = 0
    vram_resident: int | None = None
    host_peak: int = 0
    priority: int = 0
    pinned: bool = False
    state: str = "unloaded"
    vram_bytes: int | None = None
    leases: int = 0


@dataclass(kw_only=True)
class Acquire(Message):
    """Ask for a lease. ``model`` is a short name (own model) or a full ``satellite/model`` id."""

    type: ClassVar[str] = "acquire"
    model: str
    wait: bool = True
    timeout_s: float | None = None


@dataclass(kw_only=True)
class Cancel(Message):
    """Withdraw a queued ``acquire`` (by its request ``id``). An addition to the design draft."""

    type: ClassVar[str] = "cancel"
    request: int


@dataclass(kw_only=True)
class Loaded(Message):
    type: ClassVar[str] = "loaded"
    model: str
    vram_bytes: int | None = None


@dataclass(kw_only=True)
class LoadFailed(Message):
    type: ClassVar[str] = "load_failed"
    model: str
    error: str = ""


@dataclass(kw_only=True)
class Release(Message):
    type: ClassVar[str] = "release"
    lease: str


@dataclass(kw_only=True)
class Unloaded(Message):
    """The satellite unloaded a model on its own (not in response to ``evict``)."""

    type: ClassVar[str] = "unloaded"
    model: str


@dataclass(kw_only=True)
class Evicted(Message):
    type: ClassVar[str] = "evicted"
    model: str
    evict_id: str


@dataclass(kw_only=True)
class EvictRefused(Message):
    type: ClassVar[str] = "evict_refused"
    model: str
    evict_id: str
    reason: str = ""


@dataclass(kw_only=True)
class Pin(Message):
    type: ClassVar[str] = "pin"
    model: str


@dataclass(kw_only=True)
class Unpin(Message):
    type: ClassVar[str] = "unpin"
    model: str


@dataclass(kw_only=True)
class EvictModel(Message):
    """Operator request: evict this model now if it is idle (``vrambiter evict``)."""

    type: ClassVar[str] = "evict_model"
    model: str


@dataclass(kw_only=True)
class Profile(Message):
    """Activate a named profile from the config; an empty name clears the active profile."""

    type: ClassVar[str] = "profile"
    name: str = ""


@dataclass(kw_only=True)
class StatusRequest(Message):
    type: ClassVar[str] = "status"


# --------------------------------------------------------------------------- arbiter -> client


@dataclass(kw_only=True)
class Welcome(Message):
    type: ClassVar[str] = "welcome"
    satellite: str
    protocol: int = PROTOCOL_VERSION
    arbiter_version: str = ""


@dataclass(kw_only=True)
class Ok(Message):
    """Generic success. ``register`` fills ``model`` (the full id) and ``leases`` (adopted)."""

    type: ClassVar[str] = "ok"
    model: str | None = None
    leases: list[str] | None = None


@dataclass(kw_only=True)
class Error(Message):
    type: ClassVar[str] = "error"
    code: str
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Granted(Message):
    """A lease. ``action`` is ``"load"`` (you must load, then send ``loaded``) or ``"ready"``."""

    type: ClassVar[str] = "granted"
    lease: str
    action: str
    model: str = ""


@dataclass(kw_only=True)
class StatusReply(Message):
    type: ClassVar[str] = "status"
    arbiter: dict[str, Any] = field(default_factory=dict)
    devices: list[dict[str, Any]] = field(default_factory=list)
    host: dict[str, Any] = field(default_factory=dict)
    models: list[dict[str, Any]] = field(default_factory=list)
    satellites: list[dict[str, Any]] = field(default_factory=list)
    queue: list[dict[str, Any]] = field(default_factory=list)
    foreign: list[dict[str, Any]] = field(default_factory=list)


@dataclass(kw_only=True)
class Evict(Message):
    """Unsolicited: please unload ``model`` and reply ``evicted`` or ``evict_refused``."""

    type: ClassVar[str] = "evict"
    evict_id: str
    model: str
    reason: str = ""


@dataclass(kw_only=True)
class Notice(Message):
    """Unsolicited, informational (e.g. ``kind="lease_lost"``). Safe to ignore."""

    type: ClassVar[str] = "notice"
    kind: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


REQUESTS: dict[str, type[Message]] = {
    cls.type: cls
    for cls in (
        Hello,
        Register,
        Acquire,
        Cancel,
        Loaded,
        LoadFailed,
        Release,
        Unloaded,
        Evicted,
        EvictRefused,
        Pin,
        Unpin,
        EvictModel,
        Profile,
        StatusRequest,
    )
}

REPLIES: dict[str, type[Message]] = {
    cls.type: cls for cls in (Welcome, Ok, Error, Granted, StatusReply, Evict, Notice)
}


class UnknownMessageType(ProtocolError):
    """A well-formed line whose ``type`` this side does not know. Answered, never fatal."""

    code = "unknown_type"

    def __init__(self, type_name: str, request_id: Any = None) -> None:
        super().__init__(f"unknown message type {type_name!r}", detail={"type": type_name})
        self.type_name = type_name
        self.request_id = request_id


# --------------------------------------------------------------------------- framing


def encode(message: Message) -> bytes:
    """One message as one NDJSON line (compact JSON, UTF-8, trailing newline)."""
    line = json.dumps(message.to_wire(), separators=(",", ":"), ensure_ascii=False)
    return line.encode("utf-8") + b"\n"


def decode(line: bytes | str, registry: dict[str, type[Message]]) -> Message:
    """Parse one line with ``registry`` (``REQUESTS`` on the arbiter, ``REPLIES`` on the client).

    Raises :class:`UnknownMessageType` for a valid object with an unknown ``type`` (so the caller
    can reply with the request's ``id``) and :class:`ProtocolError` for everything else.
    """
    if isinstance(line, bytes):
        if len(line) > MAX_LINE_BYTES:
            raise ProtocolError(f"line exceeds {MAX_LINE_BYTES} bytes")
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError(f"line is not UTF-8: {exc}") from None
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc.msg}") from None
    if not isinstance(data, dict):
        raise ProtocolError("message must be a JSON object")
    type_name = data.get("type")
    if not isinstance(type_name, str):
        raise ProtocolError("message has no string 'type'")
    cls = registry.get(type_name)
    if cls is None:
        raise UnknownMessageType(type_name, data.get("id"))
    request_id = data.get("id")
    if request_id is not None and (not isinstance(request_id, int) or isinstance(request_id, bool)):
        raise ProtocolError("'id' must be an integer")
    return cls.from_wire(data)


def decode_request(line: bytes | str) -> Message:
    """Decode a client -> arbiter line."""
    return decode(line, REQUESTS)


def decode_reply(line: bytes | str) -> Message:
    """Decode an arbiter -> client line."""
    return decode(line, REPLIES)


def split_model_id(model_id: str) -> tuple[str, str]:
    """``"llama/gemma-4-26b"`` -> ``("llama", "gemma-4-26b")``.

    Only the first ``/`` separates: model names may themselves contain slashes (llama.cpp router
    ids look like ``ggml-org/gemma-3-4b-it-GGUF:Q4_K_M``). Satellite names may not.
    """
    satellite, sep, model = model_id.partition("/")
    if not sep or not satellite or not model:
        raise ValueError(f"not a full model id (expected 'satellite/model'): {model_id!r}")
    return satellite, model
