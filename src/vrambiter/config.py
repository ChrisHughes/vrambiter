"""``vrambiter.toml``: arbiter settings, profiles and configured satellites.

Parsing is strict on purpose. An unknown key is an error, not a silently ignored typo: a config
that says ``headrom = "8GiB"`` and runs with the default headroom is how a card gets run to its
last byte. Every error names its location (``satellite[1].model[0].vram_peak``).

Satellite kinds are inferred from what the entry declares:

* ``adapter = "llama-router"``: the arbiter loads and unloads models through the adapter. With a
  ``command`` the arbiter also runs the server process (recommended: that is what lets it attribute
  GPU memory to the server's child processes).
* ``command`` plus exactly one ``[[satellite.model]]`` and no adapter: a **black box**. Loading the
  model means starting the process, evicting it means stopping it.
* ``command`` and no models: a **cooperative** satellite that registers its own models over the
  socket. The arbiter starts it with ``VRAMBITER_SOCKET`` and ``VRAMBITER_NAME`` set and restarts
  it according to ``restart``.

**Fragments.** Beside the machine's own ``vrambiter.toml``, each application can ship its part of
the configuration as a drop-in file, ``vrambiter/conf.d/NAME.toml``, under any XDG config directory
(``/etc/xdg`` for a package, ``~/.config`` for the user). A fragment may add ``[[satellite]]``
entries and ``[profiles.*]`` pins; ``[arbiter]`` settings belong to the machine and are refused in a
fragment. A fragment in a more important directory replaces one of the same file name in a less
important one, so a user can override or mask what a package installed. Satellite names must be
unique across every file; profiles of the same name merge their pins.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .arbiter import NAME_PATTERN, ArbiterSettings, ModelSpec
from .units import parse_bytes

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib

__all__ = [
    "ConfigError",
    "Config",
    "SatelliteConfig",
    "ModelConfig",
    "SatelliteKind",
    "RestartPolicy",
    "load_config",
    "parse_config",
    "fragment_dirs",
    "find_fragments",
    "apply_fragments",
]

#: Drop-in directory name, under ``<XDG config dir>/vrambiter/``.
FRAGMENT_DIR = "conf.d"

ADAPTERS = ("llama-router",)


class ConfigError(ValueError):
    """The configuration is invalid; the message says where and why."""


class SatelliteKind(str, Enum):
    ADAPTER = "adapter"
    BLACK_BOX = "black-box"
    COOPERATIVE = "cooperative"


class RestartPolicy(str, Enum):
    NEVER = "never"
    ON_FAILURE = "on-failure"
    ALWAYS = "always"


@dataclass
class ModelConfig:
    name: str
    vram_peak: int
    vram_resident: int | None = None
    host_peak: int = 0
    priority: int = 0
    pinned: bool = False
    device: int = 0
    #: llama-router only: the id llama-server knows the model by (default: ``name``).
    router_model: str | None = None

    def spec(self) -> ModelSpec:
        return ModelSpec(
            name=self.name,
            vram_peak=self.vram_peak,
            vram_resident=self.vram_resident,
            host_peak=self.host_peak,
            priority=self.priority,
            pinned=self.pinned,
            device=self.device,
        )


@dataclass
class SatelliteConfig:
    name: str
    command: list[str] | None = None
    adapter: str | None = None
    url: str | None = None
    autostart: bool = True
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    restart: RestartPolicy = RestartPolicy.ON_FAILURE
    restart_backoff_s: float = 1.0
    health_url: str | None = None
    start_timeout_s: float = 120.0
    kill_grace_s: float | None = None
    load_timeout_s: float = 600.0
    models: list[ModelConfig] = field(default_factory=list)

    @property
    def kind(self) -> SatelliteKind:
        if self.adapter:
            return SatelliteKind.ADAPTER
        if self.models:
            return SatelliteKind.BLACK_BOX
        return SatelliteKind.COOPERATIVE


@dataclass
class Config:
    socket: str | None = None
    socket_mode: int = 0o600
    settings: ArbiterSettings = field(default_factory=ArbiterSettings)
    satellites: list[SatelliteConfig] = field(default_factory=list)
    log_level: str = "info"
    #: ``"nvml"`` (default) or ``"fake"`` for trying vrambiter on a machine without a GPU.
    gpu: str = "nvml"
    fake_vram: int = 24 * 1024**3
    source: str = "<defaults>"
    #: Drop-in files merged in by :func:`apply_fragments`, in the order applied.
    fragments: list[str] = field(default_factory=list)


def _read_toml(p: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(p.read_text())
    except FileNotFoundError:
        raise ConfigError(f"{p}: no such file") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p}: invalid TOML: {exc}") from None


def load_config(path: str | os.PathLike[str]) -> Config:
    """Read and validate a TOML file."""
    p = Path(path)
    return parse_config(_read_toml(p), source=str(p))


def fragment_dirs() -> list[Path]:
    """Drop-in directories, least important first.

    ``$VRAMBITER_CONFIG_DIR`` (``os.pathsep``-separated) replaces the search. Otherwise every
    ``$XDG_CONFIG_DIRS`` entry (default ``/etc/xdg``, where a package installs) and then
    ``$XDG_CONFIG_HOME`` (default ``~/.config``), each with ``vrambiter/conf.d`` appended.
    """
    explicit = os.environ.get("VRAMBITER_CONFIG_DIR")
    if explicit:
        return [Path(d) for d in explicit.split(os.pathsep) if d]
    system = os.environ.get("XDG_CONFIG_DIRS") or "/etc/xdg"
    user = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    # XDG lists the most important system directory first; this list runs least important first.
    bases = [d for d in system.split(os.pathsep) if d][::-1] + [user]
    return [Path(b) / "vrambiter" / FRAGMENT_DIR for b in bases]


def find_fragments(dirs: Sequence[str | os.PathLike[str]]) -> list[Path]:
    """The ``*.toml`` files in ``dirs``, ordered by file name.

    ``dirs`` runs least important first: a file in a later directory replaces one of the same name
    in an earlier one. Missing directories are skipped.
    """
    chosen: dict[str, Path] = {}
    for d in map(Path, dirs):
        if not d.is_dir():
            continue
        for f in d.glob("*.toml"):
            if f.is_file():
                chosen[f.name] = f
    return [chosen[name] for name in sorted(chosen)]


def apply_fragments(config: Config, paths: Sequence[str | os.PathLike[str]]) -> Config:
    """Merge drop-in fragments into ``config`` (in place) and return it.

    A fragment adds satellites and profile pins. ``[arbiter]`` is refused: the machine's settings
    have one owner, its ``vrambiter.toml``. A satellite defined twice is an error naming both files.
    """
    defined_in = {sat.name: config.source for sat in config.satellites}
    for path in map(Path, paths):
        data = _read_toml(path)
        if "arbiter" in data:
            raise ConfigError(
                f"{path}: [arbiter] settings belong in the machine's vrambiter.toml; a fragment "
                "may only add [[satellite]] entries and [profiles.*] pins"
            )
        try:
            fragment = parse_config(data, source=str(path))
        except ConfigError as exc:
            message = str(exc)
            raise ConfigError(
                message if message.startswith(str(path)) else f"{path}: {message}"
            ) from None
        for sat in fragment.satellites:
            if sat.name in defined_in:
                raise ConfigError(
                    f"{path}: satellite {sat.name!r} is already defined in {defined_in[sat.name]}"
                )
            defined_in[sat.name] = str(path)
            config.satellites.append(sat)
        for name, pins in fragment.settings.profiles.items():
            merged = config.settings.profiles.setdefault(name, [])
            merged.extend(pin for pin in pins if pin not in merged)
        config.fragments.append(str(path))
    return config


class _Table:
    """Typed, located access to one TOML table; reports unknown keys on :meth:`done`."""

    def __init__(self, data: Any, where: str) -> None:
        if not isinstance(data, dict):
            raise ConfigError(f"{where}: expected a table")
        self.data = data
        self.where = where
        self.seen: set[str] = set()

    def _get(self, key: str) -> Any:
        self.seen.add(key)
        return self.data.get(key)

    def _fail(self, key: str, message: str) -> ConfigError:
        return ConfigError(f"{self.where}.{key}: {message}")

    def get_str(self, key: str, default: Any = None, *, required: bool = False) -> Any:
        value = self._get(key)
        if value is None:
            if required:
                raise self._fail(key, "required")
            return default
        if not isinstance(value, str) or not value:
            raise self._fail(key, "expected a non-empty string")
        return value

    def get_bool(self, key: str, default: bool) -> bool:
        value = self._get(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            raise self._fail(key, "expected true or false")
        return value

    def get_int(self, key: str, default: int, *, minimum: int | None = None) -> int:
        value = self._get(key)
        if value is None:
            return default
        if not isinstance(value, int) or isinstance(value, bool):
            raise self._fail(key, "expected an integer")
        if minimum is not None and value < minimum:
            raise self._fail(key, f"must be >= {minimum}")
        return value

    def get_float(self, key: str, default: Any, *, positive: bool = True) -> Any:
        value = self._get(key)
        if value is None:
            return default
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise self._fail(key, "expected a number of seconds")
        if positive and value <= 0:
            raise self._fail(key, "must be > 0")
        return float(value)

    def get_size(self, key: str, default: Any = None, *, required: bool = False) -> Any:
        value = self._get(key)
        if value is None:
            if required:
                raise self._fail(key, 'required (e.g. "9GiB")')
            return default
        try:
            return parse_bytes(value)
        except ValueError as exc:
            raise self._fail(key, str(exc)) from None

    def get_str_list(self, key: str) -> list[str] | None:
        value = self._get(key)
        if value is None:
            return None
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise self._fail(key, "expected a list of strings")
        return list(value)

    def get_str_map(self, key: str) -> dict[str, str]:
        value = self._get(key)
        if value is None:
            return {}
        if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
            raise self._fail(key, "expected a table of strings")
        return dict(value)

    def get_tables(self, key: str) -> list[Any]:
        value = self._get(key)
        if value is None:
            return []
        if not isinstance(value, list):
            raise self._fail(key, "expected an array of tables ([[...]])")
        return value

    def done(self) -> None:
        unknown = sorted(set(self.data) - self.seen)
        if unknown:
            raise ConfigError(f"{self.where}: unknown key(s) {', '.join(unknown)}")


def parse_config(data: dict[str, Any], *, source: str = "<config>") -> Config:
    """Validate a decoded TOML document."""
    root = _Table(data, source)
    config = Config(source=source)

    arbiter = _Table(root._get("arbiter") or {}, "arbiter")
    defaults = ArbiterSettings()
    config.socket = arbiter.get_str("socket")
    mode = arbiter._get("socket_mode")
    if mode is not None:
        try:
            config.socket_mode = int(mode, 8) if isinstance(mode, str) else int(mode)
        except ValueError:
            raise ConfigError('arbiter.socket_mode: expected an octal string like "0660"') from None
    config.log_level = arbiter.get_str("log_level", "info")
    config.gpu = arbiter.get_str("gpu", "nvml")
    if config.gpu not in ("nvml", "fake"):
        raise ConfigError('arbiter.gpu: expected "nvml" or "fake"')
    config.fake_vram = arbiter.get_size("fake_vram", config.fake_vram)
    settings = ArbiterSettings(
        headroom=arbiter.get_size("headroom", defaults.headroom),
        host_headroom=arbiter.get_size("host_headroom", defaults.host_headroom),
        max_concurrent_loads=arbiter.get_int(
            "max_concurrent_loads", defaults.max_concurrent_loads, minimum=1
        ),
        poll_interval_s=arbiter.get_float("poll_interval_s", defaults.poll_interval_s),
        evict_timeout_s=arbiter.get_float("evict_timeout_s", defaults.evict_timeout_s),
        settle_timeout_s=arbiter.get_float("settle_timeout_s", defaults.settle_timeout_s),
        refuse_cooldown_s=arbiter.get_float("refuse_cooldown_s", defaults.refuse_cooldown_s),
        kill_grace_s=arbiter.get_float("kill_grace_s", defaults.kill_grace_s),
        kill_on_evict_timeout=arbiter.get_bool("kill_on_evict_timeout", False),
    )
    arbiter.done()

    profiles = root._get("profiles") or {}
    if not isinstance(profiles, dict):
        raise ConfigError("profiles: expected tables like [profiles.party]")
    for pname, ptable in profiles.items():
        t = _Table(ptable, f"profiles.{pname}")
        settings.profiles[pname] = t.get_str_list("pin") or []
        t.done()
    config.settings = settings

    names: set[str] = set()
    for i, raw in enumerate(root.get_tables("satellite")):
        sat = _parse_satellite(_Table(raw, f"satellite[{i}]"))
        if sat.name in names:
            raise ConfigError(f"satellite[{i}].name: duplicate satellite {sat.name!r}")
        names.add(sat.name)
        config.satellites.append(sat)
    root.done()
    return config


def _parse_satellite(t: _Table) -> SatelliteConfig:
    name = t.get_str("name", required=True)
    if not NAME_PATTERN.match(name):
        raise ConfigError(f"{t.where}.name: {name!r} is not a valid satellite name (no '/')")
    command = t.get_str_list("command")
    if command is not None and not command:
        raise ConfigError(f"{t.where}.command: must not be empty")
    restart_raw = t.get_str("restart", RestartPolicy.ON_FAILURE.value)
    try:
        restart = RestartPolicy(restart_raw)
    except ValueError:
        raise ConfigError(
            f"{t.where}.restart: expected one of {', '.join(r.value for r in RestartPolicy)}"
        ) from None
    sat = SatelliteConfig(
        name=name,
        command=command,
        adapter=t.get_str("adapter"),
        url=t.get_str("url"),
        autostart=t.get_bool("autostart", True),
        cwd=t.get_str("cwd"),
        env=t.get_str_map("env"),
        restart=restart,
        restart_backoff_s=t.get_float("restart_backoff_s", 1.0),
        health_url=t.get_str("health_url"),
        start_timeout_s=t.get_float("start_timeout_s", 120.0),
        kill_grace_s=t.get_float("kill_grace_s", None),
        load_timeout_s=t.get_float("load_timeout_s", 600.0),
    )
    model_names: set[str] = set()
    for j, raw in enumerate(t.get_tables("model")):
        m = _Table(raw, f"{t.where}.model[{j}]")
        model = ModelConfig(
            name=m.get_str("name", required=True),
            vram_peak=m.get_size("vram_peak", required=True),
            vram_resident=m.get_size("vram_resident"),
            host_peak=m.get_size("host_peak", 0),
            priority=m.get_int("priority", 0),
            pinned=m.get_bool("pinned", False),
            device=m.get_int("device", 0, minimum=0),
            router_model=m.get_str("router_model"),
        )
        m.done()
        if model.vram_resident is not None and model.vram_resident > model.vram_peak:
            raise ConfigError(f"{m.where}.vram_resident: exceeds vram_peak")
        if model.name in model_names:
            raise ConfigError(f"{m.where}.name: duplicate model {model.name!r}")
        if model.router_model and sat.adapter != "llama-router":
            raise ConfigError(
                f"{m.where}.router_model: only meaningful with the llama-router adapter"
            )
        model_names.add(model.name)
        sat.models.append(model)
    t.done()

    if sat.adapter is not None:
        if sat.adapter not in ADAPTERS:
            raise ConfigError(
                f"{t.where}.adapter: unknown adapter {sat.adapter!r} ({', '.join(ADAPTERS)})"
            )
        if not sat.url:
            raise ConfigError(f"{t.where}.url: required for adapter {sat.adapter!r}")
        if not sat.models:
            raise ConfigError(f"{t.where}: an adapter satellite needs [[satellite.model]] entries")
    elif sat.models:
        if not sat.command:
            raise ConfigError(f"{t.where}.command: a black-box satellite needs a command")
        if len(sat.models) != 1:
            raise ConfigError(
                f"{t.where}: a black-box satellite (no adapter) is one process, so exactly one "
                "[[satellite.model]]"
            )
    elif not sat.command:
        raise ConfigError(
            f"{t.where}: needs a command (cooperative), models (black box) or an adapter"
        )
    return sat
