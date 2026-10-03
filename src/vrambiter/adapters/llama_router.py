"""``llama-server`` in router mode, driven by the arbiter.

llama.cpp's server can run as a *router* (start it without ``-m``): it lists models from its cache,
a ``--models-dir`` or a ``--models-preset`` file, and loads each one on demand in its own child
``llama-server`` instance. The API this adapter uses (tools/server/README.md, "Using multiple
models"):

* ``GET /models`` -> ``{"data": [{"id": ..., "status": {"value": "loaded" | "loading" |
  "unloaded" | "sleeping" | "downloading", "failed"?: true, "exit_code"?: n}}, ...]}``
* ``POST /models/load`` ``{"model": id}`` -> ``{"success": true}``
* ``POST /models/unload`` ``{"model": id}`` -> ``{"success": true}``
* ``GET /health``

DEAD INSTANCES. A child instance can die while the router still lists its model as ``loaded``;
every request to it then fails with HTTP 500 ``proxy error: Could not establish connection``
until the model is unloaded. So ``GET /models`` is never trusted alone: the adapter checks that
the child processes it recorded at load time are alive and, when it has none recorded, probes
``GET /props?model=<id>&autoload=false`` (answered by the child itself; a dead child gives the
500 proxy error, an unloaded model a 400). A dead instance is reported to the arbiter (which
marks the model unloaded and releases its accounting), the router entry is unloaded to clear the
stale state, and the next lease loads it again.

Run the router with ``--models-max 0`` (no router-side LRU: vrambiter decides what to evict) and
``--no-models-autoload`` (a chat request for an unloaded model fails instead of loading it behind
the arbiter's back; consumers take a lease first, which loads it). Both are recommended, not
required: the adapter polls ``/models`` and tracks loads and unloads it did not ask for.

MEASURING A MODEL. Each loaded model is a child process of the router, so its VRAM is that
child's NVML per-process usage: the adapter notes the router's children before and after a load,
and the new ones are the model. When per-process numbers are unavailable (no router pid because
the arbiter does not run the router, or a driver that hides them), it falls back to the drop in
free memory across the load, which is right as long as nothing else moved meanwhile (loads are
serialised, so usually nothing did).

Stdlib only (``urllib``), blocking: the driver runs it in worker threads.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass

from .._http import HttpError, probe, request_json
from ..arbiter import LoadResult
from ..gpu import GpuBackend
from ..host import process_tree
from ..managed import ManagedProcess, wait_until_ready
from ..state import OwnerKind

__all__ = ["LlamaRouterClient", "LlamaRouterDriver", "RouterModel"]

log = logging.getLogger("vrambiter.adapters.llama_router")


@dataclass(frozen=True)
class RouterModel:
    id: str
    status: str
    failed: bool = False
    exit_code: int | None = None


class LlamaRouterClient:
    """Thin blocking client for the router endpoints."""

    def __init__(self, url: str, *, timeout: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health(self) -> bool:
        return probe(f"{self.url}/health", timeout=min(self.timeout, 2.0))

    def models(self) -> dict[str, RouterModel]:
        body = request_json("GET", f"{self.url}/models", timeout=self.timeout)
        out: dict[str, RouterModel] = {}
        for entry in (body or {}).get("data", []):
            if not isinstance(entry, dict) or "id" not in entry:
                continue
            status = entry.get("status") or {}
            if isinstance(status, str):  # tolerate a flattened status
                status = {"value": status}
            out[str(entry["id"])] = RouterModel(
                id=str(entry["id"]),
                status=str(status.get("value", "unknown")),
                failed=bool(status.get("failed", False)),
                exit_code=status.get("exit_code"),
            )
        return out

    def probe(self, model: str) -> str:
        """Ask the model's own instance something cheap: ``"alive"``, ``"dead"`` (listed as
        loaded but its child is gone: 500 ``proxy error``), ``"unloaded"`` (400) or
        ``"unknown"``. ``autoload=false`` so the probe itself never loads anything."""
        query = urllib.parse.urlencode({"model": model, "autoload": "false"})
        try:
            request_json("GET", f"{self.url}/props?{query}", timeout=min(self.timeout, 5.0))
        except HttpError as exc:
            if exc.status == 500 and "proxy error" in str(exc):
                return "dead"
            if exc.status == 400:
                return "unloaded"
            return "unknown"
        return "alive"

    def load(self, model: str) -> None:
        body = request_json(
            "POST", f"{self.url}/models/load", {"model": model}, timeout=self.timeout
        )
        _check_success(body, "load", model)

    def unload(self, model: str) -> None:
        body = request_json(
            "POST", f"{self.url}/models/unload", {"model": model}, timeout=self.timeout
        )
        _check_success(body, "unload", model)


def _check_success(body: object, verb: str, model: str) -> None:
    if isinstance(body, dict) and body.get("success") is False:
        raise HttpError(f"router refused to {verb} {model}: {body}")


class LlamaRouterDriver:
    """The arbiter's driver for one llama-server router.

    ``models`` maps vrambiter model names to router ids. ``process`` is set when the arbiter runs
    the router itself (``command`` in the config); then loads start it on demand and its process
    tree is what GPU memory is attributed to.
    """

    kind = OwnerKind.ADAPTER

    def __init__(
        self,
        client: LlamaRouterClient,
        models: dict[str, str],
        *,
        gpu: GpuBackend,
        devices: dict[str, int] | None = None,
        process: ManagedProcess | None = None,
        load_timeout_s: float = 600.0,
        start_timeout_s: float = 120.0,
        poll_interval_s: float = 0.25,
        probe_interval_s: float = 5.0,
        children: Callable[[int], set[int]] | None = None,
    ) -> None:
        self.client = client
        self.models = dict(models)
        self.gpu = gpu
        self.devices = dict(devices or {})
        self.process = process
        self.load_timeout_s = load_timeout_s
        self.start_timeout_s = start_timeout_s
        self.poll_interval_s = poll_interval_s
        self.probe_interval_s = probe_interval_s
        self._children = children or (lambda pid: process_tree(pid) - {pid})
        #: The child processes each model's load created (empty when it was not ours to see).
        self._instance_pids: dict[str, frozenset[int]] = {}
        self._last_probe: dict[str, float] = {}

    # -- ModelDriver ------------------------------------------------------------------------------

    def root_pids(self) -> frozenset[int]:
        pid = self.process.pid if self.process is not None else None
        return frozenset({pid}) if pid else frozenset()

    async def load(self, model: str) -> LoadResult:
        await self._ensure_running()
        return await asyncio.to_thread(self._load_blocking, model)

    async def unload(self, model: str) -> None:
        await asyncio.to_thread(self._unload_blocking, model)

    async def force_stop(self) -> None:
        if self.process is not None:
            await self.process.restart_now()

    async def poll(self) -> dict[str, str] | None:
        """``{model: "loaded" | "unloaded" | "dead"}`` for the declared models: changes the
        router made behind our back, and instances that died while still listed as loaded."""
        if self.process is not None and not self.process.running:
            return None
        return await asyncio.to_thread(self._poll_blocking)

    def _poll_blocking(self) -> dict[str, str] | None:
        try:
            listed = self.client.models()
        except HttpError as exc:
            log.debug("polling %s failed: %s", self.client.url, exc)
            return None
        states: dict[str, str] = {}
        # Only declared models: the router also lists cache entries and duplicates we ignore.
        for name, router_id in self.models.items():
            entry = listed.get(router_id)
            if entry is None or entry.status in ("loading", "downloading"):
                continue
            if entry.status != "loaded":
                states[name] = "unloaded"
                self._instance_pids.pop(name, None)
            elif self._instance_dead(name, router_id, periodic=True):
                log.warning("llama-router: %s is listed as loaded but its instance is dead", name)
                self._clear_dead(name, router_id)
                states[name] = "dead"
            else:
                states[name] = "loaded"
        return states

    def _instance_dead(self, name: str, router_id: str, *, periodic: bool = False) -> bool:
        """Never trust ``GET /models`` alone: a crashed child stays listed as loaded."""
        pids = self._instance_pids.get(name)
        if pids:
            return not any(_pid_alive(pid) for pid in pids)
        now = time.monotonic()
        if periodic and now - self._last_probe.get(name, 0.0) < self.probe_interval_s:
            return False
        self._last_probe[name] = now
        return self.client.probe(router_id) == "dead"

    def _clear_dead(self, name: str, router_id: str) -> None:
        """Unload the stale entry so the router will start a fresh instance next time."""
        self._instance_pids.pop(name, None)
        try:
            self.client.unload(router_id)
        except HttpError as exc:
            log.debug("unloading dead %s failed: %s", router_id, exc)

    # -- blocking work ----------------------------------------------------------------------------

    async def _ensure_running(self) -> None:
        if self.process is None:
            return
        if not self.process.running:
            await self.process.start()
        await wait_until_ready(self.process, f"{self.client.url}/health", self.start_timeout_s)

    def _router_children(self) -> set[int] | None:
        pid = self.process.pid if self.process is not None else None
        return self._children(pid) if pid else None

    def _load_blocking(self, model: str) -> LoadResult:
        router_id = self.models.get(model, model)
        device = self.devices.get(model, 0)
        before_children = self._router_children()
        free_before = self.gpu.mem_info(device)[0]

        current = self.client.models().get(router_id)
        if current is None:
            raise RuntimeError(f"llama-server at {self.client.url} has no model {router_id!r}")
        if current.status == "loaded" and self._instance_dead(model, router_id):
            log.warning("llama-router: %s listed as loaded but dead; reloading it", router_id)
            self._clear_dead(model, router_id)
            self._wait_for(router_id, want="unloaded")
            before_children = self._router_children()
            free_before = self.gpu.mem_info(device)[0]
            current = self.client.models().get(router_id) or current
        if current.status != "loaded":
            self.client.load(router_id)
            self._wait_for(router_id, want="loaded")

        pids: frozenset[int] = frozenset()
        vram: int | None = None
        after_children = self._router_children()
        if before_children is not None and after_children is not None:
            pids = frozenset(after_children - before_children)
            usage = self.gpu.process_usage(device)
            if pids and usage is not None:
                vram = sum(usage.get(pid, 0) for pid in pids)
        if vram is None:
            vram = max(0, free_before - self.gpu.mem_info(device)[0]) or None
        if pids:
            self._instance_pids[model] = pids
        else:
            pids = self._instance_pids.get(model, frozenset())  # already loaded by us earlier
        log.info("llama-router loaded %s (%s bytes, pids %s)", router_id, vram, sorted(pids))
        return LoadResult(vram_bytes=vram, pids=pids)

    def _unload_blocking(self, model: str) -> None:
        router_id = self.models.get(model, model)
        current = self.client.models().get(router_id)
        self._instance_pids.pop(model, None)
        if current is None or current.status == "unloaded":
            return
        self.client.unload(router_id)
        self._wait_for(router_id, want="unloaded")

    def _wait_for(self, router_id: str, *, want: str) -> None:
        deadline = time.monotonic() + self.load_timeout_s
        while True:
            entry = self.client.models().get(router_id)
            if want == "unloaded" and (entry is None or entry.status in ("unloaded", "sleeping")):
                return
            if entry is not None and entry.status == want:
                return
            if want == "loaded" and entry is not None and entry.failed:
                raise RuntimeError(
                    f"llama-server failed to load {router_id} (exit code {entry.exit_code})"
                )
            if want == "loaded" and entry is not None and entry.status == "unloaded":
                # Not failed, not loading: the load request was dropped (or it was unloaded
                # immediately). Treat as a failure rather than wait out the timeout.
                raise RuntimeError(f"llama-server did not load {router_id}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"{router_id} not {want} after {self.load_timeout_s:.0f}s")
            time.sleep(self.poll_interval_s)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie answers kill(0) but is dead; ask the process table when psutil is around
        import psutil

        return bool(psutil.Process(pid).status() != psutil.STATUS_ZOMBIE)
    except Exception:
        return True
