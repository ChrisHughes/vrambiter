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
import time
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
        self._children = children or (lambda pid: process_tree(pid) - {pid})

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
            await self.process.stop()

    async def poll(self) -> dict[str, bool] | None:
        """Which declared models the router has loaded, for changes made behind our back."""
        if self.process is not None and not self.process.running:
            return None
        try:
            listed = await asyncio.to_thread(self.client.models)
        except HttpError as exc:
            log.debug("polling %s failed: %s", self.client.url, exc)
            return None
        states: dict[str, bool] = {}
        for name, router_id in self.models.items():
            entry = listed.get(router_id)
            if entry is None or entry.status in ("loading", "downloading"):
                continue
            states[name] = entry.status == "loaded"
        return states

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
        log.info("llama-router loaded %s (%s bytes, pids %s)", router_id, vram, sorted(pids))
        return LoadResult(vram_bytes=vram, pids=pids)

    def _unload_blocking(self, model: str) -> None:
        router_id = self.models.get(model, model)
        current = self.client.models().get(router_id)
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
