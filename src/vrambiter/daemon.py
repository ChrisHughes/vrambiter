"""Assemble a running daemon from a :class:`~vrambiter.config.Config`.

Kept apart from ``cli.py`` so tests (and anyone embedding vrambiter) can start a complete daemon,
managed processes and adapters included, inside their own event loop, with fake backends.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from .adapters.llama_router import LlamaRouterClient, LlamaRouterDriver
from .arbiter import Arbiter
from .client import default_socket_path
from .clock import Clock
from .config import Config, RestartPolicy, SatelliteConfig, SatelliteKind
from .gpu import GpuBackend, default_gpu_backend
from .host import HostBackend, default_host_backend
from .managed import BlackBoxDriver, ManagedProcess
from .policy import Phase
from .server import Server
from .state import Registry

__all__ = ["Daemon", "SimulatedGpu", "run_daemon"]

log = logging.getLogger("vrambiter.daemon")


class SimulatedGpu:
    """``--fake-gpu`` / ``gpu = "fake"``: one card whose usage is simulated from the models.

    For trying vrambiter on a machine without an NVIDIA GPU (or with satellites that do not touch
    one). A model counts its declared resident size while resident and its peak while loading or
    leased, attributed to its satellite's pid, so admission, eviction and ``status`` behave as
    they would on hardware. It reads the arbiter's registry from the measurement thread; a read
    that races a mutation is simply retried.
    """

    def __init__(self, total: int, registry: Registry | None = None) -> None:
        self.total = total
        self.registry = registry

    def devices(self) -> list[int]:
        return [0]

    def _usage(self) -> tuple[dict[int, int], int]:
        for _ in range(10):
            try:
                return self._compute()
            except RuntimeError:  # "dictionary changed size during iteration"
                continue
        return {}, 0

    def _compute(self) -> tuple[dict[int, int], int]:
        per_pid: dict[int, int] = {}
        unattributed = 0
        if self.registry is None:
            return per_pid, unattributed
        for model in list(self.registry.models.values()):
            if model.phase is Phase.UNLOADED:
                continue
            busy = model.phase is Phase.LOADING or bool(model.leases)
            nbytes = (
                model.vram_peak
                if busy
                else (model.vram_resident or model.reported_bytes or model.vram_peak)
            )
            sat = self.registry.satellites.get(model.satellite)
            pid = next(iter(model.pids), None) or (sat.pid if sat else None)
            if pid:
                per_pid[pid] = per_pid.get(pid, 0) + nbytes
            else:
                unattributed += nbytes
        return per_pid, unattributed

    def mem_info(self, device: int) -> tuple[int, int]:
        per_pid, unattributed = self._usage()
        return max(0, self.total - sum(per_pid.values()) - unattributed), self.total

    def process_usage(self, device: int) -> dict[int, int] | None:
        return self._usage()[0]

    def close(self) -> None:
        pass


class Daemon:
    """The arbiter, its socket server, and the processes and adapters from the config."""

    def __init__(
        self,
        config: Config,
        *,
        gpu: GpuBackend | None = None,
        host: HostBackend | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        simulated: SimulatedGpu | None = None
        if gpu is None:
            if config.gpu == "fake":
                gpu = simulated = SimulatedGpu(config.fake_vram)
            else:
                gpu = default_gpu_backend()
        self.gpu = gpu
        self.host = host or default_host_backend()
        self.socket_path = config.socket or default_socket_path()
        self.arbiter = Arbiter(self.gpu, self.host, settings=config.settings, clock=clock)
        if simulated is not None:
            simulated.registry = self.arbiter.registry
        self.server = Server(self.arbiter, self.socket_path, mode=config.socket_mode)
        self.processes: dict[str, ManagedProcess] = {}
        self._background: set[asyncio.Task[None]] = set()
        for sat in config.satellites:
            self._add(sat)

    def _add(self, sat: SatelliteConfig) -> None:
        process: ManagedProcess | None = None
        if sat.command:
            process = ManagedProcess(
                sat.name,
                sat.command,
                socket_path=self.socket_path,
                env=sat.env,
                cwd=sat.cwd,
                restart=sat.restart,
                restart_backoff_s=sat.restart_backoff_s,
                kill_grace_s=(
                    sat.kill_grace_s
                    if sat.kill_grace_s is not None
                    else self.config.settings.kill_grace_s
                ),
                on_exit=lambda _rc, name=sat.name: self.arbiter.satellite_process_exited(name),
            )
            self.processes[sat.name] = process

        if sat.kind is SatelliteKind.ADAPTER:
            driver = LlamaRouterDriver(
                LlamaRouterClient(sat.url or ""),
                {m.name: m.router_model or m.name for m in sat.models},
                gpu=self.gpu,
                devices={m.name: m.device for m in sat.models},
                process=process,
                load_timeout_s=sat.load_timeout_s,
                start_timeout_s=sat.start_timeout_s,
            )
            self.arbiter.add_driven_satellite(
                sat.name, driver, [m.spec() for m in sat.models], managed=process is not None
            )
        elif sat.kind is SatelliteKind.BLACK_BOX:
            assert process is not None
            # Its lifecycle is the model's: started on demand, never restarted behind our back.
            process.restart = RestartPolicy.NEVER
            black_box = BlackBoxDriver(
                process, health_url=sat.health_url, start_timeout_s=sat.start_timeout_s
            )
            self.arbiter.add_driven_satellite(
                sat.name, black_box, [m.spec() for m in sat.models], managed=True
            )
        else:
            assert process is not None
            self.arbiter.add_cooperative_satellite(sat.name, force_stop=process.stop)

    async def start(self) -> None:
        await self.arbiter.start()
        await self.server.start()
        for sat in self.config.satellites:
            if sat.autostart:
                self._background_task(self._autostart(sat), f"autostart-{sat.name}")

    def _background_task(self, coro: object, name: str) -> None:
        task: asyncio.Task[None] = asyncio.create_task(coro, name=name)  # type: ignore[arg-type]
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _autostart(self, sat: SatelliteConfig) -> None:
        try:
            if sat.kind is SatelliteKind.BLACK_BOX:
                # Starting a black box *is* loading its model: it goes through admission.
                model_id = f"{sat.name}/{sat.models[0].name}"
                lease = await self.arbiter.acquire_internal(model_id)
                self.arbiter.release_internal(lease)
            elif sat.name in self.processes:
                await self.processes[sat.name].start()
        except Exception as exc:
            log.error("autostart of %s failed: %s", sat.name, exc)

    async def stop(self) -> None:
        for task in list(self._background):
            task.cancel()
        await asyncio.gather(*(p.stop() for p in self.processes.values()), return_exceptions=True)
        await self.server.close()
        await self.arbiter.close()
        with contextlib.suppress(Exception):
            self.gpu.close()


async def run_daemon(config: Config, *, stop: asyncio.Event | None = None) -> None:
    """Run until SIGTERM/SIGINT (or ``stop`` is set), then shut down cleanly."""
    stop = stop or asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, stop.set)
    daemon = Daemon(config)
    await daemon.start()
    log.info(
        "vrambiter daemon ready on %s (%d device(s), %d configured satellite(s))",
        daemon.socket_path,
        len(daemon.gpu.devices()),
        len(config.satellites),
    )
    try:
        await stop.wait()
    finally:
        log.info("shutting down")
        await daemon.stop()
