"""A complete daemon from a config: managed cooperative satellites, black boxes and llama-router,
all real processes (tiny Python scripts) allocating on a FakeGpu shared through a file.

Real time throughout (processes do not run on a fake clock), with short timeouts.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys

import pytest

from tests.helpers import FAKES, free_port, wait_for
from vrambiter._http import request_json
from vrambiter.client import ArbiterClient
from vrambiter.config import parse_config
from vrambiter.daemon import Daemon
from vrambiter.gpu import FakeGpu
from vrambiter.host import FakeHost
from vrambiter.units import GiB

PY = sys.executable


@pytest.fixture
def gpu(tmp_path):
    return FakeGpu("24GiB", path=tmp_path / "gpu.json", reap_dead=True)


@pytest.fixture
async def make(sock_path, gpu):
    daemons: list[Daemon] = []
    clients: list[ArbiterClient] = []

    async def start(satellites: list[dict], **arbiter) -> Daemon:
        settings = dict(
            socket=sock_path,
            headroom="0",
            host_headroom="0",
            poll_interval_s=0.1,
            evict_timeout_s=5,
            settle_timeout_s=1,
            kill_grace_s=1,
        )
        settings.update(arbiter)
        config = parse_config({"arbiter": settings, "satellite": satellites})
        daemon = Daemon(config, gpu=gpu, host=FakeHost("64GiB"))
        await daemon.start()
        daemons.append(daemon)
        return daemon

    async def client(name: str, pid: int) -> ArbiterClient:
        c = await asyncio.to_thread(ArbiterClient, name, socket=sock_path, pid=pid)
        clients.append(c)
        return c

    start.client = client  # type: ignore[attr-defined]
    yield start
    for c in clients:
        c.close()
    for d in daemons:
        await d.stop()


def models(daemon: Daemon) -> dict[str, str]:
    return {m.id: m.state_label for m in daemon.arbiter.registry.models.values()}


async def big_model(make, gpu, size: float = 20):
    """An in-test satellite that needs most of the card."""
    c = await make.client("big", 999)
    return await asyncio.to_thread(
        c.register,
        "m",
        vram_peak=int(size * GiB),
        load=lambda: gpu.allocate(999, int(size * GiB)),
        unload=lambda: gpu.free(999),
    )


def use(model):
    with model.lease() as lease:
        return lease.action


async def test_cooperative_managed_satellite_is_evicted_by_another(make, gpu):
    daemon = await make(
        [
            {
                "name": "coop",
                "command": [
                    PY,
                    os.path.join(FAKES, "cooperative.py"),
                    str(gpu._path),
                    "voice",
                    str(10 * GiB),
                ],
            },
        ]
    )
    await wait_for(lambda: models(daemon).get("coop/voice") == "resident", timeout=20)
    coop_pid = daemon.processes["coop"].pid
    assert gpu.usage(coop_pid) == 10 * GiB
    big = await big_model(make, gpu)
    assert await asyncio.to_thread(use, big) == "load"
    assert models(daemon)["coop/voice"] == "unloaded"
    assert gpu.usage(coop_pid) == 0


async def test_crashed_cooperative_satellite_restarts_and_reregisters(make, gpu):
    daemon = await make(
        [
            {
                "name": "coop",
                "restart_backoff_s": 0.1,
                "command": [
                    PY,
                    os.path.join(FAKES, "cooperative.py"),
                    str(gpu._path),
                    "voice",
                    str(4 * GiB),
                ],
            },
        ]
    )
    await wait_for(lambda: models(daemon).get("coop/voice") == "resident", timeout=20)
    first = daemon.processes["coop"].pid
    os.kill(first, signal.SIGKILL)
    await wait_for(lambda: models(daemon)["coop/voice"] == "unloaded", timeout=10)
    await wait_for(lambda: daemon.processes["coop"].restarts >= 1, timeout=10)
    await wait_for(lambda: models(daemon).get("coop/voice") == "resident", timeout=20)
    assert daemon.processes["coop"].pid != first
    # The dead process's memory went with it (the driver, here FakeGpu, reaps dead pids).
    assert first not in gpu.process_usage(0)


async def test_black_box_loads_on_demand_and_is_stopped_to_evict(make, gpu):
    port = free_port()
    daemon = await make(
        [
            {
                "name": "bb",
                "autostart": False,
                "command": [
                    PY,
                    os.path.join(FAKES, "blackbox.py"),
                    str(gpu._path),
                    str(6 * GiB),
                    "--health-port",
                    str(port),
                ],
                "health_url": f"http://127.0.0.1:{port}/",
                "start_timeout_s": 20,
                "model": [{"name": "main", "vram_peak": "8GiB"}],
            },
        ]
    )
    assert models(daemon)["bb/main"] == "unloaded" and not daemon.processes["bb"].running
    app = await make.client("app", 500)

    def consume():
        with app.lease("bb/main") as lease:
            return lease.action, daemon.processes["bb"].pid

    action, pid = await asyncio.to_thread(consume)
    assert action == "ready" and gpu.usage(pid) == 6 * GiB
    await wait_for(lambda: daemon.arbiter.registry.models["bb/main"].measured_bytes == 6 * GiB)
    big = await big_model(make, gpu)
    assert await asyncio.to_thread(use, big) == "load"
    assert not daemon.processes["bb"].running
    assert models(daemon)["bb/main"] == "unloaded"


async def test_black_box_autostart_goes_through_admission(make, gpu):
    daemon = await make(
        [
            {
                "name": "bb",
                "autostart": True,
                "command": [PY, os.path.join(FAKES, "blackbox.py"), str(gpu._path), str(2 * GiB)],
                "model": [{"name": "main", "vram_peak": "4GiB"}],
            },
        ]
    )
    await wait_for(lambda: models(daemon)["bb/main"] == "resident", timeout=20)
    assert daemon.processes["bb"].running


async def test_black_box_ignoring_sigterm_is_killed(make, gpu):
    daemon = await make(
        [
            {
                "name": "bb",
                "autostart": True,
                "kill_grace_s": 0.5,
                "command": [
                    PY,
                    os.path.join(FAKES, "blackbox.py"),
                    str(gpu._path),
                    str(12 * GiB),
                    "--ignore-term",
                ],
                "model": [{"name": "main", "vram_peak": "12GiB"}],
            },
        ]
    )
    await wait_for(lambda: models(daemon)["bb/main"] == "resident", timeout=20)
    await asyncio.sleep(0.5)
    big = await big_model(make, gpu, size=16)
    assert await asyncio.to_thread(use, big) == "load"
    assert daemon.processes["bb"].last_returncode == -signal.SIGKILL


async def test_llama_router_consumer_lease_eviction_and_external_loads(make, gpu):
    port = free_port()
    router_cfg = {"models": {"ggml-org/gemma:Q4": 8 * GiB, "qwen": 2 * GiB}, "load_delay": 0.1}
    daemon = await make(
        [
            {
                "name": "llama",
                "adapter": "llama-router",
                "url": f"http://127.0.0.1:{port}",
                "command": [
                    PY,
                    os.path.join(FAKES, "llama_router.py"),
                    str(gpu._path),
                    str(port),
                    json.dumps(router_cfg),
                ],
                "model": [
                    {"name": "gemma", "router_model": "ggml-org/gemma:Q4", "vram_peak": "9GiB"},
                    {"name": "qwen", "vram_peak": "3GiB"},
                ],
            },
        ]
    )
    await wait_for(lambda: daemon.processes["llama"].running, timeout=10)
    app = await make.client("chat", 501)

    def chat():
        with app.lease("llama/gemma") as lease:
            listing = request_json("GET", f"http://127.0.0.1:{port}/models")
            status = {m["id"]: m["status"]["value"] for m in listing["data"]}
            return lease.action, status["ggml-org/gemma:Q4"]

    assert await asyncio.to_thread(chat) == ("ready", "loaded")
    gemma = daemon.arbiter.registry.models["llama/gemma"]
    assert gemma.reported_bytes == 8 * GiB and len(gemma.pids) == 1
    await wait_for(lambda: gemma.measured_bytes == 8 * GiB)

    big = await big_model(make, gpu, size=20)
    assert await asyncio.to_thread(use, big) == "load"
    assert models(daemon)["llama/gemma"] == "unloaded"

    # Someone loads qwen straight through the router: the poll notices.
    await asyncio.to_thread(
        request_json, "POST", f"http://127.0.0.1:{port}/models/load", {"model": "qwen"}
    )
    await wait_for(lambda: models(daemon)["llama/qwen"] == "resident", timeout=10)


async def test_daemon_stop_terminates_managed_processes(make, gpu):
    daemon = await make(
        [
            {
                "name": "coop",
                "command": [
                    PY,
                    os.path.join(FAKES, "cooperative.py"),
                    str(gpu._path),
                    "voice",
                    str(GiB),
                ],
            },
        ]
    )
    await wait_for(lambda: daemon.processes["coop"].running, timeout=10)
    proc = daemon.processes["coop"]
    await daemon.stop()
    assert not proc.running


async def test_simulated_gpu_makes_fake_mode_meaningful(sock_path):
    """gpu = "fake": memory is simulated from declared sizes, so evictions happen for real."""
    config = parse_config(
        {"arbiter": {"socket": sock_path, "gpu": "fake", "fake_vram": "24GiB", "headroom": "0"}}
    )
    daemon = Daemon(config, host=FakeHost("64GiB"))
    await daemon.start()
    clients = []
    try:
        loaded = {"a": 0, "b": 0}
        unloaded = {"a": 0, "b": 0}
        ms = {}
        for name in ("a", "b"):
            c = await asyncio.to_thread(ArbiterClient, name, socket=sock_path)
            clients.append(c)
            ms[name] = await asyncio.to_thread(
                c.register,
                "m",
                vram_peak="16GiB",
                load=lambda n=name: loaded.__setitem__(n, loaded[n] + 1),
                unload=lambda n=name: unloaded.__setitem__(n, unloaded[n] + 1),
            )
        await asyncio.to_thread(use, ms["a"])
        await asyncio.to_thread(use, ms["b"])  # 16 + 16 > 24: a/m is evicted
        assert unloaded == {"a": 1, "b": 0}
        status = await asyncio.to_thread(clients[0].status)
        assert status["devices"][0]["free"] == 8 * GiB
    finally:
        for c in clients:
            c.close()
        await daemon.stop()


async def test_llama_router_dead_instance_is_unloaded_and_reloaded(make, gpu):
    port = free_port()
    router_cfg = {"models": {"gemma": 8 * GiB}, "load_delay": 0.05}
    daemon = await make(
        [
            {
                "name": "llama",
                "adapter": "llama-router",
                "url": f"http://127.0.0.1:{port}",
                "command": [
                    PY,
                    os.path.join(FAKES, "llama_router.py"),
                    str(gpu._path),
                    str(port),
                    json.dumps(router_cfg),
                ],
                "model": [{"name": "gemma", "vram_peak": "9GiB"}],
            },
        ]
    )
    await wait_for(lambda: daemon.processes["llama"].running, timeout=10)
    app = await make.client("chat", 501)
    lease = await asyncio.to_thread(app.acquire, "llama/gemma")
    gemma = daemon.arbiter.registry.models["llama/gemma"]
    (child,) = gemma.pids
    os.kill(child, signal.SIGKILL)  # the instance crashes; the router keeps listing it loaded
    # The poll notices within a second; the model is unloaded even though it is leased.
    await wait_for(lambda: models(daemon)["llama/gemma"] == "unloaded", timeout=10)
    assert not gemma.leases
    lease.release()  # releasing a lease the arbiter already ended is harmless

    def chat():
        with app.lease("llama/gemma") as again:
            return again.action

    assert await asyncio.to_thread(chat) == "ready"
    assert gemma.pids and child not in gemma.pids
