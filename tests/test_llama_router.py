"""The llama-router adapter against a fake llama-server that mimics router mode."""

from __future__ import annotations

import json
import os
import sys

import pytest

from tests.helpers import FAKES, free_port, wait_for
from vrambiter._http import HttpError
from vrambiter.adapters.llama_router import LlamaRouterClient, LlamaRouterDriver
from vrambiter.config import RestartPolicy
from vrambiter.gpu import FakeGpu
from vrambiter.managed import ManagedProcess
from vrambiter.units import GiB

ROUTER = os.path.join(FAKES, "llama_router.py")


@pytest.fixture
def gpu(tmp_path):
    return FakeGpu("24GiB", path=tmp_path / "gpu.json", reap_dead=True)


def router_process(gpu: FakeGpu, port: int, **config) -> ManagedProcess:
    cfg = {"models": {"gemma": 6 * GiB, "qwen": 4 * GiB}, "load_delay": 0.1}
    cfg.update(config)
    return ManagedProcess(
        "llama",
        [sys.executable, ROUTER, str(gpu._path), str(port), json.dumps(cfg)],
        restart=RestartPolicy.NEVER,
        kill_grace_s=2,
    )


@pytest.fixture
async def router(gpu):
    port = free_port()
    proc = router_process(
        gpu, port, fail=["broken"], models={"gemma": 6 * GiB, "qwen": 4 * GiB, "broken": GiB}
    )
    client = LlamaRouterClient(f"http://127.0.0.1:{port}", timeout=5)
    driver = LlamaRouterDriver(
        client,
        {"gemma-4": "gemma", "qwen": "qwen", "broken": "broken"},
        gpu=gpu,
        process=proc,
        load_timeout_s=20,
        start_timeout_s=20,
        poll_interval_s=0.05,
    )
    yield proc, client, driver
    await proc.stop()


async def test_client_endpoints(router):
    proc, client, driver = router
    await driver._ensure_running()
    assert client.health()
    models = client.models()
    assert set(models) == {"gemma", "qwen", "broken"}
    assert models["gemma"].status == "unloaded"
    with pytest.raises(HttpError) as info:
        client.load("nope")
    assert info.value.status == 400 and "not found" in str(info.value)


async def test_load_measures_the_child_instance_and_unload_removes_it(router, gpu):
    proc, client, driver = router
    result = await driver.load("gemma-4")
    assert result.vram_bytes == 6 * GiB
    assert len(result.pids) == 1
    (child,) = result.pids
    assert gpu.usage(child) == 6 * GiB
    assert client.models()["gemma"].status == "loaded"
    assert driver.root_pids() == {proc.pid}
    # Loading what is already loaded is a no-op that still answers.
    again = await driver.load("gemma-4")
    assert again.pids == frozenset()
    await driver.unload("gemma-4")
    assert client.models()["gemma"].status == "unloaded"
    await wait_for(lambda: child not in gpu.process_usage(0))
    await driver.unload("gemma-4")  # idempotent


async def test_failed_load_raises(router):
    _, _, driver = router
    with pytest.raises(RuntimeError, match="failed to load broken"):
        await driver.load("broken")


async def test_poll_reports_external_changes(router):
    proc, client, driver = router
    assert await driver.poll() is None  # router not running yet
    await driver._ensure_running()
    assert await driver.poll() == {"gemma-4": False, "qwen": False, "broken": False}
    client.load("qwen")  # behind the arbiter's back (a request with autoload, say)
    await wait_for(lambda: client.models()["qwen"].status == "loaded")
    states = await driver.poll()
    assert states["qwen"] is True


async def test_fallback_measurement_uses_free_memory_delta(gpu):
    port = free_port()
    proc = router_process(gpu, port)
    driver = LlamaRouterDriver(
        LlamaRouterClient(f"http://127.0.0.1:{port}"),
        {"qwen": "qwen"},
        gpu=gpu,
        process=proc,
        load_timeout_s=20,
        poll_interval_s=0.05,
        children=lambda pid: set(),
    )
    try:
        result = await driver.load("qwen")
        assert result.pids == frozenset()
        assert result.vram_bytes == 4 * GiB
    finally:
        await proc.stop()


async def test_router_not_managed_by_us(gpu):
    port = free_port()
    proc = router_process(gpu, port)
    await proc.start()
    try:
        driver = LlamaRouterDriver(
            LlamaRouterClient(f"http://127.0.0.1:{port}"),
            {"qwen": "qwen"},
            gpu=gpu,
            load_timeout_s=20,
            poll_interval_s=0.05,
        )
        await wait_for(driver.client.health, timeout=10)
        assert driver.root_pids() == frozenset()
        result = await driver.load("qwen")
        assert result.vram_bytes == 4 * GiB  # no router pid: free-memory delta
        await driver.force_stop()  # nothing to stop: not ours
        assert proc.running
    finally:
        await proc.stop()


async def test_unknown_router_model(router):
    _, _, driver = router
    driver.models["ghost"] = "ghost"
    with pytest.raises(RuntimeError, match="no model 'ghost'"):
        await driver.load("ghost")
