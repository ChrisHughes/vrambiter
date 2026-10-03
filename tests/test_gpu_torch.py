"""On a real NVIDIA GPU: a torch satellite allocates, gets evicted, and NVML shows the memory back.

Run with ``pytest -m gpu``. Skipped unless torch with CUDA and nvidia-ml-py are importable.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("pynvml")
if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
    pytest.skip("no CUDA device", allow_module_level=True)

from tests.helpers import Daemon, wait_for  # noqa: E402
from vrambiter.arbiter import ArbiterSettings  # noqa: E402
from vrambiter.client import ArbiterClient  # noqa: E402
from vrambiter.clock import LoopClock  # noqa: E402
from vrambiter.gpu import NvmlBackend  # noqa: E402
from vrambiter.host import default_host_backend  # noqa: E402
from vrambiter.units import GiB  # noqa: E402

SIZE = 2 * GiB


async def test_torch_satellite_eviction_returns_memory_to_the_driver(sock_path):
    nvml = NvmlBackend()
    daemon = Daemon(
        path=sock_path,
        gpu=nvml,
        host=default_host_backend(),
        clock=LoopClock(),
        settings=ArbiterSettings(headroom=GiB, poll_interval_s=0.2, settle_timeout_s=5),
    )
    await daemon.start()
    client = await asyncio.to_thread(ArbiterClient, "torch-sat", socket=sock_path)
    ctl = await asyncio.to_thread(
        ArbiterClient, "ops", socket=sock_path, role="control", reconnect=False
    )
    held: dict[str, object] = {}

    def load() -> None:
        held["weights"] = torch.empty(SIZE, dtype=torch.uint8, device="cuda:0")
        torch.cuda.synchronize()

    def unload() -> None:
        held.pop("weights", None)  # the client then runs gc.collect() and empty_cache()

    def usage() -> int:
        per_pid = nvml.process_usage(0)
        if per_pid is None:  # driver hides per-process numbers: use free memory instead
            return nvml.mem_info(0)[1] - nvml.mem_info(0)[0]
        return per_pid.get(os.getpid(), 0)

    try:
        model = await asyncio.to_thread(
            client.register, "weights", vram_peak=SIZE + GiB, load=load, unload=unload
        )

        def lease_once() -> None:
            with model.lease():
                pass

        await asyncio.to_thread(lease_once)
        await wait_for(lambda: daemon.state("torch-sat/weights") == "resident")
        loaded = usage()
        assert torch.cuda.memory_reserved(0) >= SIZE

        await asyncio.to_thread(ctl.evict, "torch-sat/weights")
        await wait_for(lambda: daemon.state("torch-sat/weights") == "unloaded", timeout=30)
        await wait_for(lambda: loaded - usage() >= 0.9 * SIZE, timeout=10)
        assert torch.cuda.memory_reserved(0) < SIZE  # empty_cache handed the pool back
    finally:
        client.close()
        ctl.close()
        await daemon.stop()
        nvml.close()
