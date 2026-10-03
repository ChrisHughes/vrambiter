import os
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

from vrambiter.gpu import (
    FakeGpu,
    FakeOutOfMemory,
    GpuBackend,
    GpuSnapshot,
    NvmlBackend,
    take_snapshot,
)
from vrambiter.units import GiB


def test_fake_gpu_allocate_free_and_snapshot():
    gpu = FakeGpu({0: "24GiB", 1: "16GiB"})
    assert isinstance(gpu, GpuBackend)
    gpu.allocate(100, "10GiB")
    gpu.allocate(100, 2 * GiB)
    gpu.allocate(200, "4GiB", device=1)
    gpu.set_unattributed("1GiB")
    assert gpu.mem_info(0) == (11 * GiB, 24 * GiB)
    assert gpu.process_usage(0) == {100: 12 * GiB}
    snap = take_snapshot(gpu)
    assert snap.devices == [0, 1]
    assert snap.free[1] == 12 * GiB
    assert snap.pid_usage({100, 200}, 0) == 12 * GiB
    assert snap.pid_usage({999}, 0) == 0
    gpu.free(100, "2GiB")
    assert gpu.usage(100) == 10 * GiB
    gpu.free(100)
    assert gpu.process_usage(0) == {}
    gpu.allocate(5, "1GiB", device=1)
    gpu.kill(200)
    assert gpu.process_usage(1) == {5: GiB}


def test_fake_gpu_out_of_memory():
    gpu = FakeGpu("8GiB")
    gpu.allocate(1, "6GiB")
    with pytest.raises(FakeOutOfMemory):
        gpu.allocate(2, "3GiB")
    assert gpu.usage(2) == 0


def test_unknown_process_usage_is_none_not_zero():
    gpu = FakeGpu("8GiB", report_process_usage=False)
    gpu.allocate(1, "1GiB")
    snap = take_snapshot(gpu)
    assert snap.usage[0] is None
    assert snap.pid_usage({1}, 0) is None
    assert GpuSnapshot(free={0: 1}, total={0: 2}, usage={0: {}}).pid_usage({1}, 0) == 0


def test_shared_fake_gpu_across_processes(tmp_path):
    path = tmp_path / "gpu.json"
    gpu = FakeGpu("8GiB", path=path, reap_dead=True)
    code = (
        "import sys, os; from vrambiter.gpu import FakeGpu; "
        "g = FakeGpu.attach(sys.argv[1]); g.allocate(os.getpid(), '3GiB'); print(os.getpid())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(path)], check=True, capture_output=True, text=True
    )
    child = int(out.stdout)
    # The child has exited, so a driver-like reaper drops its memory.
    assert child not in gpu.process_usage(0)
    gpu.allocate(os.getpid(), "1GiB")
    assert gpu.process_usage(0) == {os.getpid(): GiB}
    assert FakeGpu.attach(path).usage(os.getpid()) == GiB


def test_attach_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        FakeGpu.attach(tmp_path / "nope.json")


def _fake_pynvml(monkeypatch, compute, graphics, *, raise_graphics=False):
    mod = types.ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    mod.NVMLError = NVMLError
    mod.nvmlInit = lambda: None
    mod.nvmlShutdown = lambda: None
    mod.nvmlDeviceGetCount = lambda: 1
    mod.nvmlDeviceGetHandleByIndex = lambda i: f"h{i}"
    mod.nvmlDeviceGetMemoryInfo = lambda h: SimpleNamespace(free=5 * GiB, total=24 * GiB)
    mod.nvmlDeviceGetComputeRunningProcesses = lambda h: compute

    def graphics_fn(h):
        if raise_graphics:
            raise NVMLError("not supported")
        return graphics

    mod.nvmlDeviceGetGraphicsRunningProcesses = graphics_fn
    monkeypatch.setitem(sys.modules, "pynvml", mod)


def test_nvml_backend_with_stub(monkeypatch):
    proc = SimpleNamespace
    _fake_pynvml(
        monkeypatch,
        compute=[proc(pid=1, usedGpuMemory=3 * GiB), proc(pid=2, usedGpuMemory=None)],
        graphics=[proc(pid=1, usedGpuMemory=3 * GiB)],
    )
    nvml = NvmlBackend()
    assert nvml.devices() == [0]
    assert nvml.mem_info(0) == (5 * GiB, 24 * GiB)
    assert nvml.process_usage(0) == {1: 3 * GiB}  # max, not double-counted; None skipped
    nvml.close()
    nvml.close()


def test_nvml_backend_reports_unknown_usage(monkeypatch):
    _fake_pynvml(
        monkeypatch,
        compute=[SimpleNamespace(pid=2, usedGpuMemory=None)],
        graphics=[],
        raise_graphics=True,
    )
    assert NvmlBackend().process_usage(0) is None
    _fake_pynvml(monkeypatch, compute=[], graphics=[])
    assert NvmlBackend().process_usage(0) == {}
