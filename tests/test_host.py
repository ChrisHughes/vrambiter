import os
import subprocess
import sys

from vrambiter.host import FakeHost, HostBackend, ProcMeminfo, UnknownHost, process_tree
from vrambiter.units import GiB, KiB


def test_proc_meminfo(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text(
        "MemTotal:       28311552 kB\n"
        "MemFree:         1048576 kB\n"
        "MemAvailable:   20971520 kB\n"
        "HugePages_Total:       0\n"
        "Bogus:\n"
    )
    host = ProcMeminfo(path)
    assert isinstance(host, HostBackend)
    assert host.mem_available() == 20 * GiB
    assert host.mem_total() == 28311552 * KiB


def test_proc_meminfo_old_kernel_fallback(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemFree: 1024 kB\nBuffers: 1024 kB\nCached: 2048 kB\n")
    assert ProcMeminfo(path).mem_available() == 4096 * KiB


def test_fake_and_unknown_host():
    host = FakeHost("27GiB")
    host.take("10GiB")
    assert host.mem_available() == 17 * GiB
    host.give("1GiB")
    host.set_available("5GiB")
    assert host.mem_available() == 5 * GiB
    assert host.mem_total() == 27 * GiB
    assert UnknownHost().mem_available() is None


def test_process_tree_finds_children():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert {os.getpid(), child.pid} <= process_tree(os.getpid())
        assert process_tree(child.pid) == {child.pid}
    finally:
        child.kill()
        child.wait()
