"""The process supervisor, with tiny Python scripts as satellites (real time, real processes)."""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap

import pytest

from tests.helpers import FAKES, free_port, pid_alive, wait_for
from vrambiter.config import RestartPolicy
from vrambiter.gpu import FakeGpu
from vrambiter.managed import BlackBoxDriver, ManagedProcess

PY = sys.executable
BLACKBOX = os.path.join(FAKES, "blackbox.py")


@pytest.fixture
def gpu_file(tmp_path):
    path = tmp_path / "gpu.json"
    FakeGpu("24GiB", path=path, reap_dead=True)
    return str(path)


async def test_start_sets_environment_and_stop_terminates(tmp_path):
    out = tmp_path / "env.txt"
    code = textwrap.dedent(
        """
        import os, sys, time
        open(sys.argv[1], "w").write(os.environ["VRAMBITER_NAME"] + " " +
                                     os.environ["VRAMBITER_SOCKET"] + " " + os.environ["EXTRA"])
        time.sleep(60)
        """
    )
    exits = []
    proc = ManagedProcess(
        "tts",
        [PY, "-c", code, str(out)],
        socket_path="/tmp/x.sock",
        env={"EXTRA": "1"},
        on_exit=exits.append,
    )
    await proc.start()
    await proc.start()  # idempotent
    pid = proc.pid
    await wait_for(lambda: out.exists() and out.read_text())
    assert out.read_text() == "tts /tmp/x.sock 1"
    rc = await proc.stop()
    assert rc == -15 and exits == [-15]
    assert not proc.running and not pid_alive(pid)


async def test_sigkill_after_grace_when_sigterm_is_ignored(gpu_file):
    proc = ManagedProcess(
        "stubborn", [PY, BLACKBOX, gpu_file, "1", "--ignore-term"], kill_grace_s=0.5
    )
    await proc.start()
    await asyncio.sleep(0.5)  # let it install its SIGTERM handler
    rc = await proc.stop()
    assert rc == -9


async def test_restart_on_failure_with_backoff(gpu_file):
    exits = []
    proc = ManagedProcess(
        "crashy",
        [PY, BLACKBOX, gpu_file, "1", "--crash-after", "0.2"],
        restart=RestartPolicy.ON_FAILURE,
        restart_backoff_s=0.05,
        on_exit=exits.append,
    )
    await proc.start()
    await wait_for(lambda: proc.restarts >= 2, timeout=15)
    assert exits[:2] == [3, 3]
    await proc.stop()
    restarts = proc.restarts
    await asyncio.sleep(0.5)
    assert proc.restarts == restarts and not proc.running


async def test_no_restart_on_clean_exit_with_on_failure():
    proc = ManagedProcess(
        "once", [PY, "-c", "pass"], restart=RestartPolicy.ON_FAILURE, restart_backoff_s=0.01
    )
    await proc.start()
    await wait_for(lambda: not proc.running)
    await asyncio.sleep(0.2)
    assert proc.restarts == 0 and proc.last_returncode == 0


async def test_restart_always():
    proc = ManagedProcess(
        "loop", [PY, "-c", "pass"], restart=RestartPolicy.ALWAYS, restart_backoff_s=0.01
    )
    await proc.start()
    await wait_for(lambda: proc.restarts >= 1, timeout=10)
    await proc.stop()


async def test_children_left_in_the_group_are_reaped(tmp_path):
    pidfile = tmp_path / "child.pid"
    code = textwrap.dedent(
        """
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open(sys.argv[1], "w").write(str(child.pid))
        """
    )
    proc = ManagedProcess(
        "leader", [PY, "-c", code, str(pidfile)], restart=RestartPolicy.NEVER, kill_grace_s=1.0
    )
    await proc.start()
    await wait_for(lambda: pidfile.exists() and pidfile.read_text())
    grandchild = int(pidfile.read_text())
    # The leader exits at once; its orphan still holds (would hold) VRAM and must go too.
    await wait_for(lambda: not pid_alive(grandchild), timeout=10)


async def test_black_box_driver_waits_for_health(gpu_file):
    port = free_port()
    proc = ManagedProcess(
        "comfy", [PY, BLACKBOX, gpu_file, str(1 << 30), "--health-port", str(port)]
    )
    driver = BlackBoxDriver(proc, health_url=f"http://127.0.0.1:{port}/", start_timeout_s=20)
    await driver.load("main")
    assert driver.root_pids() == {proc.pid}
    assert FakeGpu.attach(gpu_file).usage(proc.pid) == 1 << 30
    await driver.unload("main")
    assert driver.root_pids() == frozenset()


async def test_black_box_dying_during_startup_fails_the_load():
    proc = ManagedProcess(
        "broken", [PY, "-c", "import sys; sys.exit(2)"], restart=RestartPolicy.NEVER
    )
    driver = BlackBoxDriver(proc, health_url="http://127.0.0.1:9/", start_timeout_s=20)
    with pytest.raises(RuntimeError, match="exited during startup"):
        await driver.load("main")


async def test_black_box_not_ready_in_time_is_stopped(gpu_file):
    proc = ManagedProcess("slow", [PY, BLACKBOX, gpu_file, "1"], kill_grace_s=1)
    driver = BlackBoxDriver(
        proc, health_url=f"http://127.0.0.1:{free_port()}/", start_timeout_s=0.5
    )
    with pytest.raises(RuntimeError, match="not ready"):
        await driver.load("main")
    assert not proc.running
