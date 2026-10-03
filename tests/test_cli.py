"""The ``vrambiter`` command."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from tests.helpers import ThreadedDaemon, pid_alive
from vrambiter.cli import main, render_status
from vrambiter.client import ArbiterClient
from vrambiter.units import GiB


@pytest.fixture
def threaded(sock_path):
    td = ThreadedDaemon(sock_path, profiles={"party": ["tts/*"]})
    with td as daemon:
        yield td, daemon


def test_version(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert "vrambiter" in capsys.readouterr().out


def test_no_daemon_exits_3(sock_path, capsys):
    assert main(["--socket", sock_path, "status"]) == 3
    assert "cannot connect" in capsys.readouterr().err


def test_status_pin_unpin_evict_profile(threaded, sock_path, capsys):
    td, daemon = threaded
    tts = ArbiterClient("tts", socket=sock_path, pid=4242)
    daemon.clients.append(tts)
    weights = {"held": False}

    def load():
        daemon.gpu.allocate(4242, 4 * GiB)
        weights["held"] = True

    def unload():
        daemon.gpu.free(4242)
        weights["held"] = False

    model = tts.register("voice", vram_peak="5GiB", load=load, unload=unload)
    with model.lease():
        pass

    assert main(["status", "--socket", sock_path]) == 0
    text = capsys.readouterr().out
    assert "tts/voice" in text and "resident" in text and "GPU 0: 24.0 GiB total" in text

    assert main(["--socket", sock_path, "status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["models"][0]["id"] == "tts/voice"

    assert main(["--socket", sock_path, "pin", "tts/voice"]) == 0
    assert td.call(lambda: daemon.model("tts/voice").pinned)
    assert main(["--socket", sock_path, "unpin", "tts/voice"]) == 0
    assert not td.call(lambda: daemon.model("tts/voice").pinned)

    assert main(["--socket", sock_path, "profile", "party"]) == 0
    assert td.call(lambda: daemon.model("tts/voice").pinned)
    assert main(["--socket", sock_path, "profile", "nope"]) == 1
    assert "unknown profile" in capsys.readouterr().err
    assert main(["--socket", sock_path, "profile", "--clear"]) == 0
    assert not td.call(lambda: daemon.model("tts/voice").pinned)

    assert main(["--socket", sock_path, "evict", "tts/voice"]) == 0
    deadline = time.monotonic() + 5
    while weights["held"] and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not weights["held"]

    assert main(["--socket", sock_path, "pin", "tts/ghost"]) == 1
    assert "unknown model" in capsys.readouterr().err


def test_warm_refuses_cooperative_models(threaded, sock_path, capsys):
    _, daemon = threaded
    tts = ArbiterClient("tts", socket=sock_path, pid=4242)
    daemon.clients.append(tts)
    tts.register("voice", vram_peak="5GiB", load=lambda: None)
    assert main(["--socket", sock_path, "warm", "tts/voice"]) == 1
    assert "consumer leases" in capsys.readouterr().err


def test_run_sets_environment(sock_path):
    code = "import os; print(os.environ['VRAMBITER_NAME'], os.environ['VRAMBITER_SOCKET'])"
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "vrambiter",
            "--socket",
            sock_path,
            "run",
            "--name",
            "tts",
            "--",
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split() == ["tts", sock_path]


def test_run_needs_a_command(capsys):
    assert main(["run", "--name", "x"]) == 2


def test_run_missing_executable(capsys):
    assert main(["run", "--name", "x", "--", "/nonexistent/binary"]) == 127


def test_daemon_command_end_to_end(sock_dir, tmp_path):
    config = tmp_path / "vrambiter.toml"
    sock = os.path.join(sock_dir, "d.sock")
    config.write_text(
        f'[arbiter]\nsocket = "{sock}"\nheadroom = "1GiB"\n\n[profiles.quiet]\npin = []\n'
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vrambiter",
            "daemon",
            "--config",
            str(config),
            "--fake-gpu",
            "8GiB",
            "--log-level",
            "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not os.path.exists(sock):
            assert proc.poll() is None, proc.stderr.read()
            assert time.monotonic() < deadline
            time.sleep(0.05)
        out = subprocess.run(
            [sys.executable, "-m", "vrambiter", "--socket", sock, "status", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        status = json.loads(out.stdout)
        assert status["devices"][0]["total"] == 8 * GiB
        assert status["devices"][0]["headroom"] == GiB
        assert status["arbiter"]["profiles"] == ["quiet"]
        # A second daemon on the same socket refuses to start.
        second = subprocess.run(
            [sys.executable, "-m", "vrambiter", "daemon", "--socket", sock, "--fake-gpu", "1GiB"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert second.returncode == 1 and "already listening" in second.stderr
    finally:
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(15) == 0
    assert not os.path.exists(sock)
    assert not pid_alive(proc.pid)


def test_daemon_bad_config(tmp_path, capsys):
    config = tmp_path / "bad.toml"
    config.write_text("[arbiter]\nheadrom = 1\n")
    assert main(["daemon", "--config", str(config)]) == 2
    assert "unknown key" in capsys.readouterr().err


def test_render_status_covers_every_section():
    status = {
        "arbiter": {
            "version": "0.1",
            "pid": 1,
            "uptime_s": 3725,
            "profile": "party",
            "max_concurrent_loads": 1,
        },
        "devices": [
            {
                "index": 0,
                "total": 96 * GiB,
                "free": 40 * GiB,
                "reserved": 2 * GiB,
                "headroom": 2 * GiB,
                "effective_free": 36 * GiB,
                "pending": GiB,
            }
        ],
        "host": {
            "available": 18 * GiB,
            "total": 27 * GiB,
            "headroom": 3 * GiB,
            "loads_in_flight": 1,
        },
        "models": [
            {
                "id": "llama/gemma",
                "state": "resident",
                "leases": 0,
                "measured": int(29.7 * GiB),
                "vram_peak": 30 * GiB,
                "priority": 0,
                "pinned": True,
                "idle_s": 130,
            },
            {
                "id": "tts/voice",
                "state": "busy",
                "leases": 2,
                "reported": 8 * GiB,
                "vram_peak": 9 * GiB,
                "priority": 5,
                "pinned": False,
                "idle_s": 0,
                "unresponsive": True,
            },
            {"id": "x/off", "state": "unloaded", "leases": 0, "vram_peak": GiB},
        ],
        "satellites": [
            {
                "name": "llama",
                "pid": 7,
                "connected": True,
                "adapter": True,
                "usage": {"0": 30 * GiB},
                "residue": {"0": GiB},
            }
        ],
        "queue": [
            {
                "model": "b/n",
                "requester": "b",
                "priority": 0,
                "waiting_s": 3,
                "timeout_s": 27,
                "reason": "waiting for tts/voice",
            }
        ],
        "foreign": [{"pid": 4321, "device": 0, "bytes": 20 * GiB}],
    }
    text = render_status(status)
    for fragment in (
        "up 1h02m",
        "profile: party",
        "36.0 GiB available (1.0 GiB returning)",
        "host RAM: 18.0 GiB available of 27.0 GiB",
        "loads in flight 1/1",
        "llama/gemma",
        "29.7 GiB",
        "yes",
        "2m10s",
        "busy (unresponsive)",
        "QUEUE",
        "times out in 27s: waiting for tts/voice",
        "FOREIGN",
        "pid 4321 on GPU 0: 20.0 GiB",
        "RESIDUE",
    ):
        assert fragment in text, fragment
