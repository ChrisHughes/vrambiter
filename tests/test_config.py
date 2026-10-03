from __future__ import annotations

import textwrap

import pytest

from vrambiter.config import (
    ConfigError,
    RestartPolicy,
    SatelliteKind,
    load_config,
    parse_config,
)
from vrambiter.units import GiB

EXAMPLE = textwrap.dedent(
    """
    [arbiter]
    socket = "/run/user/1000/vrambiter.sock"
    headroom = "2GiB"
    host_headroom = "3GiB"
    max_concurrent_loads = 1
    poll_interval_s = 1.0
    evict_timeout_s = 30

    [profiles.party]
    pin = ["cerberus-tts/*", "cerberus-stt/*", "llama/gemma-4-26b"]

    [[satellite]]
    name = "llama"
    adapter = "llama-router"
    command = ["/home/me/llama.cpp/llama-server", "--models-preset", "models.ini", "--port", "8000"]
    url = "http://127.0.0.1:8000"
    autostart = true

      [[satellite.model]]
      name = "gemma-4-26b"
      vram_peak = "30GiB"

    [[satellite]]
    name = "cerberus-tts"
    command = ["/home/me/cerberus/services/tts/.venv/bin/cerberus-tts", "--port", "8101"]
    autostart = true
    """
)


def test_design_doc_example_parses(tmp_path):
    path = tmp_path / "vrambiter.toml"
    path.write_text(EXAMPLE)
    config = load_config(path)
    assert config.socket == "/run/user/1000/vrambiter.sock"
    assert config.settings.headroom == 2 * GiB
    assert config.settings.host_headroom == 3 * GiB
    assert config.settings.evict_timeout_s == 30.0
    assert config.settings.profiles == {
        "party": ["cerberus-tts/*", "cerberus-stt/*", "llama/gemma-4-26b"]
    }
    llama, tts = config.satellites
    assert llama.kind is SatelliteKind.ADAPTER and llama.url == "http://127.0.0.1:8000"
    assert llama.models[0].vram_peak == 30 * GiB
    assert tts.kind is SatelliteKind.COOPERATIVE
    assert tts.restart is RestartPolicy.ON_FAILURE


def test_defaults():
    config = parse_config({})
    assert config.socket is None and config.satellites == []
    assert config.settings.max_concurrent_loads == 1
    assert config.socket_mode == 0o600


def test_black_box_and_options():
    config = parse_config(
        {
            "arbiter": {"socket_mode": "0660", "gpu": "fake", "fake_vram": "48GiB"},
            "satellite": [
                {
                    "name": "comfy",
                    "command": ["python", "main.py"],
                    "health_url": "http://127.0.0.1:8188/",
                    "restart": "never",
                    "env": {"CUDA_VISIBLE_DEVICES": "0"},
                    "model": [
                        {
                            "name": "sdxl",
                            "vram_peak": "17GiB",
                            "vram_resident": "8GiB",
                            "host_peak": "14GiB",
                            "priority": 2,
                            "pinned": True,
                        }
                    ],
                }
            ],
        }
    )
    assert config.socket_mode == 0o660 and config.gpu == "fake" and config.fake_vram == 48 * GiB
    (sat,) = config.satellites
    assert sat.kind is SatelliteKind.BLACK_BOX and sat.restart is RestartPolicy.NEVER
    spec = sat.models[0].spec()
    assert (spec.vram_peak, spec.vram_resident, spec.host_peak, spec.priority, spec.pinned) == (
        17 * GiB,
        8 * GiB,
        14 * GiB,
        2,
        True,
    )


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"arbiter": {"headrom": "8GiB"}}, "arbiter: unknown key(s) headrom"),
        ({"arbiter": {"headroom": "8 parsecs"}}, "arbiter.headroom"),
        ({"arbiter": {"max_concurrent_loads": 0}}, "must be >= 1"),
        ({"arbiter": {"poll_interval_s": -1}}, "must be > 0"),
        ({"arbiter": {"gpu": "tpu"}}, "arbiter.gpu"),
        ({"arbiter": {"socket_mode": "rwx"}}, "socket_mode"),
        ({"satelite": []}, "unknown key(s) satelite"),
        ({"satellite": [{"command": ["x"]}]}, "satellite[0].name: required"),
        ({"satellite": [{"name": "a/b", "command": ["x"]}]}, "not a valid satellite name"),
        ({"satellite": [{"name": "a"}]}, "needs a command"),
        ({"satellite": [{"name": "a", "command": []}]}, "must not be empty"),
        (
            {"satellite": [{"name": "a", "command": ["x"]}, {"name": "a", "command": ["y"]}]},
            "duplicate satellite",
        ),
        (
            {
                "satellite": [
                    {
                        "name": "a",
                        "adapter": "llama-router",
                        "model": [{"name": "m", "vram_peak": "1GiB"}],
                    }
                ]
            },
            "url: required",
        ),
        ({"satellite": [{"name": "a", "adapter": "vllm", "url": "http://x"}]}, "unknown adapter"),
        (
            {"satellite": [{"name": "a", "adapter": "llama-router", "url": "http://x"}]},
            "needs [[satellite.model]]",
        ),
        (
            {
                "satellite": [
                    {
                        "name": "a",
                        "command": ["x"],
                        "model": [
                            {"name": "m", "vram_peak": "1GiB"},
                            {"name": "n", "vram_peak": "1GiB"},
                        ],
                    }
                ]
            },
            "exactly one",
        ),
        (
            {"satellite": [{"name": "a", "command": ["x"], "model": [{"name": "m"}]}]},
            "vram_peak: required",
        ),
        (
            {
                "satellite": [
                    {
                        "name": "a",
                        "command": ["x"],
                        "model": [{"name": "m", "vram_peak": "1GiB", "vram_resident": "2GiB"}],
                    }
                ]
            },
            "exceeds vram_peak",
        ),
        (
            {
                "satellite": [
                    {
                        "name": "a",
                        "command": ["x"],
                        "model": [{"name": "m", "vram_peak": "1GiB", "router_model": "x"}],
                    }
                ]
            },
            "router_model",
        ),
        ({"satellite": [{"name": "a", "command": ["x"], "restart": "sometimes"}]}, "restart"),
        ({"satellite": [{"name": "a", "command": ["x"], "autostart": "yes"}]}, "true or false"),
        ({"profiles": {"p": {"pin": "x"}}}, "list of strings"),
        ({"profiles": {"p": {"pins": ["x"]}}}, "unknown key(s) pins"),
    ],
)
def test_errors_name_their_location(data, fragment):
    with pytest.raises(ConfigError) as info:
        parse_config(data)
    assert fragment in str(info.value)


def test_file_errors(tmp_path):
    with pytest.raises(ConfigError, match="no such file"):
        load_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[arbiter\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(bad)
