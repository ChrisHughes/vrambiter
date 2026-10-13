from __future__ import annotations

import os
import textwrap

import pytest

from vrambiter.config import (
    Config,
    ConfigError,
    RestartPolicy,
    SatelliteKind,
    apply_fragments,
    find_fragments,
    fragment_dirs,
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


# --- drop-in fragments -------------------------------------------------------------------------

LLAMA_FRAGMENT = textwrap.dedent(
    """
    [[satellite]]
    name = "llama"
    adapter = "llama-router"
    url = "http://127.0.0.1:8000"

      [[satellite.model]]
      name = "qwen"
      vram_peak = "40GiB"
      pinned = true

    [profiles.party]
    pin = ["llama/qwen", "tts/*"]
    """
)

DIFFUSION_FRAGMENT = textwrap.dedent(
    """
    [profiles.party]
    pin = ["tts/*", "diffusion/sdxl"]
    """
)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_fragments_add_satellites_and_merge_profile_pins(tmp_path):
    main = load_config(write(tmp_path / "vrambiter.toml", '[arbiter]\nheadroom = "3GiB"\n'))
    d = tmp_path / "conf.d"
    write(d / "20-diffusion.toml", DIFFUSION_FRAGMENT)
    write(d / "10-llama.toml", LLAMA_FRAGMENT)

    config = apply_fragments(main, find_fragments([d]))

    assert config.settings.headroom == 3 * GiB  # the machine's settings stand
    assert [s.name for s in config.satellites] == ["llama"]
    assert config.satellites[0].models[0].pinned
    # Same-named profiles merge, keeping order and dropping repeats.
    assert config.settings.profiles["party"] == ["llama/qwen", "tts/*", "diffusion/sdxl"]
    assert [p.split("/")[-1] for p in config.fragments] == ["10-llama.toml", "20-diffusion.toml"]


def test_fragments_without_a_main_file(tmp_path):
    d = tmp_path / "conf.d"
    write(d / "llama.toml", LLAMA_FRAGMENT)
    config = apply_fragments(Config(), find_fragments([d]))
    assert [s.name for s in config.satellites] == ["llama"]


def test_a_fragment_may_not_set_arbiter_settings(tmp_path):
    frag = write(tmp_path / "conf.d" / "greedy.toml", '[arbiter]\nheadroom = "0GiB"\n')
    with pytest.raises(ConfigError, match=r"greedy\.toml: \[arbiter\] settings belong"):
        apply_fragments(Config(), [frag])


def test_a_satellite_defined_twice_names_both_files(tmp_path):
    main = load_config(write(tmp_path / "vrambiter.toml", LLAMA_FRAGMENT))
    frag = write(tmp_path / "conf.d" / "again.toml", LLAMA_FRAGMENT)
    with pytest.raises(ConfigError) as exc:
        apply_fragments(main, [frag])
    message = str(exc.value)
    assert "again.toml" in message and "vrambiter.toml" in message and "'llama'" in message


def test_fragment_errors_name_the_file_and_location(tmp_path):
    bad = LLAMA_FRAGMENT.replace('vram_peak = "40GiB"', 'vram_peek = "40GiB"')
    frag = write(tmp_path / "conf.d" / "typo.toml", bad)
    with pytest.raises(ConfigError, match=r"typo\.toml: satellite\[0\]\.model\[0\].*vram_pe"):
        apply_fragments(Config(), [frag])


def test_a_more_important_directory_replaces_a_fragment_of_the_same_name(tmp_path):
    system, user = tmp_path / "etc", tmp_path / "home"
    write(system / "llama.toml", LLAMA_FRAGMENT)
    write(system / "only-system.toml", DIFFUSION_FRAGMENT)
    write(user / "llama.toml", "# masked by the user\n")
    found = find_fragments([system, user])
    assert [(p.parent.name, p.name) for p in found] == [
        ("home", "llama.toml"),
        ("etc", "only-system.toml"),
    ]
    assert apply_fragments(Config(), found).satellites == []


def test_find_fragments_skips_missing_dirs_and_other_files(tmp_path):
    d = tmp_path / "conf.d"
    write(d / "a.toml", DIFFUSION_FRAGMENT)
    write(d / "README", "not config")
    write(d / "b.toml.disabled", LLAMA_FRAGMENT)
    assert [p.name for p in find_fragments([tmp_path / "missing", d])] == ["a.toml"]


def test_fragment_dirs_follow_xdg(monkeypatch):
    monkeypatch.delenv("VRAMBITER_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_DIRS", "/etc/first:/etc/second")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/home/u/.config")
    assert [str(d) for d in fragment_dirs()] == [
        "/etc/second/vrambiter/conf.d",  # least important first
        "/etc/first/vrambiter/conf.d",
        "/home/u/.config/vrambiter/conf.d",  # the user's own wins
    ]
    monkeypatch.delenv("XDG_CONFIG_DIRS")
    assert str(fragment_dirs()[0]) == "/etc/xdg/vrambiter/conf.d"


def test_vrambiter_config_dir_replaces_the_search(monkeypatch):
    monkeypatch.setenv("VRAMBITER_CONFIG_DIR", f"/a{os.pathsep}/b")
    assert [str(d) for d in fragment_dirs()] == ["/a", "/b"]
