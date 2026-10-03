"""The shipped examples stay valid."""

from __future__ import annotations

import ast
from pathlib import Path

from vrambiter.config import SatelliteKind, load_config

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_example_config_parses():
    config = load_config(EXAMPLES / "vrambiter.toml")
    kinds = {s.name: s.kind for s in config.satellites}
    assert kinds == {
        "llama": SatelliteKind.ADAPTER,
        "tts": SatelliteKind.COOPERATIVE,
        "comfy": SatelliteKind.BLACK_BOX,
    }
    assert config.settings.profiles["party"] == ["tts/*", "llama/gemma-4-26b"]


def test_example_scripts_compile():
    for path in EXAMPLES.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
