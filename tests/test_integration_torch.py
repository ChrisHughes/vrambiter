"""The torch helpers with a stub ``torch`` module, so they run without torch installed."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from vrambiter.integrations import torch as vt


def test_module_does_not_import_torch():
    code = (
        "import sys, vrambiter.integrations.torch, vrambiter.client; print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_without_torch_everything_is_a_noop(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert not vt.torch_loaded()
    assert vt.reserved_bytes(0) is None
    assert vt.allocated_bytes() is None
    vt.release_cached_memory()
    with vt.measure_reserved(0) as m:
        pass
    assert m.delta is None


@pytest.fixture
def fake_torch(monkeypatch):
    calls = []
    state = {"reserved": 1000, "allocated": 600}
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        synchronize=lambda: calls.append("sync"),
        empty_cache=lambda: calls.append("empty_cache"),
        ipc_collect=lambda: calls.append("ipc_collect"),
        memory_reserved=lambda device=None: state["reserved"],
        memory_allocated=lambda device=None: state["allocated"],
    )
    torch = types.ModuleType("torch")
    torch.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls, state


def test_release_and_measure_with_stub_torch(fake_torch):
    calls, state = fake_torch
    assert vt.torch_loaded()
    vt.release_cached_memory()
    assert calls == ["sync", "empty_cache", "ipc_collect"]
    with vt.measure_reserved(0) as m:
        state["reserved"] = 5000
    assert m.delta == 4000
    assert vt.allocated_bytes(0) == 600


def test_unload_drops_attributes_and_keys(fake_torch):
    calls, _ = fake_torch
    holder = types.SimpleNamespace(pipe=object(), vae=object(), keep=1)
    vt.unload(holder, "pipe", "vae")
    assert holder.pipe is None and holder.vae is None and holder.keep == 1
    d = {"pipe": object(), "keep": 1}
    vt.unload(d, "pipe", "missing")
    assert d == {"keep": 1}
    assert calls.count("empty_cache") == 2


def test_cuda_not_initialised_means_not_loaded(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert not vt.torch_loaded()
    assert vt.reserved_bytes() is None
