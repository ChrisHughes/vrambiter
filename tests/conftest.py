from __future__ import annotations

import os

import pytest

from tests.helpers import make_daemon, remove_tree, short_tmpdir


@pytest.fixture
def sock_dir():
    path = short_tmpdir()
    yield path
    remove_tree(path)


@pytest.fixture
def sock_path(sock_dir):
    return os.path.join(sock_dir, "a.sock")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Tests must never talk to a real daemon on this machine."""
    monkeypatch.delenv("VRAMBITER_SOCKET", raising=False)
    monkeypatch.delenv("VRAMBITER_NAME", raising=False)


@pytest.fixture
async def daemon(sock_path):
    d = make_daemon(sock_path)
    await d.start()
    yield d
    for client in d.clients:
        client.close()
    await d.stop()
