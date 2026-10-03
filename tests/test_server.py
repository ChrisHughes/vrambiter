"""The socket front end, spoken to with raw NDJSON."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
import tempfile

import pytest

from vrambiter import protocol as p
from vrambiter.arbiter import Arbiter, ArbiterSettings
from vrambiter.clock import FakeClock
from vrambiter.gpu import FakeGpu
from vrambiter.host import FakeHost
from vrambiter.server import Server


@pytest.fixture
def sock_path():
    # macOS limits Unix socket paths to ~104 bytes: keep it short.
    d = tempfile.mkdtemp(dir="/tmp", prefix="vb")
    yield os.path.join(d, "s.sock")
    for name in os.listdir(d):
        os.unlink(os.path.join(d, name))
    os.rmdir(d)


@pytest.fixture
async def served(sock_path):
    arb = Arbiter(FakeGpu("24GiB"), FakeHost(), settings=ArbiterSettings(), clock=FakeClock())
    await arb.start()
    server = Server(arb, sock_path)
    await server.start()
    yield arb, server
    await server.close()
    await arb.close()


class Raw:
    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer

    @classmethod
    async def open(cls, path):
        return cls(*await asyncio.open_unix_connection(path, limit=p.MAX_LINE_BYTES * 2))

    async def send(self, obj) -> None:
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode() + b"\n"
        self.writer.write(data)
        await self.writer.drain()

    async def recv(self) -> dict:
        line = await asyncio.wait_for(self.reader.readline(), 5)
        return json.loads(line)

    async def close(self):
        self.writer.close()


async def test_hello_and_status_over_the_socket(served, sock_path):
    c = await Raw.open(sock_path)
    await c.send({"type": "hello", "id": 1, "name": "raw", "pid": os.getpid()})
    welcome = await c.recv()
    assert welcome["type"] == "welcome" and welcome["id"] == 1
    await c.send({"type": "status", "id": 2})
    status = await c.recv()
    assert status["type"] == "status" and status["id"] == 2
    assert status["devices"][0]["total"] == 24 * 1024**3
    await c.close()


async def test_bad_input_is_answered_not_fatal(served, sock_path):
    c = await Raw.open(sock_path)
    await c.send(b"this is not json\n")
    assert (await c.recv())["code"] == "protocol_error"
    await c.send({"type": "teleport", "id": 9})
    reply = await c.recv()
    assert reply == {
        "type": "error",
        "id": 9,
        "code": "unknown_type",
        "message": "unknown message type 'teleport'",
        "detail": {"type": "teleport"},
    }
    await c.send({"type": "acquire", "id": 3})
    assert (await c.recv())["code"] == "protocol_error"
    await c.send(b"\n")  # blank lines are ignored
    await c.send({"type": "hello", "id": 4, "name": "ok", "pid": 1})
    assert (await c.recv())["type"] == "welcome"
    await c.close()


async def test_overlong_line_hangs_up(served, sock_path):
    c = await Raw.open(sock_path)
    await c.send(b"x" * (p.MAX_LINE_BYTES + 10) + b"\n")
    reply = await c.recv()
    assert reply["message"] == "line too long"
    assert await asyncio.wait_for(c.reader.read(), 5) == b""  # closed


async def test_disconnect_reaches_the_arbiter(served, sock_path):
    arb, _ = served
    c = await Raw.open(sock_path)
    await c.send({"type": "hello", "id": 1, "name": "gone", "pid": 1})
    await c.recv()
    assert arb.registry.satellites["gone"].connected
    await c.close()
    for _ in range(100):
        if not arb.registry.satellites["gone"].connected:
            break
        await asyncio.sleep(0.01)
    assert not arb.registry.satellites["gone"].connected


async def test_socket_permissions_and_stale_socket(sock_path):
    # A stale socket file (nobody listening) is replaced.
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(sock_path)
    stale.close()
    arb = Arbiter(FakeGpu("1GiB"), FakeHost(), clock=FakeClock())
    await arb.start()
    server = Server(arb, sock_path)
    await server.start()
    try:
        assert stat.S_IMODE(os.stat(sock_path).st_mode) == 0o600
        # A second daemon on the same path refuses to start.
        with pytest.raises(RuntimeError, match="already listening"):
            await Server(arb, sock_path).start()
    finally:
        await server.close()
        await arb.close()
    assert not os.path.exists(sock_path)


async def test_refuses_to_replace_a_regular_file(sock_path):
    with open(sock_path, "w") as fh:
        fh.write("important")
    arb = Arbiter(FakeGpu("1GiB"), FakeHost(), clock=FakeClock())
    with pytest.raises(RuntimeError, match="not a socket"):
        await Server(arb, sock_path).start()
