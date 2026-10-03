"""The Unix-socket front end: accepts connections, frames NDJSON, hands messages to the arbiter.

Deliberately thin. Each connection gets a reader task that decodes one line at a time and calls
:meth:`Arbiter.handle` synchronously, so a connection's messages are applied strictly in order and
nothing here ever blocks the loop. Writes are buffered by the transport; a client that stops
reading is disconnected once its buffer passes :data:`MAX_WRITE_BUFFER` rather than letting the
daemon's memory grow.

Malformed input is answered, never fatal: an unknown message type or a bad field gets an ``error``
reply (with the request's ``id`` when it could be read) and the connection carries on.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import os
import socket
import stat
from pathlib import Path

from . import protocol as p
from .arbiter import Arbiter
from .errors import ProtocolError

__all__ = ["Server", "SocketConnection", "MAX_WRITE_BUFFER"]

log = logging.getLogger("vrambiter.server")

MAX_WRITE_BUFFER = 8 * 1024 * 1024


class SocketConnection:
    """One accepted client. ``send`` never blocks."""

    def __init__(self, conn_id: int, writer: asyncio.StreamWriter) -> None:
        self.id = conn_id
        self._writer = writer
        self.closed = False

    def send(self, message: p.Message) -> None:
        if self.closed:
            return
        try:
            self._writer.write(p.encode(message))
        except (ConnectionError, RuntimeError):
            self.close()
            return
        transport = self._writer.transport
        if transport is not None and transport.get_write_buffer_size() > MAX_WRITE_BUFFER:
            log.warning("connection %d is not reading its replies; closing it", self.id)
            self.close()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            with contextlib.suppress(Exception):
                self._writer.close()


class Server:
    """Serves ``arbiter`` on the Unix socket at ``path``."""

    def __init__(self, arbiter: Arbiter, path: str | os.PathLike[str], *, mode: int = 0o600):
        self.arbiter = arbiter
        self.path = Path(path)
        self.mode = mode
        self._server: asyncio.base_events.Server | None = None
        self._ids = itertools.count(1)
        self._conns: dict[int, SocketConnection] = {}
        self._handlers: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self._claim_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._server = await asyncio.start_unix_server(
            self._serve, path=str(self.path), limit=p.MAX_LINE_BYTES + 1
        )
        # The socket is the access control: whoever can connect can evict models.
        os.chmod(self.path, self.mode)
        log.info("listening on %s", self.path)

    def _claim_path(self) -> None:
        """Refuse to start over a live daemon; remove a stale socket left by a dead one."""
        if not self.path.exists() and not self.path.is_symlink():
            return
        if not stat.S_ISSOCK(self.path.lstat().st_mode):
            raise RuntimeError(f"{self.path} exists and is not a socket; refusing to replace it")
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        try:
            probe.connect(str(self.path))
        except OSError:
            log.info("removing stale socket %s", self.path)
            self.path.unlink()
        else:
            raise RuntimeError(f"another vrambiter daemon is already listening on {self.path}")
        finally:
            probe.close()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
        for conn in list(self._conns.values()):
            conn.close()
        for task in list(self._handlers):
            task.cancel()
        for task in list(self._handlers):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self._server is not None:
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        conn = SocketConnection(next(self._ids), writer)
        self._conns[conn.id] = conn
        self.arbiter.connected(conn)
        try:
            while not conn.closed:
                try:
                    line = await reader.readline()
                except ValueError:  # line longer than the limit: framing is lost, so hang up
                    conn.send(p.Error(code=ProtocolError.code, message="line too long"))
                    break
                except (ConnectionError, OSError):
                    break
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    message = p.decode_request(line)
                except p.UnknownMessageType as exc:
                    conn.send(p.Error(id=exc.request_id, **exc.to_wire()))
                    continue
                except ProtocolError as exc:
                    conn.send(p.Error(**exc.to_wire()))
                    continue
                self.arbiter.handle(conn, message)
        except asyncio.CancelledError:
            pass
        finally:
            conn.close()
            self._conns.pop(conn.id, None)
            self.arbiter.disconnected(conn)
            if task is not None:
                self._handlers.discard(task)
