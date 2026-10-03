"""Supervising satellite processes the arbiter starts itself.

:class:`ManagedProcess` starts a command, watches it, restarts it by policy and stops it
(``SIGTERM``, then ``SIGKILL`` after a grace period). Each process gets its own session, and
signals go to the whole *process group*: ``llama-server`` in router mode runs every model in a
child process, and stopping the router while its children keep their VRAM would defeat the point.
For the same reason, when a leader exits on its own, stragglers left in its group are terminated.

Cooperative children get ``VRAMBITER_SOCKET`` and ``VRAMBITER_NAME`` in their environment, so
``vrambiter.connect()`` in the child finds the daemon and takes the configured name.

:class:`BlackBoxDriver` turns a process that does not speak the protocol into a model: loading it
means starting the process (and waiting for its health check), evicting it means stopping it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from collections.abc import Callable, Sequence

from ._http import probe
from .arbiter import LoadResult
from .config import RestartPolicy
from .state import OwnerKind

__all__ = ["ManagedProcess", "BlackBoxDriver"]

log = logging.getLogger("vrambiter.managed")


class ManagedProcess:
    """One supervised command."""

    def __init__(
        self,
        name: str,
        command: Sequence[str],
        *,
        socket_path: str | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        restart: RestartPolicy = RestartPolicy.ON_FAILURE,
        restart_backoff_s: float = 1.0,
        max_backoff_s: float = 60.0,
        kill_grace_s: float = 10.0,
        on_exit: Callable[[int | None], None] | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self.name = name
        self.command = list(command)
        self.socket_path = socket_path
        self.env = dict(env or {})
        self.cwd = cwd
        self.restart = restart
        self.restart_backoff_s = restart_backoff_s
        self.max_backoff_s = max_backoff_s
        self.kill_grace_s = kill_grace_s
        self.on_exit = on_exit
        self.restarts = 0
        self.last_returncode: int | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._restart_task: asyncio.Task[None] | None = None
        self._want_running = False
        self._backoff = restart_backoff_s
        self._started_at = 0.0
        self._start_lock = asyncio.Lock()

    @property
    def pid(self) -> int | None:
        proc = self._proc
        return proc.pid if proc is not None and proc.returncode is None else None

    @property
    def running(self) -> bool:
        return self.pid is not None

    async def start(self) -> None:
        """Start the process if it is not running. Idempotent."""
        async with self._start_lock:
            self._want_running = True
            if self.running:
                return
            env = os.environ.copy()
            env.update(self.env)
            env["VRAMBITER_NAME"] = self.name
            if self.socket_path:
                env["VRAMBITER_SOCKET"] = self.socket_path
            proc = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=self.cwd,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True,  # own process group: signals reach its children too
                # Pass no inherited descriptors (the default, kept explicit on purpose): a router
                # launched holding, say, a flock fd hands it to every model child, which then
                # keeps the lock open forever.
                close_fds=True,
            )
            self._proc = proc
            self._started_at = asyncio.get_running_loop().time()
            self._watch_task = asyncio.create_task(self._watch(proc), name=f"watch-{self.name}")
            log.info("%s started (pid %d): %s", self.name, proc.pid, " ".join(self.command))

    async def _watch(self, proc: asyncio.subprocess.Process) -> None:
        returncode = await proc.wait()
        self.last_returncode = returncode
        if self._proc is proc:
            self._proc = None
        level = logging.INFO if returncode == 0 or not self._want_running else logging.WARNING
        log.log(level, "%s (pid %d) exited with %s", self.name, proc.pid, returncode)
        # Children left in the group (router instances) still hold VRAM: end them too.
        await self._reap_group(proc.pid)
        if self.on_exit is not None:
            try:
                self.on_exit(returncode)
            except Exception:
                log.exception("%s: exit callback failed", self.name)
        if self._should_restart(returncode):
            uptime = asyncio.get_running_loop().time() - self._started_at
            if uptime > self.max_backoff_s:
                self._backoff = self.restart_backoff_s  # it ran fine for a while: start over
            delay = self._backoff
            self._backoff = min(self._backoff * 2, self.max_backoff_s)
            log.info("%s: restarting in %.1fs", self.name, delay)
            self._restart_task = asyncio.create_task(self._restart_later(delay))

    def _should_restart(self, returncode: int | None) -> bool:
        if not self._want_running:
            return False
        if self.restart is RestartPolicy.ALWAYS:
            return True
        return self.restart is RestartPolicy.ON_FAILURE and returncode != 0

    async def _restart_later(self, delay: float) -> None:
        await asyncio.sleep(delay)
        if not self._want_running or self.running:
            return
        self.restarts += 1
        try:
            await self.start()
        except OSError as exc:
            log.error("%s: restart failed: %s", self.name, exc)

    async def stop(self, grace: float | None = None) -> int | None:
        """Stop the process group: ``SIGTERM``, then ``SIGKILL`` after ``grace`` seconds.

        Disables restarts. Returns the leader's exit code (``None`` if it was not running). Waits
        until the exit callback has run.
        """
        self._want_running = False
        if self._restart_task is not None:
            self._restart_task.cancel()
        proc = self._proc
        grace = self.kill_grace_s if grace is None else grace
        if proc is not None and proc.returncode is None:
            log.info("%s: stopping (pid %d)", self.name, proc.pid)
            _killpg(proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(asyncio.shield(proc.wait()), grace)
            except asyncio.TimeoutError:
                log.warning("%s did not exit within %.0fs of SIGTERM; killing it", self.name, grace)
                _killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
        if self._watch_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(self._watch_task)
        return proc.returncode if proc is not None else None

    async def restart_now(self) -> None:
        """Stop the process group, then start it again unless the restart policy is ``never``.

        The last resort after an eviction timed out: the models in it are gone either way, but a
        service (a cooperative satellite, a llama-server router) should come back.
        """
        policy = self.restart
        await self.stop()
        if policy is not RestartPolicy.NEVER:
            self.restarts += 1
            await self.start()

    async def _reap_group(self, pgid: int) -> None:
        if not _group_alive(pgid):
            return
        _killpg(pgid, signal.SIGTERM)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.kill_grace_s
        # There is nothing to await for "other processes in a group exited": poll.
        while _group_alive(pgid) and loop.time() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.1)
        if _group_alive(pgid):
            log.warning("%s: processes left in its group; killing them", self.name)
            _killpg(pgid, signal.SIGKILL)


def _killpg(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:  # pragma: no cover - should not happen for our own children
        with contextlib.suppress(ProcessLookupError):
            os.kill(pgid, sig)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover
        return True
    return True


class BlackBoxDriver:
    """A process that knows nothing about vrambiter, treated as one model.

    Loading starts it and waits until ``health_url`` answers 2xx (or, without one, until it has
    stayed up for ``settle_s``). Unloading stops it. Its GPU memory is measured through NVML as
    the usage of its process tree.
    """

    kind = OwnerKind.MANAGED

    def __init__(
        self,
        process: ManagedProcess,
        *,
        health_url: str | None = None,
        start_timeout_s: float = 120.0,
        settle_s: float = 0.5,
    ) -> None:
        self.process = process
        self.health_url = health_url
        self.start_timeout_s = start_timeout_s
        self.settle_s = settle_s

    async def load(self, model: str) -> LoadResult:
        await self.process.start()
        await wait_until_ready(self.process, self.health_url, self.start_timeout_s, self.settle_s)
        return LoadResult()

    async def unload(self, model: str) -> None:
        await self.process.stop()

    def root_pids(self) -> frozenset[int]:
        pid = self.process.pid
        return frozenset({pid}) if pid else frozenset()

    async def force_stop(self) -> None:
        await self.process.stop()


async def wait_until_ready(
    process: ManagedProcess, health_url: str | None, timeout: float, settle_s: float = 0.5
) -> None:
    """Wait for ``health_url`` to answer (or ``settle_s`` of uptime); fail if the process dies."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    started = loop.time()
    while True:
        if not process.running:
            raise RuntimeError(
                f"{process.name} exited during startup (exit code {process.last_returncode})"
            )
        if health_url is not None:
            if await asyncio.to_thread(probe, health_url):
                return
        elif loop.time() - started >= settle_s:
            return
        if loop.time() > deadline:
            await process.stop()
            raise RuntimeError(f"{process.name} was not ready within {timeout:.0f}s")
        await asyncio.sleep(0.2)
