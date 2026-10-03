"""Time, injected.

The arbiter never calls ``time.monotonic`` or ``asyncio.sleep`` for policy decisions. It asks a
:class:`Clock` for the current time (LRU order, idle durations) and schedules every timeout (request
deadlines, eviction deadlines, the NVML poll) with :meth:`Clock.call_later`. Production uses
:class:`LoopClock`, which is the event loop's own monotonic clock; tests use :class:`FakeClock` and
move time forward explicitly, so a 30-second eviction timeout is tested in microseconds and the
order in which timers fire is deterministic.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

__all__ = ["Clock", "TimerHandle", "LoopClock", "FakeClock"]


class TimerHandle(Protocol):
    def cancel(self) -> None: ...


class Clock(Protocol):
    def now(self) -> float:
        """Monotonic seconds. Only differences are meaningful."""
        ...

    def call_later(self, delay: float, callback: Callable[[], None]) -> TimerHandle:
        """Run ``callback`` (on the event loop thread) after ``delay`` seconds."""
        ...


class LoopClock:
    """The running event loop's clock. Must be used from inside that loop."""

    def now(self) -> float:
        return asyncio.get_running_loop().time()

    def call_later(self, delay: float, callback: Callable[[], None]) -> TimerHandle:
        return asyncio.get_running_loop().call_later(max(0.0, delay), callback)


@dataclass(order=True)
class _FakeTimer:
    deadline: float
    seq: int
    callback: Callable[[], None] = field(compare=False)
    cancelled: bool = field(default=False, compare=False)

    def cancel(self) -> None:
        self.cancelled = True


class FakeClock:
    """A clock that only moves when told to.

    :meth:`advance` fires due timers in deadline order (ties in scheduling order), setting
    :meth:`now` to each timer's deadline before calling it, so callbacks observe the time they were
    scheduled for. Timers scheduled by a callback inside the advanced window fire too.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start
        self._timers: list[_FakeTimer] = []
        self._seq = itertools.count()

    def now(self) -> float:
        return self._now

    def call_later(self, delay: float, callback: Callable[[], None]) -> TimerHandle:
        timer = _FakeTimer(self._now + max(0.0, delay), next(self._seq), callback)
        heapq.heappush(self._timers, timer)
        return timer

    def advance(self, seconds: float) -> None:
        """Move time forward by ``seconds``, firing every timer that falls due."""
        if seconds < 0:
            raise ValueError("time only moves forward")
        target = self._now + seconds
        while self._timers and self._timers[0].deadline <= target:
            timer = heapq.heappop(self._timers)
            if timer.cancelled:
                continue
            self._now = max(self._now, timer.deadline)
            timer.callback()
        self._now = target

    async def advance_async(self, seconds: float) -> None:
        """:meth:`advance`, then yield to the event loop so the callbacks' effects run."""
        self.advance(seconds)
        for _ in range(5):
            await asyncio.sleep(0)

    def pending(self) -> int:
        """Number of live timers (for tests asserting nothing leaked)."""
        return sum(1 for t in self._timers if not t.cancelled)
