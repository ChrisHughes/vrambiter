import asyncio

import pytest

from vrambiter.clock import FakeClock, LoopClock


def test_fake_clock_fires_in_order_and_sees_its_deadline():
    clock = FakeClock(start=0.0)
    seen = []
    clock.call_later(5, lambda: seen.append(("b", clock.now())))
    clock.call_later(1, lambda: seen.append(("a", clock.now())))
    clock.call_later(5, lambda: seen.append(("c", clock.now())))
    cancelled = clock.call_later(2, lambda: seen.append(("x", clock.now())))
    cancelled.cancel()
    clock.advance(4)
    assert seen == [("a", 1.0)]
    assert clock.now() == 4.0
    clock.advance(1)
    assert seen == [("a", 1.0), ("b", 5.0), ("c", 5.0)]
    assert clock.pending() == 0


def test_fake_clock_chained_timers_fire_within_window():
    clock = FakeClock(start=0.0)
    seen = []

    def tick():
        seen.append(clock.now())
        if len(seen) < 10:
            clock.call_later(1, tick)

    clock.call_later(1, tick)
    clock.advance(3.5)
    assert seen == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        clock.advance(-1)


async def test_loop_clock():
    clock = LoopClock()
    fired = asyncio.Event()
    start = clock.now()
    clock.call_later(0.01, fired.set)
    await asyncio.wait_for(fired.wait(), 1)
    assert clock.now() - start >= 0.009
