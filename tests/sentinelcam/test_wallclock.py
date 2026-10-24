"""WallClock: capture-time -> wall-clock across a wall-clock step.

The clocks are faked so a step can be provoked on demand. The default step is
the one east took on 2026-10-19: booted on a clock 3h46m slow after a power
outage, corrected by NTP six minutes later (measured from the journal's
realtime-vs-monotonic jump).
"""

import logging
from datetime import datetime

from sentinelcam.wallclock import WallClock

OFFSET = 1_792_419_000.0          # wall - monotonic, before the step
EAST_STEP = 13_584.718884         # s, 2026-10-19
FRAME = 1 / 30


class FakeClocks:
    def __init__(self, mono=400.0, offset=OFFSET):
        self.t = mono
        self.offset = offset
        self.glitches = []         # added to successive wall() reads, then 0

    def mono(self):
        return self.t

    def wall(self):
        extra = self.glitches.pop(0) if self.glitches else 0.0
        return self.t + self.offset + extra

    def step(self, delta):
        self.offset += delta


def make(**kw):
    clocks = FakeClocks()
    return clocks, WallClock(wall=clocks.wall, mono=clocks.mono, **kw)


def run(clock, clocks, seconds, idle):
    """Feed frames for `seconds`; returns the capture-times observed."""
    cts = []
    for _ in range(round(seconds / FRAME)):
        ct = clocks.t - 0.05                       # captured a little before now
        clock.observe(ct, idle)
        cts.append(ct)
        clocks.t += FRAME
    return cts


def iso(ct, offset):
    return datetime.fromtimestamp(ct + offset).isoformat()


def test_maps_capture_time_to_wall():
    clocks, clock = make()
    ct = clocks.t - 0.1
    assert clock.stamp(ct) == iso(ct, OFFSET)
    assert clock.stamp(ct) == clock.stamp(ct)      # every stream stamps alike


def test_step_while_idle_rebases_at_that_frame():
    clocks, clock = make()
    before = run(clock, clocks, 2, idle=True)
    clocks.step(EAST_STEP)
    after = run(clock, clocks, 2, idle=True)
    assert clock.rebases == 1
    assert clock.stamp(after[0]) == iso(after[0], OFFSET + EAST_STEP)
    # a lagging thread stamping an older frame still gets the old offset, so
    # that frame's image and trk names stay identical
    assert clock.stamp(before[-1]) == iso(before[-1], OFFSET)


def test_step_during_event_waits_for_idle():
    clocks, clock = make()
    clocks.step(EAST_STEP)
    busy = run(clock, clocks, 5, idle=False)
    assert clock.rebases == 0
    assert clock.stamp(busy[-1]) == iso(busy[-1], OFFSET)   # event uniformly old
    idle = run(clock, clocks, 1, idle=True)
    assert clock.rebases == 1
    assert clock.stamp(idle[0]) == iso(idle[0], OFFSET + EAST_STEP)
    assert clock.stamp(busy[-1]) == iso(busy[-1], OFFSET)


def test_defer_limit_forces_rebase_mid_event():
    clocks, clock = make(defer_limit=10.0)
    clocks.step(EAST_STEP)
    run(clock, clocks, 9, idle=False)
    assert clock.rebases == 0
    run(clock, clocks, 2, idle=False)
    assert clock.rebases == 1


def test_backward_step():
    clocks, clock = make()
    clocks.step(-EAST_STEP)
    after = run(clock, clocks, 1, idle=True)
    assert clock.rebases == 1
    assert clock.stamp(after[-1]) == iso(after[-1], OFFSET - EAST_STEP)


def test_change_below_threshold_is_not_a_step():
    clocks, clock = make()
    clocks.step(0.1)
    run(clock, clocks, 2, idle=True)
    assert clock.rebases == 0


def test_single_wild_sample_is_ignored():
    # a thread preempted between the two clock reads yields one bad sample
    clocks, clock = make()
    clocks.glitches = [5.0]
    run(clock, clocks, 1, idle=True)
    assert clock.rebases == 0


def test_step_that_reverts_before_idle_is_dropped():
    clocks, clock = make()
    clocks.step(EAST_STEP)
    run(clock, clocks, 2, idle=False)
    clocks.step(-EAST_STEP)
    run(clock, clocks, 2, idle=True)
    assert clock.rebases == 0


def test_frames_between_recent_rebases_keep_their_offset():
    clocks, clock = make()
    clocks.step(10.0)
    first = run(clock, clocks, 1, idle=True)
    clocks.step(10.0)
    run(clock, clocks, 1, idle=True)
    assert clock.rebases == 2
    assert clock.stamp(first[-1]) == iso(first[-1], OFFSET + 10.0)


def test_rebase_logs_the_step_size(caplog):
    # the log line is the record a data repair needs: exact step, and where
    clocks, clock = make()
    clocks.step(EAST_STEP)
    with caplog.at_level(logging.WARNING):
        run(clock, clocks, 2, idle=False)
        run(clock, clocks, 1, idle=True)
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert "+13584.718884s" in messages[0] and "deferred" in messages[0]
    assert "+13584.718884s" in messages[1] and "rebased at capture-time" in messages[1]
