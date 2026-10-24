"""WallClock — maps OAK capture-time to wall-clock, and survives a clock step.

Frame timestamps from ImgFrame.getTimestamp() are already on the host's
monotonic clock (DepthAI syncs device time to std::chrono::steady_clock, which
is CLOCK_MONOTONIC on Linux — the same clock as time.monotonic()). So the wall
time of a frame is simply

    wall = ct + offset,    offset = time.time() - time.monotonic()

NTP slews the two clocks together, so the offset is constant in steady state.
It moves only when the wall clock is STEPPED — which is exactly what happens on
a node without an RTC that boots on a stale clock and is corrected by NTP a few
minutes later. The previous design anchored once on the first frame and never
looked again; east carried a -3h46m error for four days after a power outage
(2026-10-19) because the outpost started before NTP did.

Two rules keep the cross-stream correlation intact through a step:

  1. The offset is chosen by the frame's capture-time, not by when stamp() is
     called. Scene (Plane 1) and trk/crp (Plane 2) stamp the same frame from
     different threads at different moments; each frame must map one way only,
     so image[N] and trk[N] keep their identical timestamp string. A rebase
     applies to frames at or after the capture-time it was taken at, and older
     frames still find the offset they were stamped with.

  2. The rebase is taken only between events, on the thread that owns event
     state (the intake drain, via observe()). A 3-hour jump inside one event
     wrecks image naming and every speed computed across it; an event that is
     uniformly wrong is the lesser harm. A defer limit bounds the wait so a
     never-idle scene cannot hold the old offset forever.

stamp() is a pure lookup — no clock reads — and is safe from any thread: the
segment history is an immutable tuple replaced wholesale, never mutated.

A side effect worth knowing: the old first-frame anchor baked that frame's
host processing latency (~0.1s) into every timestamp. This mapping stamps true
capture time.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable, Optional


class WallClock:
    def __init__(
        self,
        step_threshold: float = 0.25,     # s; timesyncd steps for offsets > 0.4s
        defer_limit: float = 120.0,       # s to wait for an idle moment before forcing
        history: int = 8,                 # segments retained for lagging stampers
        wall: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = step_threshold
        self._defer_limit = defer_limit
        self._history = history
        self._wall = wall
        self._mono = mono
        # (start_ct, offset), oldest first. The first segment covers all time.
        self._segments: tuple = ((float("-inf"), self._sample()),)
        self._pending_since: Optional[float] = None   # monotonic, step first seen
        self.rebases = 0

    @property
    def offset(self) -> float:
        return self._segments[-1][1]

    def _sample(self) -> float:
        return self._wall() - self._mono()

    def _offset_for(self, ct: float) -> float:
        segments = self._segments                    # one read; never mutated
        for start, offset in reversed(segments):
            if ct >= start:
                return offset
        return segments[0][1]                        # older than retained history

    def stamp(self, ct: float) -> str:
        """ISO wall-clock for a capture-time on the host monotonic clock."""
        return datetime.fromtimestamp(ct + self._offset_for(ct)).isoformat()

    def observe(self, ct: float, idle: bool) -> None:
        """Per-frame step check. Call from the thread that owns event state,
        before that frame is processed. `idle` = no event open and no subject
        that could open one."""
        current = self.offset
        if abs(self._sample() - current) <= self._threshold:
            self._pending_since = None               # no step, or it reverted
            return
        # Confirm with a fresh sample: a thread preempted between the two clock
        # reads produces one wild sample, a real step produces every one after.
        sample = self._sample()
        if abs(sample - current) <= self._threshold:
            return
        now = self._mono()
        if self._pending_since is None:
            self._pending_since = now
            if not idle:
                logging.warning(
                    f"WallClock: wall clock stepped {sample - current:+.6f}s; "
                    f"rebase deferred until the event plane is idle")
        waited = now - self._pending_since
        forced = not idle and waited >= self._defer_limit
        if not (idle or forced):
            return
        self._segments = self._segments[-(self._history - 1):] + ((ct, sample),)
        self._pending_since = None
        self.rebases += 1
        note = ""
        if waited > 0:
            note = f" after deferring {waited:.1f}s"
        if forced:
            note += " (defer limit reached, mid-event)"
        logging.warning(
            f"WallClock: wall clock stepped {sample - current:+.6f}s; "
            f"rebased at capture-time {ct:.6f}{note}")
