"""PicameraIntake — the §4.10 picamera event plane.

The source-agnostic HostTracker + EventManager (host_tracker_design.md), driven
from the imagenode detector callback rather than a device drain thread (the OAK
difference, §4.10(i)). SpyGlass is demoted to a raw object detector: motion gates
*whether* a frame gets a SpyGlass inference (the NN-scheduler role, §4.7), and
this adapter consumes the results.

Two entry points, mirroring the tracker's two clocks:

  observe(ct, detections)     the NN ran on the frame captured at `ct`. `detections`
                              is the structured lens output — a list of
                              (class_name, (x1,y1,x2,y2) px, confidence) tuples (§4.10
                              detection contract, same shape the OAK device carries).
                              Decode -> tracker.observe() -> event manager. An EMPTY
                              list is a valid "saw nothing" observation (a present
                              subject that left is correctly missed) — it is still an
                              observe(), NOT a quiet frame.
  tick(ct)                    a motion-quiet frame: no NN inference ran. Advance the
                              tracker clock only — never observe([]), which would
                              falsely gap-out a standing subject (§4.10(ii)).

Both paths run the EventManager so a subject that banks to QUIESCENT on a tick
closes its event just as it would on an observe.

The adapter is transport-free and depthai-free: the caller (outpost.object_tracker
picamera branch) owns motion detection, the SpyGlass single-frame IPC, and the
device->float capture-time mapping; it hands this adapter only decoded geometry.
`classify` maps a specific class name ("car") to a baseclass ("vehicle") or None to
drop (the interesting-class filter, host_tracker §3).
"""

from __future__ import annotations

from typing import Callable, Optional


class PicameraIntake:
    def __init__(
        self,
        view: str,
        camsize: tuple,                                  # (width, height) of the detection frame
        tracker,                                         # HostTracker
        event_manager,                                   # EventManager
        classify: Callable[[Optional[str]], Optional[str]],
    ) -> None:
        self.view = view
        self.cam_w, self.cam_h = camsize
        self._tracker = tracker
        self._events = event_manager
        self._classify = classify
        # Clock model + monotonic guard. The caller stamps BOTH observe() and tick()
        # with the CURRENT frame's capture-time — not the lagged frametime of the
        # frame the NN actually ran on. The SpyGlass is async, so a DETECT result
        # lands ~0.2-1s after its source frame; its geometry is simply applied at
        # "now", which is correct because detections are the only position source and
        # there is no fresher one at the 1-5 fps cadence. Current-time stamping keeps
        # the tracker's window math (quiescence_window, confirm_time, gap accounting)
        # on a clean non-decreasing clock with real wall intervals — and lets
        # confirm_time see a genuine span between consecutive detections (clamping a
        # stale frametime forward instead would collapse a burst onto one instant and
        # never confirm). _advance() is then a cheap belt-and-suspenders guard against
        # metadata jitter / rare reorder. OAK never needs this (every frame is
        # observe() at strictly-increasing device time).
        self._last_ct: Optional[float] = None
        # keepalive telemetry (surfaced to the detector callback / heartbeat)
        self.dets_seen = 0
        self.observes = 0
        self.ticks = 0

    def _advance(self, capture_time: float) -> float:
        ct = capture_time if self._last_ct is None else max(capture_time, self._last_ct)
        self._last_ct = ct
        return ct

    # -- the two clocks --------------------------------------------------- #

    def observe(self, capture_time: float, detections) -> None:
        """A SpyGlass DETECT result is in. `capture_time` is the CURRENT frame's
        capture-time (clock note in __init__): the result's geometry is from a
        slightly earlier frame but applied now. Empty list == a valid miss.
        `detections` is the structured lens output: (class_name, bbox_px, conf)."""
        ct = self._advance(capture_time)
        dets = self._decode(detections)
        self.dets_seen += len(dets)
        self.observes += 1
        transitions = self._tracker.observe(ct, dets)
        self._events.process(ct, transitions, self._tracker)

    def tick(self, capture_time: float) -> None:
        """A motion-quiet frame: advance the clock, never observe([])."""
        ct = self._advance(capture_time)
        self.ticks += 1
        transitions = self._tracker.tick(ct)
        self._events.process(ct, transitions, self._tracker)

    # -- helpers ---------------------------------------------------------- #

    def _decode(self, detections) -> list:
        """Structured lens output -> tracker detections (baseclass, bbox, label, conf):
        each input is (class_name, (x1,y1,x2,y2) px, confidence). Pixel boxes are
        normalized to [0,1] against the detection frame; the specific class is mapped
        to a baseclass (None dropped at ingest); the display label ("car: 0.9600") is
        BUILT here from the structured fields. 4-tuple matches HostTracker.observe()
        and converges with the OAK intake's decode shape (outpost_intake §decode)."""
        out = []
        W, H = self.cam_w, self.cam_h
        for det in detections:
            name, bbox, conf = det[0], det[1], float(det[2])
            base = self._classify(name)
            if base is None:                             # not interesting — drop at ingest
                continue
            x1, y1, x2, y2 = bbox
            nbbox = (x1 / W, y1 / H, x2 / W, y2 / H)
            label = f"{name}: {conf:.4f}"
            out.append((base, nbbox, label, conf))
        return out
