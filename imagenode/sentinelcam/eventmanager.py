"""EventManager — turns tracker transitions into the OTE event log (promotion).

Spec: host_tracker_design.md §7. A separate consumer (NOT part of the tracker)
that reacts to the `(tid, from_state, to_state)` transitions the HostTracker emits
and drives the outpost's OTE log surface:

    *->ACTIVE and no event open : open event (new id) -> emit `ote start`
    while event open, each observe: emit `trk` for each ACTIVE track
    no track ACTIVE             : close event -> emit `ote end` (+ sentinel tasks)

`ote end` task submission is gated by a minimum-viable-event (MIN_EVENT) so a
sub-half-second blip submits no sentinel work (host_tracker §7). Scene-image
saving is NOT gated here — it is driven by `ote start` at CamWatcher; the entry
crop survives the confirmation debounce via the EventSampler (decoupled streams).

Wire format is matched to what CamWatcher already parses (outpost.py / spyglass.py):
  start : {'id','view','type':'start','new':True,'timestamp','camsize'}
  trk   : {'id','view','type':'trk','timestamp','obj','clas','rect'}   rect = INT px
  end   : {'id','view','type':'end','timestamp','tasks':[(name,priority),...]}
Each record is emitted as the single string  f"ote{json.dumps(record)}"  — the
same Python-logging → ZMQ PUB path the legacy outpost uses.

Two injected seams keep this testable and policy-free:
  emit(str)            how a record leaves (default: root logger .info)
  timestamp_fn(ct)->str  capture-time -> ISO wall-clock. DEFAULT IS A PLACEHOLDER
                       (wall-clock now); the real device-seqnum->wall-clock mapping
                       is an intake concern (anti-pattern #5) — wire it in there.

Production home: outpost-side (e.g. sentinelcam/eventmanager.py or within the
intake module). Tracker stays standalone; this reads it (active_tracks/has_active).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Callable, Optional

ACTIVE = "ACTIVE"


class EventManager:
    def __init__(
        self,
        view: str,
        camsize: tuple,                                  # (width, height) of the scene images
        sentinel_tasks: Optional[dict] = None,          # baseclass -> task name (+ optional 'default')
        min_event_frames: int = 3,                      # MIN_EVENT gate on task submission (§7)
        emit: Optional[Callable[[str], None]] = None,
        timestamp_fn: Optional[Callable[[float], str]] = None,
    ) -> None:
        self.view = view
        self.cam_w, self.cam_h = camsize
        self.sentinel_tasks = sentinel_tasks or {}
        self.min_event_frames = min_event_frames
        self._emit = emit or (lambda s: logging.getLogger().info(s))
        # NOTE placeholder: capture-time should map to image-capture wall-clock,
        # not datetime.now(). The device-time->wall mapping belongs to the intake.
        self._ts = timestamp_fn or (lambda ct: datetime.now().isoformat())

        self._event_id: Optional[str] = None
        self._classes: set = set()                      # baseclasses seen ACTIVE this event
        self._frames = 0                                # observes while event open (MIN_EVENT basis)
        self._start_ct: Optional[float] = None

        # telemetry (keepalive / health)
        self.events_opened = 0
        self.events_with_tasks = 0
        self.trk_emitted = 0

    @property
    def event_id(self) -> Optional[str]:
        return self._event_id

    @property
    def event_open(self) -> bool:
        return self._event_id is not None

    # -- the per-observe entry point (called every frame by the drain loop) -- #

    def process(self, capture_time: float, transitions, tracker) -> None:
        # 1. opening transitions (PROVISIONAL->ACTIVE or QUIESCENT->ACTIVE)
        for tid, _frm, to in transitions:
            if to == ACTIVE:
                t = tracker.get_track(tid)
                if t is not None:
                    self._classes.add(t.baseclass)
                if self._event_id is None:
                    self._open(capture_time)

        # 2. while open: a trk per ACTIVE track that was OBSERVED this frame
        # (fresh geometry only — a track merely bridging a miss has a stale bbox
        # and emits nothing). MIN_EVENT counts these real-tracking frames, NOT the
        # frames the event spends open bridging the K-miss gap before END.
        if self._event_id is not None:
            observed = [t for t in tracker.active_tracks()
                        if t.last_observed == capture_time]
            if observed:
                self._frames += 1
                for t in observed:
                    self._classes.add(t.baseclass)
                    t.event_id = self._event_id        # stamp so the EventSampler can join crops
                    self._emit_trk(capture_time, t)
            # close when no track remains ACTIVE (all ended or banked QUIESCENT)
            if not tracker.has_active():
                self._close(capture_time)

    # -- record builders -------------------------------------------------- #

    def _open(self, ct: float) -> None:
        self._event_id = uuid.uuid1().hex
        self._classes = set()
        self._frames = 0
        self._start_ct = ct
        self.events_opened += 1
        self._emit_ote({
            "id": self._event_id, "view": self.view, "type": "start", "new": True,
            "timestamp": self._ts(ct), "camsize": [self.cam_w, self.cam_h],
        })

    def _emit_trk(self, ct: float, track) -> None:
        x1, y1, x2, y2 = track.bbox
        rect = (int(x1 * self.cam_w), int(y1 * self.cam_h),
                int(x2 * self.cam_w), int(y2 * self.cam_h))
        self._emit_ote({
            "id": self._event_id, "view": self.view, "type": "trk",
            "timestamp": self._ts(ct), "obj": track.tid,
            # specific label + confidence ("car: 0.96") so downstream consumers
            # (VehicleSpeed mask, Watchtower overlays, re-ID) keep the detail; the
            # tracker tracks by baseclass but reports the specific class here.
            "clas": track.classname, "rect": rect,
        })
        self.trk_emitted += 1

    def _close(self, ct: float) -> None:
        tasks = []
        if self._frames >= self.min_event_frames:        # MIN_EVENT gate (§7)
            tasks = [(self.sentinel_tasks[c], 1)
                     for c in sorted(self._classes) if c in self.sentinel_tasks]
            if "default" in self.sentinel_tasks:
                tasks.append((self.sentinel_tasks["default"], 2))
            if tasks:
                self.events_with_tasks += 1
        self._emit_ote({
            "id": self._event_id, "view": self.view, "type": "end",
            "timestamp": self._ts(ct), "tasks": tasks,
        })
        self._event_id = None

    def _emit_ote(self, record: dict) -> None:
        self._emit("ote" + json.dumps(record))
