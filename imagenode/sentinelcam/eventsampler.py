"""EventSampler — stage-2 host crop selection (promotion of EventSamplerCapture).

Spec: OAK_OUTPOST_REDESIGN_PLAN.md §3.5 + host_tracker_design.md §7. Promotes the
replay-validated `EventSamplerCapture` (harness_a2.py) with two deliberate changes:

  1. IDENTITY COMES FROM THE TRACKER (§3.5.4). The harness ran its OWN per-class IoU
     clustering (`_select_track`, `_SUBJECT_IOU_MATCH`, `_GAP_FRAMES`) to invent track
     identity. That is deleted: each crop is associated to an existing HostTracker
     `tid` (centroid-nearest same-class track), and phase-progression state is keyed
     by that tid. One source of identity — no second clustering to drift.

  2. NO SCENE PAIRING HERE. The harness buffered a scene-JPEG ring and saved the
     paired full frame itself (it had no CamWatcher). In production the scene goes out
     on Plane 1 and CamWatcher correlates the crop to its scene by seqnum (Phase 5),
     so the `_jpeg_buf` ring is gone. The EventSampler emits only the crop + the `crp`
     OTE correlation record.

The validated three-phase lateral logic becomes `LateralTraversalStrategy` behind a
`PhaseStrategy` interface (§3.5.2), so Approach/Dwell strategies can plug in later.

Retroactive entry crop (§7): the entry phase can fire while the track is still
PROVISIONAL (before `ote start`). Such a selection is BUFFERED per tid and flushed
once the event manager has stamped `track.event_id`; if the track never confirms
(flicker), the buffered entry is discarded. `flush_ready()` is called once per intake
tick to flush ready pendings and drop state for vanished tids.

I/O is injected (encode / publish / emit_ote / spyglass_offer) so the selection core
is unit-testable with no depthai/network. Production home: `sentinelcam/eventsampler.py`
(shared — the Sentinel can re-run selection over replay).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional, Protocol


class CropMetaLike(Protocol):
    seqnum: int
    det_idx: int
    label: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    @property
    def class_name(self) -> Optional[str]: ...


# --------------------------------------------------------------------------- #
# Phase strategies (§3.5.2)                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class PhaseState:
    """Per-tid progression state. Phase semantics are the strategy's; the base
    EventSampler only stores and passes it."""
    phase: int = 0            # strategy-defined; LateralTraversal: 0=entry 1=centre 2=far 3=done
    entry_cx: float = 0.5
    direction: int = 1        # +1 = left-to-right, -1 = right-to-left


class PhaseStrategy(Protocol):
    name: str
    def update(self, state: PhaseState, meta: CropMetaLike, track) -> Optional[str]:
        """Mutate `state`; return a phase label ('entry'|'centre'|'far'|...) if THIS
        crop should be selected, else None. `track` exposes the HostTracker history
        for strategies that need kinematics beyond the single meta."""
        ...


class LateralTraversalStrategy:
    """Validated cx-based selection for street-side cameras: entry (fully in view) →
    centre (cx≈0.5) → far (~75% across, direction-aware). Thresholds ported verbatim
    from harness_a2.py."""

    name = "lateral"

    def __init__(self, edge_margin: float = 0.05, center_half: float = 0.22,
                 far_threshold: float = 0.70) -> None:
        self.edge_margin = edge_margin
        self.center_half = center_half
        self.far_threshold = far_threshold

    def update(self, state: PhaseState, meta: CropMetaLike, track) -> Optional[str]:
        if state.phase == 3:
            return None
        cx = (meta.xmin + meta.xmax) / 2.0
        if state.phase == 0:
            fully_in = (
                meta.xmin > self.edge_margin and meta.xmax < (1.0 - self.edge_margin)
                and meta.ymin > self.edge_margin and meta.ymax < (1.0 - self.edge_margin)
            )
            if not fully_in:
                return None
            state.entry_cx = cx
            state.direction = 1 if cx < 0.5 else -1
            state.phase = 1
            return "entry"
        if state.phase == 1:
            if abs(cx - 0.5) < self.center_half:
                state.phase = 2
                return "centre"
            return None
        # phase == 2
        trigger = (cx >= self.far_threshold) if state.direction == 1 \
            else (cx <= (1.0 - self.far_threshold))
        if trigger:
            state.phase = 3
            return "far"
        return None


# --------------------------------------------------------------------------- #
# EventSampler                                                                 #
# --------------------------------------------------------------------------- #

def _frame_ct(frame) -> float:
    try:
        return frame.getTimestamp().total_seconds()
    except Exception:
        return 0.0


class EventSampler:
    def __init__(
        self,
        view: str,
        tracker,
        lookback,
        *,
        encode_crop: Callable[[object], bytes],          # device frame -> JPEG bytes
        publish_crop: Callable[[str, bytes], None],      # (imagezmq text, jpeg) -> publish
        emit_ote: Callable[[str], None],                 # crp record sink (logger path)
        spyglass_offer: Optional[Callable[[object, CropMetaLike], None]] = None,
        strategy: Optional[PhaseStrategy] = None,
        timestamp_fn: Optional[Callable[[float], str]] = None,
        assoc_max_dist: float = 0.15,                    # normalized centroid match radius
    ) -> None:
        self._view = view
        self._tracker = tracker
        self._lookback = lookback
        self._encode = encode_crop
        self._publish = publish_crop
        self._emit_ote = emit_ote
        self._spyglass = spyglass_offer
        self._strategy = strategy or LateralTraversalStrategy()
        self._ts = timestamp_fn or (lambda ct: f"{ct:.3f}")
        self._assoc_max_dist = assoc_max_dist

        self._state: dict[int, PhaseState] = {}
        self._pending: dict[int, list] = {}              # tid -> [(crop_key, meta, phase)]

        # telemetry
        self.selected = 0
        self.emitted = 0
        self.dropped_no_track = 0
        self.dropped_no_frame = 0

    # -- per-paired-crop entry point (called by the drain loop) ------------- #

    def note(self, crop_key: int, meta: CropMetaLike) -> None:
        track = self._associate(meta)
        if track is None:
            self.dropped_no_track += 1
            return
        self._flush_tid(track)                           # event_id may now be available
        state = self._state.setdefault(track.tid, PhaseState())
        phase = self._strategy.update(state, meta, track)
        if phase is None:
            return
        self.selected += 1
        if track.event_id is None:                       # entry before confirmation (§7)
            self._pending.setdefault(track.tid, []).append((crop_key, meta, phase))
        else:
            self._emit(crop_key, meta, track.event_id, track.tid, phase)

    # -- once per intake tick ---------------------------------------------- #

    def flush_ready(self) -> None:
        """Flush pendings whose track now has an event_id; drop state/pendings for
        tids no longer in the tracker (flicker entries are discarded unflushed)."""
        live = {t.tid: t for t in self._tracker.tracks}
        for tid in list(self._pending):
            t = live.get(tid)
            if t is None:
                self._pending.pop(tid, None)             # never confirmed -> discard
            elif t.event_id is not None:
                self._flush_tid(t)
        for tid in list(self._state):
            if tid not in live:
                self._state.pop(tid, None)

    # -- internals --------------------------------------------------------- #

    def _associate(self, meta: CropMetaLike):
        """Attach the crop to an existing tracker identity (centroid-nearest
        same-class track). Not a clustering pass — it only looks up an id the
        tracker already owns (§3.5.4)."""
        mcls = meta.class_name
        mcx = (meta.xmin + meta.xmax) / 2.0
        mcy = (meta.ymin + meta.ymax) / 2.0
        best = None
        best_d = self._assoc_max_dist
        for t in self._tracker.tracks:
            if t.baseclass != mcls:
                continue
            tcx = (t.bbox[0] + t.bbox[2]) / 2.0
            tcy = (t.bbox[1] + t.bbox[3]) / 2.0
            d = ((mcx - tcx) ** 2 + (mcy - tcy) ** 2) ** 0.5
            if d <= best_d:
                best_d = d
                best = t
        return best

    def _flush_tid(self, track) -> None:
        if track.event_id is None:
            return
        pend = self._pending.pop(track.tid, None)
        if not pend:
            return
        for crop_key, meta, phase in pend:
            self._emit(crop_key, meta, track.event_id, track.tid, phase)

    def _emit(self, crop_key: int, meta: CropMetaLike, event_id: str,
              tid: int, phase: str) -> None:
        entry = self._lookback.get(crop_key)
        if entry is None:                                # evicted before flush
            self.dropped_no_frame += 1
            return
        frame, _meta = entry
        jpeg = self._encode(frame)
        ts = self._ts(_frame_ct(frame))
        # Plane-2 crop publish: separate socket; text demux per §4.4.
        text = f"{self._view}|crop_{meta.class_name}|{event_id}|{tid}|{meta.seqnum}|{phase}"
        self._publish(text, jpeg)
        # crp OTE — correlation key only, no duplicated geometry (§4.5). CamWatcher
        # joins to the trk record for (event, obj, seqnum) to reach the bbox.
        self._emit_ote("ote" + json.dumps({
            "id": event_id, "view": self._view, "type": "crp", "timestamp": ts,
            "obj": tid, "seq": meta.seqnum, "det": meta.det_idx,
            "clas": meta.class_name, "phase": phase,
        }))
        if self._spyglass is not None:                   # LensTasking handoff (§4.6)
            self._spyglass(frame, meta)
        self.emitted += 1
