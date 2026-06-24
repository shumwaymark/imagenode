"""HostTracker — persistent host-side multi-subject tracker (production promotion).

Promotes the replay-validated `twister/tracker_replay.py:Tracker` into a standalone,
config-driven, dependency-light component. Spec: `host_tracker_design.md`. Intended
production home: `sentinelcam/hosttracker.py` (shared library — the outpost drives it
live, the Sentinel can drive it over replay; Open Decision 5).

Design contract (host_tracker §2): a pure track-state maintainer over a stream of
timestamped detections. It answers exactly one question — "what subjects are in the
scene right now, where, and is each moving?" — and exposes the answer as `(tid,
from_state, to_state)` transitions plus the live `tracks` (with `history`). It does
NOT run inference, decide events, select crops, or touch camera state. It persists
across event boundaries — never reset.

`observe()` is ported VERBATIM from the validated replay (only field renames + the
module constants lifted to `TrackerConfig`, defaulting to the validated values), so
`dev/test_hosttracker_replay.py` can prove behavior equivalence against the original
over the captured `_dets.jsonl` corpus. `tick()` is NEW (the picamera path, not
exercised by the OAK replay) — see its docstring.

No numpy / no depthai: this runs on the Pi outpost and must stay cheap and portable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

PROVISIONAL = "PROVISIONAL"
ACTIVE = "ACTIVE"
QUIESCENT = "QUIESCENT"
END = "END"

Transition = tuple[int, str, str]            # (tid, from_state, to_state)


# --------------------------------------------------------------------------- #
# Tunables (host_tracker §8) — per-camera, rendered from host_vars by Ansible.  #
# Defaults are the replay-validated street-cam values.                          #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrackerConfig:
    interesting_classes: frozenset = frozenset({"person", "vehicle"})
    match_iou: float = 0.30          # detection<->track association floor
    min_confidence_new: float = 0.0  # admission floor: a detection below this may still
                                     #    associate/sustain an existing track, but must NOT
                                     #    BIRTH a new one (ByteTrack high/low split, §12.1).
                                     #    Kills low-confidence static-blob storms at the birth
                                     #    site. 0.0 = replay-identical (no gating).
    confirm_obs: int = 2             # min sightings to confirm PROVISIONAL->ACTIVE
    confirm_time: float = 0.05       # AND min capture-time span (s)
    quiescence_window: float = 1.2   # s of no relocation -> QUIESCENT
    quiescence_iou: float = 0.95     # window-endpoint IoU above this == "hasn't moved"
    relocation_iou: float = 0.50     # banked-vs-now IoU below this == "relocated" -> re-open
    gap_misses: Optional[int] = 30   # K: max consecutive observe()-misses before END
                                     #    (None = never end; analysis pass A only)
    quiescent_gap_misses: Optional[int] = None
                                     # coast K for QUIESCENT (banked) tracks. A parked subject
                                     #    survives this many consecutive misses before END, so a
                                     #    brief detection dropout (glare/focus blink) re-associates
                                     #    at its banked bbox instead of ending + re-opening a fresh
                                     #    event (§6/§12.1). Set well above gap_misses. None = fall
                                     #    back to gap_misses (replay-identical).
    history_len: int = 256           # bounded per-track (ts, bbox) history depth
    # §12.3 open item — centroid-distance association fallback for the IoU=0 tail
    # (fastest movers / re-detect jumps). None = DISABLED = replay-identical default.
    centroid_match_dist: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "TrackerConfig":
        """Build from a host_vars `tracker:` block. Unknown keys are ignored;
        `interesting_classes` may be given as a list."""
        if not d:
            return cls()
        known = {f.name for f in fields_of(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        if "interesting_classes" in kw and kw["interesting_classes"] is not None:
            kw["interesting_classes"] = frozenset(kw["interesting_classes"])
        return cls(**kw)


def fields_of(dc):
    from dataclasses import fields as _fields
    return _fields(dc)


# --------------------------------------------------------------------------- #

def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _centroid_dist(a, b) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


class Track:
    """One subject's persistent state. Public fields per host_tracker §4. Phase
    (entry/centre/far) deliberately does NOT live here — the EventSampler reads
    `history` and computes it (keeps phase strategies swappable, §3.5.2)."""

    __slots__ = ("tid", "baseclass", "bbox", "first_seen", "last_observed",
                 "state", "obs", "misses", "banked_bbox", "history",
                 "went_active", "q_enter", "event_id", "classname")

    def __init__(self, tid: int, baseclass: str, bbox: tuple,
                 capture_time: float, history_len: int,
                 classname: Optional[str] = None) -> None:
        self.tid = tid
        self.baseclass = baseclass
        # display/report label for the latest sighting (e.g. "car: 0.9600"); the
        # tracker uses `baseclass` for identity, but downstream trk records and
        # overlays want the specific class + confidence. Defaults to baseclass.
        self.classname = classname if classname is not None else baseclass
        self.bbox = bbox
        self.first_seen = capture_time
        self.last_observed = capture_time           # last authoritative sighting (observe-match)
        self.state = PROVISIONAL
        self.obs = 1
        self.misses = 0
        self.banked_bbox: Optional[tuple] = None    # bbox at QUIESCENT entry (relocation ref)
        self.history = deque([(capture_time, bbox)], maxlen=history_len)
        self.went_active = False
        self.q_enter: Optional[float] = None        # ts entered QUIESCENT (banked-duration acct)
        self.event_id: Optional[str] = None         # stamped by the event manager; tracker ignores


class HostTracker:
    """Persistent multi-subject tracker. `observe()` per detection frame (the NN
    ran); `tick()` per detection-less frame (picamera quiet frames only)."""

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self.cfg = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_tid = 0
        # telemetry / analysis
        self.recovered_gaps: list[int] = []         # miss-runs later re-matched (pass A)
        self.quiescent_durs: list[float] = []       # banked-duration per QUIESCENT spell (s)

    # -- public reads (the EventSampler / event manager consume these) ------ #

    def get_track(self, tid: int) -> Optional[Track]:
        for t in self.tracks:
            if t.tid == tid:
                return t
        return None

    def has_active(self) -> bool:
        return any(t.state == ACTIVE for t in self.tracks)

    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.state == ACTIVE]

    def scene_state(self) -> list[Track]:
        """Quiescent tracks ARE the scene state (host_tracker §9)."""
        return [t for t in self.tracks if t.state == QUIESCENT]

    # -- internals ---------------------------------------------------------- #

    def _new(self, baseclass: str, bbox: tuple, capture_time: float,
             classname: Optional[str] = None) -> Track:
        t = Track(self._next_tid, baseclass, bbox, capture_time,
                  self.cfg.history_len, classname)
        self._next_tid += 1
        self.tracks.append(t)
        return t

    def _leave_quiescent(self, t: Track, ts: float) -> None:
        if t.q_enter is not None:
            self.quiescent_durs.append(ts - t.q_enter)
            t.q_enter = None

    # -- the authoritative path (ported verbatim from the validated replay) -- #

    def observe(self, capture_time: float, detections) -> list[Transition]:
        """The NN ran. `detections` is an iterable of (baseclass, bbox[, label[, conf]]):
        bbox normalized (x1,y1,x2,y2); optional `label` is the display string
        ("car: 0.96"); optional `conf` is the numeric detection confidence consulted by
        the admission floor (`min_confidence_new`). When `conf` is absent it defaults to
        1.0 (unknown == unfiltered), keeping the 2-/3-tuple replay corpus identical.
        Returns the transitions this frame produced. Interesting-class filtering is
        applied here (host_tracker §3)."""
        cfg = self.cfg
        ts = capture_time
        # dets stays a 2-tuple list so the validated association logic is unchanged;
        # `labels` is a parallel list (same index) carrying the optional display
        # label (3rd detection element, e.g. "car: 0.96"), default = baseclass.
        dets = []
        labels = []
        confs = []
        for _d in detections:
            c, b = _d[0], _d[1]
            if c not in cfg.interesting_classes:
                continue
            dets.append((c, tuple(b)))
            labels.append(_d[2] if len(_d) > 2 else c)
            confs.append(_d[3] if len(_d) > 3 else 1.0)

        # greedy IoU association within class
        pairs = []
        for di, (cls, bb) in enumerate(dets):
            for t in self.tracks:
                if t.baseclass == cls:
                    v = iou(t.bbox, bb)
                    if v >= cfg.match_iou:
                        pairs.append((v, di, t))
        pairs.sort(reverse=True, key=lambda p: p[0])
        used_d: set[int] = set()
        used_t: set[int] = set()
        for v, di, t in pairs:
            if di in used_d or t.tid in used_t:
                continue
            used_d.add(di)
            used_t.add(t.tid)
            cls, bb = dets[di]
            if t.misses:                            # was missed, now re-matched -> recovered gap
                self.recovered_gaps.append(t.misses)
            t.bbox = bb
            t.last_observed = ts
            t.obs += 1
            t.misses = 0
            t.history.append((ts, bb))
            t.classname = labels[di]

        # §12.3 optional centroid fallback for the IoU=0 tail (disabled by default).
        if cfg.centroid_match_dist is not None:
            for di, (cls, bb) in enumerate(dets):
                if di in used_d:
                    continue
                best = None
                best_d = cfg.centroid_match_dist
                for t in self.tracks:
                    if t.tid in used_t or t.baseclass != cls:
                        continue
                    d = _centroid_dist(t.bbox, bb)
                    if d <= best_d:
                        best_d = d
                        best = t
                if best is not None:
                    used_d.add(di)
                    used_t.add(best.tid)
                    if best.misses:
                        self.recovered_gaps.append(best.misses)
                    best.bbox = bb
                    best.last_observed = ts
                    best.obs += 1
                    best.misses = 0
                    best.history.append((ts, bb))
                    best.classname = labels[di]

        transitions: list[Transition] = []

        # unmatched dets -> new provisional tracks, gated by admission confidence.
        # A sub-floor detection already had its chance to associate above (sustain);
        # it must not BIRTH a track (ByteTrack high/low split, §12.1).
        for di, (cls, bb) in enumerate(dets):
            if di not in used_d and confs[di] >= cfg.min_confidence_new:
                self._new(cls, bb, ts, labels[di])

        # unmatched tracks accrue misses; END past K. QUIESCENT (banked) tracks coast
        # on a larger K (quiescent_gap_misses) so a brief dropout — glare/focus blink —
        # doesn't END + re-open a still-present parked subject (§6/§12.1). gap_misses
        # is None still means "never end" for every state (analysis pass A).
        survivors = []
        for t in self.tracks:
            if t.tid not in used_t:
                t.misses += 1
                if cfg.gap_misses is not None:
                    K = cfg.gap_misses
                    if t.state == QUIESCENT and cfg.quiescent_gap_misses is not None:
                        K = cfg.quiescent_gap_misses
                    if t.misses > K:
                        if t.state != PROVISIONAL:
                            transitions.append((t.tid, t.state, END))
                        self._leave_quiescent(t, t.last_observed)
                        continue                    # drop
            survivors.append(t)
        self.tracks = survivors

        # state transitions
        for t in self.tracks:
            if t.state == PROVISIONAL:
                if t.obs >= cfg.confirm_obs and (ts - t.first_seen) >= cfg.confirm_time:
                    t.state = ACTIVE
                    t.went_active = True
                    transitions.append((t.tid, PROVISIONAL, ACTIVE))
            elif t.state == ACTIVE:
                # window-endpoint relocation: IoU(now, earliest bbox within window)
                cutoff = ts - cfg.quiescence_window
                anchor = None
                for (hts, hbb) in t.history:
                    if hts >= cutoff:
                        anchor = hbb
                        break
                if anchor is not None and (ts - t.first_seen) >= cfg.quiescence_window \
                        and iou(t.bbox, anchor) >= cfg.quiescence_iou:
                    t.state = QUIESCENT
                    t.banked_bbox = t.bbox
                    t.q_enter = ts
                    transitions.append((t.tid, ACTIVE, QUIESCENT))
            elif t.state == QUIESCENT:
                # relocation, not raw Δ, gates the re-open (host_tracker §5)
                if iou(t.bbox, t.banked_bbox) < cfg.relocation_iou:
                    t.state = ACTIVE
                    self._leave_quiescent(t, ts)
                    transitions.append((t.tid, QUIESCENT, ACTIVE))
        return transitions

    # -- the picamera-only path (NEW; not replay-validated) ----------------- #

    def tick(self, capture_time: float) -> list[Transition]:
        """Clock advance only — no detection evidence (host_tracker §3). The NN
        did NOT run this frame (picamera, motion-quiet). NEVER accrues misses and
        NEVER ends a track: absence is known only through observe() (a subject
        cannot leave without motion → motion → an observe() that omits it, §6).

        The one time-based effect: a still-present subject whose motion has
        stopped banks to QUIESCENT once it has gone unobserved for
        `quiescence_window` — the picamera quiescence path (§6).

        NOT exercised by the OAK replay corpus (OAK runs the NN every frame and
        only ever calls observe()). First cut — validate on the picamera
        fast-follow (§4.10) before relying on it. OAK never calls tick()."""
        cfg = self.cfg
        transitions: list[Transition] = []
        for t in self.tracks:
            if t.state == ACTIVE and (capture_time - t.last_observed) >= cfg.quiescence_window:
                t.state = QUIESCENT
                t.banked_bbox = t.bbox
                t.q_enter = capture_time
                transitions.append((t.tid, ACTIVE, QUIESCENT))
        return transitions

    def finalize(self, capture_time: float) -> None:
        """Close out banked-duration accounting at end of stream/run."""
        for t in self.tracks:
            self._leave_quiescent(t, capture_time)
