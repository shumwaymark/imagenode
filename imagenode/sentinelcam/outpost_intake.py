"""Phase 3 outpost event-plane drain thread.

Maps to OAK_OUTPOST_REDESIGN_PLAN.md §3.1–3.6 and host_tracker_design.md. Built
ALONGSIDE the legacy ``setup_OAK`` path (replaced in Phase 4), not through it.
Per §4.1 the imagenode detector callback becomes a thin KEEPALIVE; the threads
here own the OAK device queues and run free at the device cadence.

Structure:
  CONCRETE here  the two-plane split, the drain loop + ordering, crop/meta pairing,
                 the device-frame crop lookback (§3.3, probe-confirmed safe to hold),
                 and the consumer wiring.
  INJECTED       Tracker / EventManager / EventSampler are constructed by the
                 assembler (outpost.py) and passed in — their implementations are
                 sibling modules:
                   - sentinelcam.hosttracker.HostTracker   (host_tracker_design.md)
                   - sentinelcam.eventmanager.EventManager (host_tracker_design.md §7)
                   - sentinelcam.eventsampler.EventSampler (§3.5)
                 The Protocols below document the contract each must satisfy; the
                 device types are Protocols too, so this module stays depthai-free
                 and unit-testable.

Why threads, not processes (vs. CLAUDE.md anti-pattern #1): the heavy image work
(crop inference) is offloaded to the SpyGlass child PROCESS via the existing
LensTasking handoff. What runs here is light orchestration (pairing, tracker
kinematics, dispatch) plus I/O (scene publish). The scene publisher's blocking
``send`` releases the GIL, so a sibling thread decouples it from the drain
without a process. No frame-rate CPU work happens under the GIL on this path.

Init sequence (§3.4, adapted — no new shared-memory ring, so no fork/handshake):
  1. OakCamera.start() -> RunningPipeline (owns the device + 4 queues)
  2. construct Tracker, EventManager, EventSampler, ScenePublisher
  3. SpyGlass LensTasking fork/handshake (existing mechanism, repurposed payload)
  4. start ScenePublisher thread (Plane 1) and OutpostIntake thread (Plane 2)
  5. imagenode keeps calling the detector with None -> keepalive only
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, cast

from sentinelcam.pairing import CropPairer, MetaLike, PairMode

if TYPE_CHECKING:
    from sentinelcam.eventsampler import CropMetaLike


# --------------------------------------------------------------------------- #
# Structural types for the device-side objects (so this file needs no depthai). #
# --------------------------------------------------------------------------- #

class _Msg(Protocol):
    def getSequenceNum(self) -> int: ...
    def getData(self) -> Any: ...                # numpy buffer (host-owned)


class _ImgFrame(_Msg, Protocol):
    def getWidth(self) -> int: ...
    def getHeight(self) -> int: ...


class _DetMsg(_Msg, Protocol):
    detections: list                             # each: .label .confidence .xmin .ymin .xmax .ymax
    def getTimestamp(self) -> Any: ...           # timedelta — device capture-time


class _Queue(Protocol):
    def tryGet(self) -> Any: ...                  # -> message | None (non-blocking)


class RunningPipelineLike(Protocol):
    """The four host-side output queues from OakCamera.start() (Phase 2.2).

    Read-only properties (not bare attributes) so the members are covariant: a
    concrete RunningPipeline whose queues are depthai MessageQueues (a structural
    subtype of _Queue) satisfies the protocol. Mutable attributes would be
    invariant and reject the subtype.
    """
    @property
    def q_jpeg(self) -> _Queue: ...
    @property
    def q_det(self) -> _Queue: ...
    @property
    def q_crops(self) -> _Queue: ...
    @property
    def q_meta(self) -> _Queue: ...


# --------------------------------------------------------------------------- #
# Consumer seams — bodies specified/validated elsewhere, Protocols here.        #
# --------------------------------------------------------------------------- #

Transition = tuple[int, str, str]                # (tid, from_state, to_state)


# A detection fed to the tracker: (baseclass, bbox), bbox normalized (x1,y1,x2,y2).
# MUST be a 2-tuple — HostTracker.observe() unpacks each as `(c, b)`. A dataclass
# is not unpackable and crashes observe() on the first non-empty frame (the silent
# drain-thread death that looked like "live feed but no events", fixed 2026-06-08).
Detection = tuple[str, tuple]


class Tracker(Protocol):
    """Persistent host-side tracker (host_tracker_design.md). Source-agnostic:
    OAK only ever calls observe() (NN every frame); the picamera path also uses
    tick() on motion-quiet frames. Promote twister/tracker_replay.py."""
    def observe(self, capture_time: float, detections: list[Detection]) -> list[Transition]: ...
    def tick(self, capture_time: float) -> list[Transition]: ...


class EventManager(Protocol):
    """Reacts to tracker transitions; emits ote start/trk/end (host_tracker §7).
    Called every observe (not just on transitions): it emits a trk per ACTIVE
    track while an event is open, so it needs the tracker each frame. Owns the
    current open event_id; the EventSampler queries it to tag crops."""
    def process(self, capture_time: float, transitions: list[Transition], tracker: "Tracker") -> None: ...
    @property
    def event_id(self) -> Optional[str]: ...


class EventSampler(Protocol):
    """Stage-2 host selection (§3.5). Holds refs to the crop lookback, the tracker
    (for per-track history/tid), the event manager (for event_id), and the crop
    publisher + SpyGlass LensTasking handoff. ``note`` is called once per paired
    crop; the sampler decides when a phase fires, pulls the frame from the
    lookback by key, encodes the selected ~3/traversal, publishes + hands to
    SpyGlass. Promote EventSamplerCapture from harness_a2.py."""
    def note(self, crop_key: int, meta: "CropMetaLike") -> None: ...
    def flush_ready(self) -> None: ...               # once per tick: flush retroactive entries


# Positional callable: (seqnum, device_capture_seconds, jpeg_bytes) -> None.
ScenePublishFn = Callable[[int, float, bytes], None]


# --------------------------------------------------------------------------- #
# Crop lookback — bounded deque of DEVICE frames (§3.3).                        #
# --------------------------------------------------------------------------- #

class CropLookback:
    """Recent paired crops, keyed by correlation key, holding the device ImgFrame
    directly. The poolhold probe (2026-06-06) confirmed holding pulled crop frames
    does not pin a device pool, so we defer the NV12->BGR->JPEG copy to the ~3
    crops the EventSampler actually selects — cheaper than copy-on-drain.

    Bounded by count: a few seconds of stage-1-gated crops is ample for the
    entry-phase lookback that survives the confirmation debounce (host_tracker §7).
    Oldest entries evict (and their device frames release) automatically.
    """

    def __init__(self, maxlen: int = 90) -> None:
        self._maxlen = maxlen
        self._by_key: "OrderedDict[int, tuple[_ImgFrame, MetaLike]]" = OrderedDict()

    def add(self, key: int, frame: _ImgFrame, meta: MetaLike) -> None:
        self._by_key[key] = (frame, meta)
        while len(self._by_key) > self._maxlen:
            self._by_key.popitem(last=False)     # evict oldest; releases the device frame

    def get(self, key: int) -> Optional[tuple[_ImgFrame, MetaLike]]:
        return self._by_key.get(key)

    def __len__(self) -> int:
        return len(self._by_key)


# --------------------------------------------------------------------------- #
# Plane 1 — scene publisher (always on, 24×7, event-independent).               #
# --------------------------------------------------------------------------- #

class ScenePublisher(threading.Thread):
    """Drains q_jpeg and publishes device-MJPEG bytes to the scene ImageZMQ PUB.
    Always on — runs in the dark for the Watchtower live feed — and isolated from
    the event plane so a Plane-2 stall never blackouts the live view. I/O-bound
    ``send`` releases the GIL, decoupling it from the drain thread (§4.2)."""

    def __init__(self, q_jpeg: _Queue, publish: ScenePublishFn, idle_sleep: float = 0.002) -> None:
        super().__init__(name="ScenePublisher", daemon=True)
        self._q = q_jpeg
        self._publish = publish
        self._idle = idle_sleep
        self._stop = threading.Event()
        self.published = 0

    def run(self) -> None:
        last_err_log = 0.0
        while not self._stop.is_set():
            try:
                msg = self._q.tryGet()
                if msg is None:
                    time.sleep(self._idle)
                    continue
                # pass the device capture-time so the scene timestamp matches the
                # trk/crp timestamp for the same frame (shared device->wall clock).
                dev_s = msg.getTimestamp().total_seconds()
                self._publish(msg.getSequenceNum(), dev_s, bytes(msg.getData()))
                self.published += 1
            except Exception:
                now = time.monotonic()           # never die silently (see OutpostIntake.run)
                if now - last_err_log >= 30.0:
                    logging.exception("ScenePublisher error")
                    last_err_log = now
                time.sleep(0.05)

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------- #
# Plane 2 — event-machinery drain thread.                                       #
# --------------------------------------------------------------------------- #

class OutpostIntake(threading.Thread):
    """Single drain thread (§3.1). Owns q_det / q_meta / q_crops; never blocks on
    a consumer. Per tick, drains in dependency order:

        det   -> decode -> tracker.observe() -> transitions -> event manager
        meta  -> pairer.push_meta()                 (meta BEFORE crops: the device
        crops -> pair by key -> lookback + sampler   sends meta first, XLink keeps
                                                     per-queue order)

    On OAK the NN runs every frame, so every det message is an observe(); tick()
    is unused here (it is the picamera path). q_jpeg is NOT drained here — that is
    Plane 1's ScenePublisher (§4.2).
    """

    def __init__(
        self,
        running: RunningPipelineLike,
        tracker: Tracker,
        event_manager: EventManager,
        event_sampler: EventSampler,
        lookback: CropLookback,
        classify: Callable[[int], Optional[str]],
        decode_meta: Callable[[bytes], MetaLike],
        # label idx -> specific class name (e.g. 7 -> "car"); injected so this
        # module stays depthai-free. When None, trk reports the baseclass only.
        label_name: Optional[Callable[[int], str]] = None,
        # SEQNUM matches what the device emits today; COMPOSITE is the deferred
        # defense-in-depth upgrade and additionally needs the device Script to
        # stamp seqnum*STRIDE+det_idx onto each crop (§3.2, never-observed edge).
        pair_mode: PairMode = PairMode.SEQNUM,
        idle_sleep: float = 0.002,
    ) -> None:
        super().__init__(name="OutpostIntake", daemon=True)
        self._running = running
        self._tracker = tracker
        self._events = event_manager
        self._sampler = event_sampler
        self._lookback = lookback
        self._classify = classify
        self._label_name = label_name
        self._decode_meta = decode_meta
        self._pairer = CropPairer(mode=pair_mode)
        self._idle = idle_sleep
        self._stop = threading.Event()
        # keepalive telemetry — surfaced to the imagenode detector callback (§4.1)
        self.dets_seen = 0
        self.crops_paired = 0

    # -- the loop --------------------------------------------------------- #

    def run(self) -> None:
        last_err_log = 0.0
        while not self._stop.is_set():
            try:
                did_work = self._drain_once()
            except Exception:
                # Never let the drain thread die silently — that failure mode
                # masquerades as "live feed but no events" (scene publisher is a
                # separate thread). Log the traceback (throttled so a persistent
                # fault can't flood the log) and keep draining so a transient
                # glitch self-recovers.
                now = time.monotonic()
                if now - last_err_log >= 30.0:
                    logging.exception("OutpostIntake drain error")
                    last_err_log = now
                time.sleep(0.05)
                continue
            if not did_work:
                time.sleep(self._idle)          # nothing queued — yield briefly

    def _drain_once(self) -> bool:
        worked = False
        r = self._running

        # det -> tracker -> event manager
        while True:
            d: Optional[_DetMsg] = r.q_det.tryGet()
            if d is None:
                break
            worked = True
            self.dets_seen += 1
            ct = _capture_time(d)
            dets = self._decode_detections(d)
            transitions = self._tracker.observe(ct, dets)
            # called every observe: emits trk for ACTIVE tracks + opens/closes events
            self._events.process(ct, transitions, self._tracker)

        # meta FIRST (so a paired crop finds its meta already queued)
        while True:
            m: Optional[_Msg] = r.q_meta.tryGet()
            if m is None:
                break
            worked = True
            self._pairer.push_meta(self._decode_meta(bytes(m.getData())))

        # crops -> pair -> lookback + sampler
        while True:
            frame: Optional[_ImgFrame] = r.q_crops.tryGet()
            if frame is None:
                break
            worked = True
            key = frame.getSequenceNum()
            meta = self._pairer.pair(key)
            if meta is None:
                continue                         # orphan/desync already counted in pairer.stats
            self._lookback.add(key, frame, meta)
            # On OAK the meta is a CropMeta (carries bbox), which the pairer types
            # only as MetaLike; the EventSampler needs the richer CropMetaLike.
            self._sampler.note(key, cast("CropMetaLike", meta))
            self.crops_paired += 1

        # flush retroactive entry crops now that this tick's dets opened any events
        self._sampler.flush_ready()
        return worked

    # -- helpers ---------------------------------------------------------- #

    def _decode_detections(self, d: _DetMsg) -> list[Detection]:
        out: list[Detection] = []
        for x in d.detections:
            base = self._classify(int(x.label))
            if base is None:                     # not an interesting class — drop at ingest
                continue
            bbox = (float(x.xmin), float(x.ymin), float(x.xmax), float(x.ymax))
            # 3rd element: specific label + confidence ("car: 0.9600"), legacy
            # format that downstream (VehicleSpeed, overlays) parse. The tracker
            # uses base for identity and carries this through to the trk record.
            if self._label_name is not None:
                label = f"{self._label_name(int(x.label))}: {float(x.confidence):.4f}"
                out.append((base, bbox, label))
            else:
                out.append((base, bbox))         # baseclass-only (picamera default)
        return out

    def stats(self) -> dict:
        """Keepalive snapshot for the imagenode detector callback (§4.1)."""
        s = self._pairer.stats
        return {
            "dets_seen": self.dets_seen,
            "crops_paired": self.crops_paired,
            "orphaned_meta": s.orphaned_meta,
            "crop_no_meta": s.crop_no_meta,
            "lookback_depth": len(self._lookback),
        }

    def stop(self) -> None:
        self._stop.set()
        self._pairer.flush_orphans()


def _capture_time(msg: _DetMsg) -> float:
    """Device capture-time in seconds (anti-pattern #5: never datetime.now())."""
    return msg.getTimestamp().total_seconds()


# --------------------------------------------------------------------------- #
# Assembly sketch (production wiring lives in outpost.py / setup_OAK's heir).   #
# --------------------------------------------------------------------------- #
#
#   cam   = OakCamera(nn_archive, cfg["oak_pipeline"])
#   run   = cam.start()                              # RunningPipeline (4 queues)
#
#   tracker  = HostTracker(**cfg["tracker"])         # promote tracker_replay.Tracker
#   events   = EventManagerImpl(log_pub, ...)        # emits ote start/trk/end
#   lookback = CropLookback(maxlen=cfg.get("lookback_frames", 90))
#   sampler  = EventSamplerImpl(lookback, tracker, events, crop_pub, spyglass,
#                               strategy=phase_strategy_for(cfg))   # §3.5
#
#   scene = ScenePublisher(run.q_jpeg, scene_pub.send)              # Plane 1
#   intake = OutpostIntake(run, tracker, events, sampler, lookback,
#                          classify=oak.classify_detection,
#                          decode_meta=oak.CropMeta.unpack)         # Plane 2
#   scene.start(); intake.start()
#
#   # imagenode detector callback (§4.1) -> keepalive only:
#   def detect(image=None):
#       return heartbeat_from(intake.stats())   # no image work on the callback
