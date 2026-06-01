"""SentinelCam OAK camera module — DepthAI Pipeline A2 (Phase 2).

Hosts the validated Pipeline A2 architecture: a single DepthAI pipeline that
terminates into four host-side output queues. The camera module owns the
pipeline; the outpost owns consumption (see OAK_OUTPOST_REDESIGN_PLAN.md
Phase 2.2 — pull interface via `OakCamera`).

Architecture (post-livetest10 consolidation):

    Camera ──► requestOutput(768×432 NV12) ──► VideoEncoder (MJPEG)  ──► host "jpeg"
       │
       ├──► requestOutput(640×360 BGR) ──► ImageManip(→300×300) ──► DetectionNetwork
       │                                                                   │
       │                                                                   ├─► host "det"
       │                                                                   ▼
       └──► requestOutput(1920×1080 NV12) ────────────────────────────► Script
                                                                            │
                                                       ┌────────────────────┼─────────┐
                                                       ▼                              ▼
                                                  crop_manip                    crop_meta Buffer
                                                       │                              │
                                                       ▼                              ▼
                                                 host "crops"                   host "meta"

Two-stage filter, per-class on the device, single dispatch off the device:

  Stage 1 (device, Script): per-class time + IoU throttle, raw-det edge guard,
  pad/aspect fix, area gate, post-pad edge guard. Whatever survives is sent —
  with its metadata sidecar — through a single ImageManip and a single output
  queue. Host demuxes by meta.label.

  Stage 2 (host, EventSampler): per-subject IoU-clustered tracks, three-phase
  position selection (entry / centre / ~75% across), seqnum-paired scene save.
  Promoted to `sentinelcam/eventsampler.py` in Phase 3.5; not in this module.

Metadata payload (26 bytes, '<I2Bf4f'):
    seqnum:uint32, det_idx:uint8, label:uint8, confidence:float32,
    xmin:float32, ymin:float32, xmax:float32, ymax:float32
  (bbox is post-pad-and-clamp, not raw det; det_idx=0xFE reserved as sentinel.)

================================ HARD-WON RULES ================================
Four rules emerged from oak_benchmark/ livetests 1–12. Reintroducing any of
them silently breaks the device pipeline. If anything tempts you to "simplify"
them away, re-read the livetest log each is anchored to.

  1. ONE ImageManip handles all crop classes, not one per class. The two-node
     split crashed the device front-end when warp operations on different
     ImageManip nodes happened back-to-back (livetest8, livetest9). One node
     serializes all warp work and eliminates that surface area. Host demuxes
     by meta.label.

  2. setMaxOutputFrameSize must match the actual OUTPUT size, not the worst-
     case input. When the warp engine misbehaves on a config it can't honour,
     it falls back to emitting the full uncropped input frame. A buffer sized
     to the input silently accepts that garbage and the device front-end
     stalls (livetest4-7). A buffer sized to the requested output rejects the
     fallback with an explicit log line and the device keeps running
     (livetest10). Applies to every ImageManip, not just crop nodes.

  3. BOTH pre-pad AND post-pad edge guards belong on the device, at matching
     per-class margins (0.05 person, 0.05 vehicle). Near-edge configs wedge
     the warp engine. Pre-pad guards reject raw det bboxes near the edge;
     post-pad guards catch geometry the pad/aspect-fix step pushes into the
     hazard zone. Earlier 0.02 / 0.001 thresholds both proved insufficient.

  4. DO NOT warm up cold ImageManip state at startup with synthetic emits.
     Two warmups 4ms apart killed the device immediately (livetest9). The
     first real detection-driven emit warms the warp engine in normal use.
===============================================================================

DepthAI v3.x. Uses Camera.build(socket) + requestOutput() pattern.
The legacy v2 `PipelineFactory` (used by outpost.py until Phase 4 integration)
is preserved at the bottom of this module.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import depthai as dai

log = logging.getLogger(__name__)


MOBILENET_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

PERSON_LABEL_IDX = MOBILENET_LABELS.index("person")        # 15
CAR_LABEL_IDX = MOBILENET_LABELS.index("car")              # 7
BUS_LABEL_IDX = MOBILENET_LABELS.index("bus")              # 6
TRAIN_LABEL_IDX = MOBILENET_LABELS.index("train")          # 19
MOTORBIKE_LABEL_IDX = MOBILENET_LABELS.index("motorbike")  # 14
BICYCLE_LABEL_IDX = MOBILENET_LABELS.index("bicycle")      # 2

VEHICLE_LABEL_INDICES = {
    CAR_LABEL_IDX, BUS_LABEL_IDX, TRAIN_LABEL_IDX,
    MOTORBIKE_LABEL_IDX, BICYCLE_LABEL_IDX,
}

# Metadata struct format — must match the device Script packer and every host
# consumer (CropMeta.unpack below, the test harness, the Phase 3.5 EventSampler).
META_STRUCT_FMT = "<I2Bf4f"
META_STRUCT_SIZE = 26  # struct.calcsize(META_STRUCT_FMT)


# ---------------------------------------------------------------------------
# Crop profiles — per-class output geometry and stage-1 throttle parameters
# ---------------------------------------------------------------------------

@dataclass
class CropProfile:
    """Per-class crop output geometry and stage-1 throttle parameters."""
    width: int
    height: int
    padding: float                  # fractional pad around detection bbox
    max_area_fraction: float        # skip if padded bbox exceeds this fraction of frame
    min_interval_s: float           # emit floor (seconds since last emit)
    min_iou: float                  # below this, treat as a new subject view
    edge_margin: float              # skip if raw det bbox is within this of any frame edge
    max_aspect_deviation: float     # reject crop if final aspect drifts > this fraction


# Validated defaults from oak_benchmark/pipeline_a2.py. edge_margin protects
# ImageManip from the asymmetric-clamp crop configs that wedge the warp engine
# (Hard-Won Rule 3); the values below are the ones that survived livetests 1–12.
DEFAULT_PERSON_PROFILE = CropProfile(
    width=256, height=384, padding=0.15,
    max_area_fraction=0.35,
    min_interval_s=0.20,            # 5 Hz floor
    min_iou=0.70,
    edge_margin=0.05,               # mandatory: edge crops are the documented crash mode
    max_aspect_deviation=0.08,      # tight; only well-shaped subjects survive
)
DEFAULT_VEHICLE_PROFILE = CropProfile(
    width=512, height=192, padding=0.10,
    max_area_fraction=0.25,
    min_interval_s=0.15,            # ~6.5 Hz; traversal self-throttles
    min_iou=0.50,
    edge_margin=0.05,               # raised from 0.02 after livetest8: a first
                                    # cold near-edge vehicle crop wedged the
                                    # warp engine; align with host edge margin.
    max_aspect_deviation=0.15,
)


def crop_profile_from_config(cfg: Optional[dict], default: CropProfile) -> CropProfile:
    """Build a CropProfile from an `oak_pipeline.crop_profiles.<class>` mapping.

    Any field absent from `cfg` falls back to the corresponding field on
    `default`, so a host_vars block need only override what differs from the
    validated defaults. Schema (OAK_OUTPOST_REDESIGN_PLAN.md Phase 2.4)::

        crop_profiles:
          person:
            width: 256
            height: 384
            padding: 0.15
            max_area_fraction: 0.35
            min_interval_s: 0.20
            min_iou: 0.70
            edge_margin: 0.05
            max_aspect_deviation: 0.08
    """
    if not cfg:
        return default
    return CropProfile(
        width=int(cfg.get("width", default.width)),
        height=int(cfg.get("height", default.height)),
        padding=float(cfg.get("padding", default.padding)),
        max_area_fraction=float(cfg.get("max_area_fraction", default.max_area_fraction)),
        min_interval_s=float(cfg.get("min_interval_s", default.min_interval_s)),
        min_iou=float(cfg.get("min_iou", default.min_iou)),
        edge_margin=float(cfg.get("edge_margin", default.edge_margin)),
        max_aspect_deviation=float(
            cfg.get("max_aspect_deviation", default.max_aspect_deviation)
        ),
    )


# ---------------------------------------------------------------------------
# Crop metadata sidecar — host-side decoder for the 26-byte wire format
# ---------------------------------------------------------------------------

DET_IDX_WARMUP_SENTINEL = 0xFE  # reserved; single-node refactor no longer emits it


@dataclass
class CropMeta:
    """Decoded crop metadata sidecar (the bbox is post-pad-and-clamp)."""
    seqnum: int
    det_idx: int
    label: int
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @classmethod
    def unpack(cls, raw: bytes) -> "CropMeta":
        s, di, lb, conf, x1, y1, x2, y2 = struct.unpack(
            META_STRUCT_FMT, raw[:META_STRUCT_SIZE],
        )
        return cls(s, di, lb, conf, x1, y1, x2, y2)

    @property
    def class_name(self) -> Optional[str]:
        return classify_detection(self.label)

    @property
    def label_name(self) -> str:
        idx = self.label
        return MOBILENET_LABELS[idx] if 0 <= idx < len(MOBILENET_LABELS) else f"idx{idx}"


def classify_detection(label_idx: int) -> Optional[str]:
    if label_idx == PERSON_LABEL_IDX:
        return "person"
    if label_idx in VEHICLE_LABEL_INDICES:
        return "vehicle"
    return None


# ---------------------------------------------------------------------------
# Device Script source
# ---------------------------------------------------------------------------

def _build_script_source(
    person_profile: CropProfile,
    vehicle_profile: CropProfile,
    person_idx: int,
    vehicle_indices: set,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> str:
    """Build the on-device Script body (runs inside the DepthAI Script node).

    Drains full_frame every loop (blocking get) to keep the 1080p ISP pool
    recycling — the discipline A1 violated and the deadlock that followed.

    Per-class stage-1 throttle uses Clock.now() seconds + IoU(last emit bbox).
    Emits a metadata Buffer for every crop on the crop_meta output, sent
    BEFORE cfg+img so the meta queue can never lag the crops queue.
    """
    return f"""
import struct

PERSON_IDX = {person_idx}
VEHICLE_INDICES = {sorted(vehicle_indices)}
PERSON_W = {person_profile.width}
PERSON_H = {person_profile.height}
VEHICLE_W = {vehicle_profile.width}
VEHICLE_H = {vehicle_profile.height}
PERSON_PAD = {person_profile.padding}
VEHICLE_PAD = {vehicle_profile.padding}
PERSON_MAX_AREA = {person_profile.max_area_fraction}
VEHICLE_MAX_AREA = {vehicle_profile.max_area_fraction}
FRAME_W = {frame_width}
FRAME_H = {frame_height}

PERSON_MIN_INTERVAL_S = {person_profile.min_interval_s}
PERSON_MIN_IOU = {person_profile.min_iou}
VEHICLE_MIN_INTERVAL_S = {vehicle_profile.min_interval_s}
VEHICLE_MIN_IOU = {vehicle_profile.min_iou}

PERSON_EDGE_MARGIN = {person_profile.edge_margin}
VEHICLE_EDGE_MARGIN = {vehicle_profile.edge_margin}
PERSON_MAX_ASPECT_DEV = {person_profile.max_aspect_deviation}
VEHICLE_MAX_ASPECT_DEV = {vehicle_profile.max_aspect_deviation}

PERSON_ASPECT = PERSON_W / PERSON_H
VEHICLE_ASPECT = VEHICLE_W / VEHICLE_H

last_emit_t = {{
    'person': -1.0e9,
    'vehicle': -1.0e9,
}}
last_emit_bbox = {{
    'person': None,
    'vehicle': None,
}}


def iou(a, b):
    if a is None or b is None:
        return 0.0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = x2 - x1
    ih = y2 - y1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def pad_and_aspect_fix(xmin, ymin, xmax, ymax, padding,
                       target_aspect, max_aspect_deviation):
    w = (xmax - xmin) * (1.0 + 2.0 * padding)
    h = (ymax - ymin) * (1.0 + 2.0 * padding)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    w_px = w * FRAME_W
    h_px = h * FRAME_H
    cur_aspect = w_px / h_px if h_px > 0 else 1.0

    if cur_aspect < target_aspect:
        w = (h_px * target_aspect) / FRAME_W
    elif cur_aspect > target_aspect:
        h = (w_px / target_aspect) / FRAME_H

    xmin = max(0.0, cx - w / 2.0)
    xmax = min(1.0, cx + w / 2.0)
    ymin = max(0.0, cy - h / 2.0)
    ymax = min(1.0, cy + h / 2.0)

    final_w_px = (xmax - xmin) * FRAME_W
    final_h_px = (ymax - ymin) * FRAME_H
    if final_h_px <= 0 or final_w_px <= 0:
        return None
    final_aspect = final_w_px / final_h_px

    if abs(final_aspect / target_aspect - 1.0) > max_aspect_deviation:
        return None

    return xmin, ymin, xmax, ymax


# --- Main loop ---
while True:
    frame = node.io['full_frame'].get()
    dets_msg = node.io['detections'].get()
    if dets_msg is None:
        continue

    frame_seq = frame.getSequenceNum()
    now_s = Clock.now().total_seconds()

    det_idx = -1
    for det in dets_msg.detections:
        det_idx += 1
        label = det.label
        is_person = (label == PERSON_IDX)
        is_vehicle = (label in VEHICLE_INDICES)
        if not (is_person or is_vehicle):
            continue

        if is_person:
            class_key = 'person'
            pad = PERSON_PAD
            aspect = PERSON_ASPECT
            out_w = PERSON_W
            out_h = PERSON_H
            max_area = PERSON_MAX_AREA
            min_interval = PERSON_MIN_INTERVAL_S
            min_iou = PERSON_MIN_IOU
            edge_margin = PERSON_EDGE_MARGIN
            max_aspect_dev = PERSON_MAX_ASPECT_DEV
        else:
            class_key = 'vehicle'
            pad = VEHICLE_PAD
            aspect = VEHICLE_ASPECT
            out_w = VEHICLE_W
            out_h = VEHICLE_H
            max_area = VEHICLE_MAX_AREA
            min_interval = VEHICLE_MIN_INTERVAL_S
            min_iou = VEHICLE_MIN_IOU
            edge_margin = VEHICLE_EDGE_MARGIN
            max_aspect_dev = VEHICLE_MAX_ASPECT_DEV

        # --- Edge-clip guard (Hard-Won Rule 3, pre-pad) ---
        # Raw det bboxes touching the frame edge produce asymmetrically
        # clamped crop regions that have been observed to wedge the warp
        # engine on persons (A2_livetest4, 2026-05-29).  Reject before any
        # geometry math so the dangerous configs never reach ImageManip.
        if (det.xmin < edge_margin or det.ymin < edge_margin or
            det.xmax > (1.0 - edge_margin) or det.ymax > (1.0 - edge_margin)):
            continue

        # --- Stage 1: time + IoU throttle ---
        det_bbox = (det.xmin, det.ymin, det.xmax, det.ymax)
        elapsed = now_s - last_emit_t[class_key]
        moved_iou = iou(det_bbox, last_emit_bbox[class_key])
        time_ok = elapsed >= min_interval
        moved_ok = moved_iou < min_iou
        if not (time_ok or moved_ok):
            continue

        # --- Geometry gates ---
        result = pad_and_aspect_fix(
            det.xmin, det.ymin, det.xmax, det.ymax, pad, aspect, max_aspect_dev
        )
        if result is None:
            continue
        xmin, ymin, xmax, ymax = result

        if (xmin < 0.0) or (xmax > 1.0) or (ymin < 0.0) or (ymax > 1.0):
            continue

        crop_area = (xmax - xmin) * (ymax - ymin)
        if crop_area > max_area:
            continue

        # --- Post-pad edge guard (Hard-Won Rule 3, post-pad) ---
        # pad_and_aspect_fix can produce padded regions that approach the
        # frame boundary even when the raw det bbox cleared the pre-pad
        # guard. A2_livetest5 showed a 0.001 epsilon ("touching the edge")
        # was insufficient; A2_livetest6 still wedged on a config whose
        # post-pad region was within 0.05 of the edge but not against it.
        # Apply the same per-class margin both pre- and post-pad so no
        # near-edge config can reach ImageManip.
        if (xmin < edge_margin or xmax > (1.0 - edge_margin) or
            ymin < edge_margin or ymax > (1.0 - edge_margin)):
            continue

        # --- Commit: update throttle, emit metadata, then crop ---
        last_emit_t[class_key] = now_s
        last_emit_bbox[class_key] = det_bbox

        meta_bytes = struct.pack(
            '{META_STRUCT_FMT}',
            frame_seq & 0xFFFFFFFF,
            det_idx & 0xFF,
            label & 0xFF,
            float(det.confidence),
            float(xmin), float(ymin), float(xmax), float(ymax),
        )
        meta = Buffer(len(meta_bytes))
        meta.setData(meta_bytes)

        x_px = int(xmin * FRAME_W)
        y_px = int(ymin * FRAME_H)
        w_px = max(1, int((xmax - xmin) * FRAME_W))
        h_px = max(1, int((ymax - ymin) * FRAME_H))
        cfg = ImageManipConfig()
        cfg.addCrop(x_px, y_px, w_px, h_px)
        cfg.setOutputSize(out_w, out_h, ImageManipConfig.ResizeMode.STRETCH)

        # Single dispatch — host demuxes by meta.label. Person emits get a
        # warn because they're rare and historically associated with warp
        # hangs; vehicle emits don't (too noisy under normal traffic).
        if is_person:
            node.warn(
                "person_emit seq={{}} det={{}} bbox=({{:.3f}},{{:.3f}},{{:.3f}},{{:.3f}}) "
                "crop_px=({{}},{{}})+{{}}x{{}} out={{}}x{{}} conf={{:.2f}}".format(
                    frame_seq, det_idx,
                    xmin, ymin, xmax, ymax,
                    x_px, y_px, w_px, h_px,
                    out_w, out_h, float(det.confidence),
                )
            )
        node.io['crop_meta'].send(meta)
        node.io['crop_img'].send(frame)
        node.io['crop_cfg'].send(cfg)
"""


# ---------------------------------------------------------------------------
# Pipeline build + pull interface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineOutputs:
    """Build-time outputs needed to start the pipeline on the host."""

    pipeline: dai.Pipeline
    jpeg: "dai.Node.Output"
    detections: "dai.Node.Output"
    crops: "dai.Node.Output"
    meta: "dai.Node.Output"

    def start(self) -> "RunningPipeline":
        q_jpeg = self.jpeg.createOutputQueue(maxSize=4, blocking=False)
        q_det = self.detections.createOutputQueue(maxSize=8, blocking=False)
        # Single mixed-class queue; sized to absorb short bursts of both
        # classes since the warp engine now serializes everything.
        q_crops = self.crops.createOutputQueue(maxSize=120, blocking=False)
        q_meta = self.meta.createOutputQueue(maxSize=120, blocking=False)
        self.pipeline.start()
        device = self.pipeline.getDefaultDevice()
        return RunningPipeline(
            pipeline=self.pipeline,
            device=device,
            q_jpeg=q_jpeg,
            q_det=q_det,
            q_crops=q_crops,
            q_meta=q_meta,
        )


@dataclass(frozen=True)
class RunningPipeline:
    """The four host-side output queues, exposed to the intake loop.

    Pull interface (OAK_OUTPOST_REDESIGN_PLAN.md Phase 2.2): the camera module
    owns the pipeline, the outpost owns the consumption. Drain order matters —
    pull `q_meta` before `q_crops` so a paired crop always finds its meta
    already queued (the Script sends meta before img+cfg; XLink preserves
    per-queue order).
    """
    pipeline: dai.Pipeline
    device: dai.Device
    q_jpeg: "dai.MessageQueue"
    q_det: "dai.MessageQueue"
    q_crops: "dai.MessageQueue"
    q_meta: "dai.MessageQueue"


def build_pipeline(
    nn_archive,
    camera_fps: int = 30,
    jpeg_quality: int = 90,
    detection_confidence: float = 0.5,
    rotate_180: bool = True,
    person_profile: CropProfile = DEFAULT_PERSON_PROFILE,
    vehicle_profile: CropProfile = DEFAULT_VEHICLE_PROFILE,
) -> PipelineOutputs:
    """Assemble Pipeline A2. Lifted verbatim from oak_benchmark/pipeline_a2.py.

    See the module docstring's HARD-WON RULES before touching the topology,
    the buffer-sizing, or the edge/aspect/area/throttle gates.
    """
    nn_archive = Path(nn_archive)
    if not nn_archive.is_file():
        raise FileNotFoundError(f"NN Archive not found: {nn_archive}")

    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    if rotate_180:
        cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

    scene_out = cam.requestOutput(
        size=(768, 432), type=dai.ImgFrame.Type.NV12, fps=camera_fps,
    )

    nn_intermediate = cam.requestOutput(
        size=(640, 360), type=dai.ImgFrame.Type.BGR888p, fps=camera_fps,
    )
    nn_scale = pipeline.create(dai.node.ImageManip)
    # Rule 2: size to the OUTPUT (300×300×3), not the 640×360 input.
    nn_scale.setMaxOutputFrameSize(300 * 300 * 3)
    nn_scale.initialConfig.setOutputSize(
        300, 300, dai.ImageManipConfig.ResizeMode.STRETCH,
    )
    nn_intermediate.link(nn_scale.inputImage)

    full_out = cam.requestOutput(
        size=(1920, 1080), type=dai.ImgFrame.Type.NV12, fps=camera_fps,
    )

    jpeg_encoder = pipeline.create(dai.node.VideoEncoder)
    jpeg_encoder.setDefaultProfilePreset(
        camera_fps, dai.VideoEncoderProperties.Profile.MJPEG,
    )
    jpeg_encoder.setQuality(jpeg_quality)
    scene_out.link(jpeg_encoder.input)

    archive = dai.NNArchive(nn_archive)
    nn = pipeline.create(dai.node.DetectionNetwork)
    nn.setNNArchive(archive, numShaves=6)
    nn.setConfidenceThreshold(detection_confidence)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    nn_scale.out.link(nn.input)

    script = pipeline.create(dai.node.Script)
    script.setScript(
        _build_script_source(
            person_profile=person_profile,
            vehicle_profile=vehicle_profile,
            person_idx=PERSON_LABEL_IDX,
            vehicle_indices=VEHICLE_LABEL_INDICES,
            frame_width=1920,
            frame_height=1080,
        )
    )
    nn.out.link(script.inputs["detections"])
    full_out.link(script.inputs["full_frame"])
    script.inputs["detections"].setBlocking(False)
    script.inputs["full_frame"].setBlocking(False)

    # Rule 1: SINGLE ImageManip for all crops, regardless of class. Output
    # dimensions are set per-config via setOutputSize, so person (256×384) and
    # vehicle (512×192) requests both flow through the same warp context. The
    # two-node split was retired after A2_livetest8/9 showed that back-to-back
    # warp operations on different ImageManip nodes can corrupt shared device
    # state; serializing everything through one node eliminates that surface
    # area. Host demuxes by meta.label.
    #
    # Rule 2: size to the largest requested OUTPUT, not the 1080p input. A
    # buffer sized to the input would silently accept the warp engine's
    # full-frame fallback and stall the device; sizing to the output rejects
    # the fallback with a log line and keeps the device running.
    max_crop_bytes = max(
        person_profile.width * person_profile.height,
        vehicle_profile.width * vehicle_profile.height,
    ) * 3
    crop_manip = pipeline.create(dai.node.ImageManip)
    crop_manip.setMaxOutputFrameSize(max_crop_bytes)
    crop_manip.inputImage.setBlocking(False)
    crop_manip.inputConfig.setBlocking(False)
    script.outputs["crop_cfg"].link(crop_manip.inputConfig)
    script.outputs["crop_img"].link(crop_manip.inputImage)

    return PipelineOutputs(
        pipeline=pipeline,
        jpeg=jpeg_encoder.bitstream,
        detections=nn.out,
        crops=crop_manip.out,
        meta=script.outputs["crop_meta"],
    )


# ---------------------------------------------------------------------------
# OakCamera — high-level façade the outpost owns (Phase 2.2 pull interface)
# ---------------------------------------------------------------------------

class OakCamera:
    """Owns the A2 DepthAI pipeline lifecycle; exposes the four output queues.

    Reads the `oak_pipeline` YAML config block (host_vars per OAK outpost) and
    builds the validated Pipeline A2. The outpost constructs one of these,
    calls `start()`, and then drains `q_jpeg` / `q_det` / `q_meta` / `q_crops`
    from its intake loop (Phase 3). Drain `q_meta` before `q_crops`.

    Config schema (OAK_OUTPOST_REDESIGN_PLAN.md Phase 2.4)::

        oak_pipeline:
          jpeg_quality: 90
          detection_confidence: 0.5
          rotate_180: true
          camera_fps: 30
          crop_profiles:
            person: { width: 256, height: 384, ... }
            vehicle: { width: 512, height: 192, ... }
    """

    def __init__(self, nn_archive, config: Optional[dict] = None) -> None:
        config = config or {}
        crop_cfg = config.get("crop_profiles", {}) or {}
        self.person_profile = crop_profile_from_config(
            crop_cfg.get("person"), DEFAULT_PERSON_PROFILE
        )
        self.vehicle_profile = crop_profile_from_config(
            crop_cfg.get("vehicle"), DEFAULT_VEHICLE_PROFILE
        )
        log.debug(
            "Building OAK Pipeline A2 (person=%dx%d, vehicle=%dx%d)",
            self.person_profile.width, self.person_profile.height,
            self.vehicle_profile.width, self.vehicle_profile.height,
        )
        self._outputs = build_pipeline(
            nn_archive,
            camera_fps=int(config.get("camera_fps", 30)),
            jpeg_quality=int(config.get("jpeg_quality", 90)),
            detection_confidence=float(config.get("detection_confidence", 0.5)),
            rotate_180=bool(config.get("rotate_180", True)),
            person_profile=self.person_profile,
            vehicle_profile=self.vehicle_profile,
        )
        self._running: Optional[RunningPipeline] = None

    def start(self) -> RunningPipeline:
        """Start the device pipeline and return the running output queues."""
        if self._running is None:
            self._running = self._outputs.start()
        return self._running

    @property
    def running(self) -> Optional[RunningPipeline]:
        return self._running

    @property
    def q_jpeg(self):
        return self._running.q_jpeg

    @property
    def q_det(self):
        return self._running.q_det

    @property
    def q_crops(self):
        return self._running.q_crops

    @property
    def q_meta(self):
        return self._running.q_meta

    def close(self) -> None:
        if self._running is not None:
            self._outputs.pipeline.stop()
            self._running = None


# ===========================================================================
# LEGACY — DepthAI v2 pipeline. Still imported by outpost.py:413 (setup_OAK)
# until Phase 4 integration replaces the OAK intake path. Do not extend; the
# A2 architecture above supersedes it. Removed when Phase 4 lands.
# ===========================================================================

class PipelineFactory:

    def MobileNetSSD(self, nn_path=None):
        NN_SIZE = (300,300)
        NN_PATH = nn_path

        # Create pipeline
        pipeline = dai.Pipeline()

        # Define nodes and outputs
        nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
        cam = pipeline.create(dai.node.ColorCamera)
        encoder = pipeline.create(dai.node.VideoEncoder)

        xoutFrames = pipeline.create(dai.node.XLinkOut)
        xoutJPEG = pipeline.create(dai.node.XLinkOut)
        xoutNN = pipeline.create(dai.node.XLinkOut)

        xoutFrames.setStreamName("frames")
        xoutJPEG.setStreamName("jpegs")
        xoutNN.setStreamName("nn")

        # Properties
        nn.setConfidenceThreshold(0.5)
        nn.setBlobPath(NN_PATH)

        cam.setPreviewSize(NN_SIZE)
        cam.setPreviewKeepAspectRatio(False)
        cam.setInterleaved(False)
        cam.setIspScale(1,3)
        cam.setFps(30)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        # scale collection down from 4K to just FullHD
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setVideoSize(640,360) # reduce further for storage
        # For OAK-1 camera, when USB cable pointed down
        cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

        encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)

        # Linking
        cam.video.link(encoder.input)
        cam.video.link(xoutFrames.input)
        encoder.bitstream.link(xoutJPEG.input)
        cam.preview.link(nn.input)
        nn.out.link(xoutNN.input)

        # Connect to device and start pipeline
        device = dai.Device(pipeline)
        return device

    def __init__(self, pipeline, nn_path=None) -> None:
        logging.debug(f"Starting DepthAI pipeline '{pipeline}'")
        PipeLines = {
            'MobileNetSSD' : self.MobileNetSSD
        }
        self.device = PipeLines[pipeline](nn_path)
