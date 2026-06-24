"""outpost: sentinelcam integration with imagenode
Support for image publishing and outpost functionality

Copyright (c) 2021 by Mark K Shumway, mark.shumway@swanriver.dev
License: MIT, see the sentinelcam LICENSE for more details.
"""

import cv2
import logging
import logging.config
import json
import socket
import threading
import time
import zmq
import imagezmq
import numpy as np
import simplejpeg
from ast import literal_eval
from datetime import datetime
from sentinelcam.utils import FPS
from sentinelcam.spyglass import SpyGlass, LensTasking

class Outpost:
    """ SentinelCam outpost functionality wrapped up as a Detector for the
    imagenode. Employs vision analysis to provide object detection and tracking
    data with real-time event logging, and image publication over imageZMQ.

    Parameters:
        detector (object): reference to the ImageNode Detector instance
        config (dict): configuration dictionary for this detector
        nodename (str): nodename to identify event messages and images sent
        viewnane (str): viewname to identify event messages and images sent
    """

    logger = None     # ZeroMQ async log publisher
    # Process-wide singletons, created once in __init__ before any use. Typed
    # non-Optional (with an ignored None seed) so member access doesn't trip the
    # type checker on this initialize-once pattern.
    publisher: "imagezmq.ImageSender" = None  # type: ignore[assignment]  # image publishing over imageZMQ

    def __init__(self, detector, config, nodename, viewname):
        self.nodename = nodename
        self.viewname = viewname
        self.detector = detector
        # configuration and setups
        self.cfg = config
        self.setups(config)
        # start at most one instance each of log and image publishing
        if not Outpost.publisher:
            Outpost.publisher = imagezmq.ImageSender("tcp://*:{}".format(
                self.publish_cam),
                REQ_REP=False)
        if not Outpost.logger:
            logging.config.dictConfig(self.logconfig)
            Outpost.logger = logging.getLogger()
        # optional self-introduction to a running camwatcher
        if self.camwatcher:
            self.camwatcher_greeting()
        # SpyGlass tooling and pipeline counters
        self._rate = FPS()
        self.sg = SpyGlass(viewname, self.dimensions, self.cfg)
        self._heartbeat = (0,0)
        self._looks = 0
        self._tick = 0
        self._evts = 0
        if self.depthAI:
            self.setup_OAK(config["depthai"])
        elif self.picamTracker:
            self.setup_picamera(config)

    def camwatcher_greeting(self):
        # only called when self.camwatcher is a configured connection string (truthy)
        assert isinstance(self.camwatcher, str)
        _host = socket.gethostname()
        handoff = {'cmd': 'CamUp',
                   'node': self.nodename,
                   'view': self.viewname,
                   'logger': f"tcp://{_host}:{self.publish_log}",
                   'images': f"tcp://{_host}:{self.publish_cam}"}
        msg = json.dumps(handoff)
        with zmq.Context().instance().socket(zmq.REQ) as sock:
            sock.connect(self.camwatcher)
            sock.send(msg.encode("ascii"))

    def object_tracker(self, camera, image, send_q):
        """ Called as an imagenode Detector for each image in the pipeline.

        On the OAK path this is a keepalive only — the event plane runs on its
        own threads (setup_OAK). On the picamera path it publishes the live
        scene, runs motion detection (the §4.10 NN scheduler), and drives the
        persistent host tracker + event manager through the PicameraIntake
        adapter. The legacy correlation-tracking cascade / scene-management /
        status machine was retired in the §4.10 closeout.

        Parameters:
            camera (Camera object): current camera
            image (OpenCV image): current image
            send_q (Deque): where (text, image) tuples can be passed
                            to the imagehub for Librarian processing
        """
        if self.depthAI:
            # OAK path: the event plane runs on its own threads (setup_OAK). This
            # callback is a keepalive — heartbeat only, no per-frame work here.
            self._oak_keepalive()
            return

        if self.publish_cam:
            if self.encoder[0] == 'c':
                buffer = simplejpeg.encode_jpeg(image,
                    quality=camera.jpeg_quality,
                    colorspace='BGR')
            else:
                # TODO add support for uncompressed, and video formats
                buffer = None
                self.publish_cam = False
                logging.error(f"JPEG only. Unsupported compression for image publishing '{self.encoder}', function disabled.")

            if buffer:
                self._rate.update()
                Outpost.publisher.send_jpg('|'.join([camera.text, self._rate.lastStamp().isoformat()]), buffer)
                # Heartbeat message with current pipeline frame rates over the logger.
                # TODO: Make this a configurable setting. Currently every 5 minutes.
                mm = self._rate.get_min()
                if mm % 5 == 0 and mm != self._heartbeat[1]:
                    tickrate = (self._tick - self._heartbeat[0]) / (5 * 60)
                    logging.info(f"fps({self._tick}, {self._looks}, {self._evts}, {tickrate:.2f}, {self._rate.fps():.2f})")
                    self._heartbeat = (self._tick, mm)

        if self.picamTracker:
            # §4.10 unified lifecycle: motion detection is the NN scheduler; the
            # persistent host tracker + event manager own events. Apply background
            # subtraction within the region of interest only — fast, every frame.
            x1, y1 = self.detector.top_left
            x2, y2 = self.detector.bottom_right
            ROI = image[y1:y2, x1:x2]
            gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            motionRect = self.sg.detect_motion(gray)
            self._picam_object_tracker(image, motionRect)
            self._tick += 1

    def setup_picamera(self, config) -> None:
        """§4.10 picamera unified lifecycle. Build the source-agnostic event plane —
        a persistent HostTracker + EventManager — driven from object_tracker via the
        PicameraIntake adapter. SpyGlass is demoted to a DETECT-only inference engine;
        motion (run every frame) schedules the inferences (the NN-scheduler role)."""
        from sentinelcam.hosttracker import HostTracker, TrackerConfig
        from sentinelcam.eventmanager import EventManager
        from sentinelcam.picamera_intake import PicameraIntake

        VEHICLE = {"car", "truck", "bus", "motorbike", "motorcycle", "bicycle", "train"}

        def classify(name):
            if name == "person":
                return "person"
            if name in VEHICLE:
                return "vehicle"
            return None                          # not interesting — dropped at ingest

        tracker = HostTracker(TrackerConfig.from_dict(self.tracker_cfg))
        events = EventManager(
            self.viewname, self.dimensions,
            sentinel_tasks=self.sentinel_tasks,
            emit=logging.getLogger().info,
            timestamp_fn=lambda ct: datetime.fromtimestamp(ct).isoformat(),
        )
        self._picam_tracker = tracker
        self._picam_events = events
        self._picam_intake = PicameraIntake(
            self.viewname, self.dimensions, tracker, events, classify)
        logging.info(f"Picamera event plane ready (host tracker, view {self.viewname})")

    def _picam_object_tracker(self, image, motionRect) -> None:
        """§4.10 per-frame event plane. Motion schedules SpyGlass DETECT inferences;
        results drive the HostTracker via observe(); quiet frames advance it with
        tick(). Exactly one SpyGlass request is in flight (REQ/REP): every recv
        (get_data) is paired with a send (apply_lens). While quiet and idle the
        pending result lingers unconsumed — pairing intact — until motion resumes."""
        ct = self._rate.lastStamp().timestamp()
        active = self._picam_events.event_open or self._picam_tracker.has_active()
        consumed = False
        if motionRect or active:
            if self.sg.has_result():
                (lens, dets) = self.sg.get_data()                     # the owed recv
                self.sg.apply_lens(LensTasking.Request_DETECT, image, # paired send (re-prime)
                                   self._rate.lastStamp())
                self._picam_intake.observe(ct, dets)
                self._looks += 1
                consumed = True
            # else: inference in flight — do nothing, keep the REQ/REP pairing intact
        if not consumed:
            self._picam_intake.tick(ct)
        self._evts = self._picam_events.events_opened

    def setups(self, config) -> None:
        if 'camwatcher' in config:
            self.camwatcher = config['camwatcher']
        else:
            self.camwatcher = False
        if 'publish_log' in config:
            self.publish_log = config['publish_log']
        else:
            self.publish_log = False
        if 'publish_cam' in config:
            self.publish_cam = config['publish_cam']
        else:
            self.publish_cam = False
        if 'spyglass' in config:
            self.dimensions = literal_eval(config['spyglass'])
        else:
            self.dimensions = (1024, 768)
        if 'encoder' in config:
            self.encoder = config["encoder"]
        else:
            self.encoder = 'cpu'
        self.depthAI = 'depthai' in config
        # §4.10 config-home graduation: the host-tracker config lives at the
        # detector level (detector.tracker) for ALL node types — the legacy
        # `tracker: none` string flag is retired. A dict here opts a non-OAK node
        # into the unified host-tracker lifecycle (setup_picamera); absent means
        # publish-only (live scene, no event plane). setup_OAK reads it too.
        self.tracker_cfg = config.get("tracker")
        self.picamTracker = not self.depthAI and isinstance(self.tracker_cfg, dict)
        self.sentinel_tasks = config['sentinel_tasks']
        self.logconfig = config['logconfig']

    def setup_OAK(self, config) -> None:
        """Build and start the redesigned OAK event plane (Phase 3/4).

        Replaces the retired DepthAI v2 PipelineFactory path. Constructs the
        Phase-2 device pipeline (one NN Archive, four host queues) and the
        in-process event plane — persistent tracker, event manager, EventSampler
        — then starts the two threads that own the device: the Plane-1
        ScenePublisher (q_jpeg -> scene PUB, always on) and the Plane-2
        OutpostIntake drain (det/meta/crops -> tracker/events/sampler). From here
        the imagenode detector callback is a keepalive (see object_tracker).
        """
        from sentinelcam.oak_camera import (OakCamera, CropMeta, classify_detection,
                                            MOBILENET_LABELS)
        from sentinelcam.hosttracker import HostTracker, TrackerConfig
        from sentinelcam.eventmanager import EventManager
        from sentinelcam.eventsampler import EventSampler, LateralTraversalStrategy
        from sentinelcam.outpost_intake import OutpostIntake, ScenePublisher, CropLookback

        oak_cfg = config.get("oak_pipeline", {})
        self._crop_quality = int(oak_cfg.get("jpeg_quality", 90))
        # Scene publication size produced by oak_camera.build_pipeline (768x432).
        # Keep in sync with that requestOutput; trk rects denormalize against it.
        scene_size = (768, 432)

        # Shared device->wall clock state. The scene publisher (Plane 1) and the
        # event manager / sampler (Plane 2) BOTH stamp records via _oak_clock(ct)
        # off the SAME per-frame device capture-time, so image[N] and trk/crp[N]
        # carry an identical timestamp string (exact-match correlation downstream:
        # camwatcher image naming, VehicleSpeed timestamp_to_offset, overlays).
        self._oak_t0_wall = None
        self._oak_t0_dev = None
        self._oak_clock_lock = threading.Lock()

        # Phase-2 device pipeline: one NN Archive, four host output queues.
        self._oak = OakCamera(config["nn_archive"], oak_cfg)
        run = self._oak.start()

        # Persistent tracker + event manager (host_tracker_design.md). The tracker
        # config now lives at the detector level (detector.tracker), read in setups
        # as self.tracker_cfg — NOT under the depthai block (§4.10 config graduation).
        tracker = HostTracker(TrackerConfig.from_dict(self.tracker_cfg))
        events = EventManager(
            self.viewname, scene_size,
            sentinel_tasks=self.sentinel_tasks,
            emit=logging.getLogger().info,
            timestamp_fn=self._oak_clock,        # maps device capture-time ct -> wall ISO
        )

        # Crop lookback holds device frames directly (poolhold-confirmed safe, §3.3).
        lookback = CropLookback(maxlen=int(config.get("lookback_frames", 90)))

        # Plane-2 crop publisher: a SECOND ImageZMQ socket, crop stream only (§4.4).
        self._crop_pub = imagezmq.ImageSender(
            "tcp://*:{}".format(config["crop_publish"]), REQ_REP=False)

        sampler = EventSampler(
            self.viewname, tracker, lookback,
            encode_crop=self._encode_crop,
            publish_crop=lambda text, jpg: self._crop_pub.send_jpg(text, jpg),
            emit_ote=logging.getLogger().info,
            spyglass_offer=None,            # TODO Phase 4.6: repurposed LensTasking crop inference
            strategy=LateralTraversalStrategy(),
            timestamp_fn=self._oak_clock,    # same device->wall clock as scene + trk
        )

        # Plane-1 scene publisher (always on, 24x7) over the shared image sender.
        # Stamps each frame from ITS device capture-time via the shared clock, so
        # the image timestamp matches the trk/crp timestamp for the same frame.
        scene = ScenePublisher(
            run.q_jpeg,
            lambda seq, dev_s, jpg: Outpost.publisher.send_jpg(
                "|".join([' '.join([self.nodename, self.viewname]).strip(),
                          'jpg', self._oak_clock(dev_s)]), jpg),
        )

        # Plane-2 drain thread (owns det/meta/crops).
        intake = OutpostIntake(
            run, tracker, events, sampler, lookback,
            classify=classify_detection,
            decode_meta=CropMeta.unpack,
            # specific MobileNet-SSD label name for the trk record (car/bus/train/…)
            label_name=lambda i: MOBILENET_LABELS[i] if 0 <= i < len(MOBILENET_LABELS) else str(i),
        )

        scene.start()
        intake.start()
        self._oak_tracker = tracker
        self._oak_events = events
        self._oak_intake = intake
        self._oak_scene = scene
        self._oak_t0 = time.monotonic()
        self._oak_last_hb = 0.0
        logging.info(
            f"OAK event plane started: view={self.viewname} "
            f"scene=:{self.publish_cam} crops=:{config['crop_publish']}")

    def _oak_clock(self, dev_seconds: float) -> str:
        """ISO timestamp for OAK OTE/scene records.

        Maps the device capture-time (`dev_seconds` = ImgFrame.getTimestamp()
        .total_seconds(), a monotonic device epoch) to wall-clock, anchoring the
        offset ONCE on the first frame seen (anti-pattern #5: capture-time, not
        per-call now()). Scene, trk, and crp all route through here off the same
        per-frame `dev_seconds`, so the same camera frame yields an IDENTICAL
        timestamp string across streams — required for exact-match correlation
        (camwatcher image naming, VehicleSpeed timestamp_to_offset, overlays).

        Note: relies on the scene and detection requestOutputs carrying the same
        device timestamp for a given source frame (same capture, propagated
        seqnum). If hardware ever shows them diverging, switch the anchor key to
        the seqnum (guaranteed identical across streams).
        """
        if self._oak_t0_wall is None:
            with self._oak_clock_lock:
                if self._oak_t0_wall is None:
                    self._oak_t0_wall = time.time()
                    self._oak_t0_dev = dev_seconds
        # both anchored together above; capture to locals so the invariant is
        # explicit (the double-checked lock defeats the type-narrowing on the attrs)
        t0_wall, t0_dev = self._oak_t0_wall, self._oak_t0_dev
        assert t0_wall is not None and t0_dev is not None
        return datetime.fromtimestamp(t0_wall + (dev_seconds - t0_dev)).isoformat()

    def _encode_crop(self, frame) -> bytes:
        """NV12 device crop -> JPEG. DepthAI v3 cannot hardware-encode dynamic
        crop sizes, so the selected crops are host-encoded (§4.3)."""
        nv12 = frame.getFrame()
        bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        return simplejpeg.encode_jpeg(bgr, quality=self._crop_quality, colorspace="BGR")

    def _oak_keepalive(self) -> None:
        """Heartbeat for the OAK detector callback. The event plane runs on its
        own threads; this only emits the periodic fps() heartbeat. Cadence is set
        by OAKcamera.read() pacing the imagenode loop."""
        self._tick += 1
        now = time.monotonic()
        if now - self._oak_last_hb >= 300:          # every 5 minutes
            elapsed = now - self._oak_t0
            scene_fps = self._oak_scene.published / elapsed if elapsed > 0 else 0.0
            st = self._oak_intake.stats()
            logging.info(
                f"fps({self._oak_scene.published}, {st['dets_seen']}, "
                f"{self._oak_events.events_opened}, {scene_fps:.2f}, {scene_fps:.2f})")
            self._oak_last_hb = now

class OAKcamera:
    """Keepalive camera shim for an OAK outpost (redesign Phase 3/4).

    In the redesigned OAK path the device output queues are owned by the event
    plane (the OutpostIntake drain thread + ScenePublisher started in
    ``Outpost.setup_OAK``), NOT by the imagenode camera loop. But imagenode's
    main loop (``while not send_q: read_cameras()``) still calls ``read()`` once
    per iteration and is paced ONLY by that call blocking. So this shim does no
    device I/O: it sleeps a fixed cadence to keep the loop off a hot spin and
    returns a tiny placeholder frame. ``Outpost.object_tracker`` is a keepalive
    on the OAK path and ignores the returned image.

    The OAK node's camera YAML must NOT set ``vflip``/``resize_width`` (frame
    geometry is handled on-device, e.g. ``oak_pipeline.rotate_180``); the
    framework would otherwise transform this placeholder needlessly.
    """

    _PACE_S = 0.05                       # ~20 Hz housekeeping; real 30 fps work is on the threads
    _PLACEHOLDER = np.zeros((4, 4, 3), dtype=np.uint8)

    def __init__(self, view) -> None:
        self.view = view
        self.frame = None
    def read(self) -> object:
        time.sleep(self._PACE_S)         # pace the imagenode loop without touching the device
        return OAKcamera._PLACEHOLDER
    def getImgFrame(self) -> object:
        return self.frame
    def stop(self) -> None:
        pass
