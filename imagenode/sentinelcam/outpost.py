"""outpost: sentinelcam integration with imagenode 
Support for image publishing and outpost functionality

Copyright (c) 2021 by Mark K Shumway, mark.shumway@swanriver.dev
License: MIT, see the sentinelcam LICENSE for more details.
"""

import cv2
import imutils
import logging
import json
import socket
import sys
import time
import zmq
import imagezmq
import numpy as np
import simplejpeg
from ast import literal_eval
from datetime import datetime
from zmq.log.handlers import PUBHandler
from sentinelcam.utils import FPS
from sentinelcam.spyglass import SpyGlass

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
    publisher = None  # image publishing over imageZMQ
    oakCameras = {}   # OAK camera image queues, keyed by viewname 

    Status_INACTIVE = 0
    Status_QUIET = 1
    Status_ACTIVE = 2
    
    Status = ["Inactive","Quiet","Active"]

    Lens_MOTION = 0
    Lens_DETECT = 1
    Lens_TRACK = 2
    Lens_REDETECT = 3
    Lens_RESET = 4

    Lens = ["Motion","Detect","Track","ReDetect","Reset"]

    # MobilenetSSD label texts
    MobileNetSSD_labels = ["background", "aeroplane", "bicycle", "bird", 
        "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", 
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def __init__(self, detector, config, nodename, viewname):
        self.nodename = nodename
        self.viewname = viewname
        self.detector = detector
        # configuration and setups
        self.cfg = config
        self.setups(config)
        # when configured, start at most one instance each of log and image publishing
        if self.publish_log:
            if not Outpost.logger:
                Outpost.logger = self.start_logPublisher(self.publish_log)
        if self.publish_cam:
            if not Outpost.publisher:
                Outpost.publisher = imagezmq.ImageSender("tcp://*:{}".format(
                    self.publish_cam), 
                    REQ_REP=False)
        # setup CentroidTracker and SpyGlass tooling 
        self._rate = FPS()
        self.sg = SpyGlass(viewname, self.dimensions, self.cfg)
        self.status = Outpost.Status_INACTIVE
        self.nextLens = Outpost.Lens_MOTION
        self.event_start = datetime.utcnow()
        self._lastPublished = 0
        self._heartbeat = (0,0)
        self._noMotion = 0
        self._looks = 0
        self._tick = 0
        if self.depthAI:
            self.setup_OAK(config["depthai"])

    def start_logPublisher(self, publish):
        log = logging.getLogger()
        log.info('Activating log publisher on port {}'.format(publish))
        zmq_log_handler = PUBHandler("tcp://*:{}".format(publish))
        zmq_log_handler.setFormatter(logging.Formatter(fmt='{asctime}|{message}', style='{'))
        zmq_log_handler.root_topic = self.nodename
        log.addHandler(zmq_log_handler)
        if self.camwatcher:
            handoff = {'node': self.nodename, 
                       'log': self.publish_log, 
                       'video': self.publish_cam,
                       'host': socket.gethostname()}
            msg = "CameraUp|" + json.dumps(handoff)
            try:
                with zmq.Context().socket(zmq.REQ) as sock:
                    log.debug('connecting to ' + self.camwatcher)
                    sock.connect(self.camwatcher)
                    sock.send(msg.encode("ascii"))
                    resp = sock.recv().decode("ascii")
            except Exception as ex:
                log.exception('Unable to connect with camwatcher:' + ex)
                sys.exit()
        log.handlers.remove(log.handlers[0]) # OK, all logging over PUB socket only
        log.setLevel(logging.INFO)
        return log
    
    def object_tracker(self, camera, image, send_q):
        """ Called as an imagnode Detector for each image in the pipeline.
        Using this code will deploy the SpyGlass for view and event managment.
        
        The SpyGlass provides for motion detection, followed by object detection,
        along with object tracking, in a cascading technique. Parallelization is
        implemented through multiprocessing with a data exchange in shared memory. 

        Parameters:
            camera (Camera object): current camera
            image (OpenCV image): current image
            send_q (Deque): where (text, image) tuples can be passed
                            to the imagehub for Librarian processing

        """
        if self.publish_cam:
            ns = time.time_ns()
            if self.encoder[0] == 'c':
                # Try not to over-publish, estimate throttling based on expected frame rate
                if (ns - self._lastPublished) > (1000000000 // camera.framerate):
                    buffer = simplejpeg.encode_jpeg(image, 
                        quality=camera.jpeg_quality, 
                        colorspace='BGR')
                    # TODO: Question: does the threshold above need to allow for image 
                    # compression overhead? Asked for 32 FPS, getting about 24-25 per second 
                    # for a (640,480) image size, within a pipeline velocity of 40-50 ticks/second.
                else:
                    buffer = None
            elif self.encoder[0] == 'o':
                encFrameMsg = self.jpegQ.tryGet()
                if encFrameMsg is not None:
                    buffer = bytearray(encFrameMsg.getData())
                else:
                    buffer = None
            else:
                # TODO add support for uncompressed, and video formats
                buffer = None
                self.publish_cam = False
                logging.error(f"JPEG only. Unsupported compression for image publishing '{self.encoder}', function disabled.") 
                    
            if buffer:
                Outpost.publisher.send_jpg(camera.text, buffer)
                self._lastPublished = ns
                # Heartbeat message with current pipeline frame rates over the logger. 
                # TODO: Make this a configurable setting. Currently every 5 minutes.
                mm = self._rate.update().minute
                if mm % 5 == 0 and mm != self._heartbeat[1]: 
                    tickrate = (self._tick - self._heartbeat[0]) / (5 * 60)
                    logging.info(f"fps({self._tick}, {self._looks}, {tickrate:.2f}, {self._rate.fps():.2f})")
                    self._heartbeat = (self._tick, mm)

        rects = []                      # fresh start here, no determinations made
        targets = self.sg.get_count()   # number of ojects tracked by the SpyGlass
        interestingTargetFound = False  # only begin event capture when interested

        # Always apply the motion detector. It's fast and the information 
        # is generally useful. Apply background subtraction model within
        # region of interest only.
        x1, y1 = self.detector.top_left
        x2, y2 = self.detector.bottom_right
        ROI = image[y1:y2, x1:x2]
        # convert to grayscale and smoothen using gaussian kernel
        gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Motion detection. This returns an aggregate 
        # rectangle of the estimated area of motion. 
        lens = Outpost.Lens_MOTION
        motionRect = self.sg.detect_motion(gray)
        if motionRect: 
            self._noMotion = 0
            # motion-only mode provides for the capture and logging
            # of only the aggregate area of motion, without applying
            # any specialized lenses to the SpyGlass
            if self.motion_only:
                self.nextLens = Outpost.Lens_MOTION
                rects = [motionRect]
                labels = ["motion: 0"]
            else:
                if self.status != Outpost.Status_ACTIVE: 
                    self._looks += 1
                    if self.nextLens not in [Outpost.Lens_REDETECT, Outpost.Lens_RESET]:
                        self.nextLens = Outpost.Lens_DETECT
        else:
            self._noMotion += 1

        if self.depthAI:
            # When running DepthAI on an OAK camera, pull down any neural net results now. 
            # SpyGlass can provide for optional supplemental analysis, append any such results afterwards.
            cnt = 0
            imageSeqThreshold = camera.cam.getImgFrame().getSequenceNum() - 2
            normVals = np.full(4, self.dimensions[1])
            normVals[::2] = self.dimensions[0]
            for net in self.nnQs:
                # multiple neural nets may be used in parallel 
                for nnMsg in net.tryGetAll(): 
                    # Each message relates to a single image, and can contain multiple results.
                    # Discard history up to an estimated starting point to better align with 
                    # the current image. This is an experimental factor used for initial testing. 
                    nnSeq = nnMsg.getSequenceNum()
                    if nnSeq < imageSeqThreshold:  
                        logging.debug(f"Discarding ImgDetection sequence {nnSeq}")
                    else:
                        (rects, labels) = ([],[])
                        for nnDet in nnMsg.detections:
                            bbox = [nnDet.xmin, nnDet.ymin, nnDet.xmax, nnDet.ymax]
                            logging.debug(f"nnDet[{cnt}] {nnDet.label}, bbox {bbox} image {nnSeq}, look {self._looks}")
                            # normalize detection result bounding boxes to frame dimensioms 
                            bbox = np.array(bbox)
                            rects.append((np.clip(bbox, 0, 1) * normVals).astype(int))
                            labels.append("{}: {:.4f}".format(
                                Outpost.MobileNetSSD_labels[nnDet.label],
                                nnDet.confidence))
                            cnt += 1
                        interesting = self.sg.reviseTargetList(Outpost.Lens_DETECT, rects, labels)
                        if interesting:
                            interestingTargetFound = True
            if cnt:
                # TODO: might want to apply a special lens to the SpyGlass?
                self.nextLens = Outpost.Lens_MOTION  

        if self.spyGlassOnly and (targets > 0 or motionRect):
            # ----------------------------------------------------------------------------------------
            #                     SpyGlass-only draft design pattern
            # ----------------------------------------------------------------------------------------
            # Motion was detected or already have targets in view
            # Alternating between detection and tracking 
            # - object detection first, begin looking for characteristics on success
            # - initiate and run with tracking after objects detected
            # - re-deploy detection periodically, more frequently for missing expected attributes
            #   - such as persons without faces (after applying a specialized lens)
            #   - these supplementary results would not have individual trackers applied to them
            #   - otherwise re-detect as configured, building a new list of trackers as an outcome
            # ----------------------------------------------------------------------------------------
            #  This a lot to ask of the imagenode module on a Raspberry Pi 4B. Minus some tricked-out 
            #  hardware provisioning, such as a USB-driven accelerator, most of this should be in 
            #  batch jobs on the Sentinel instead of out here on the edge.
            # ----------------------------------------------------------------------------------------

            if self.sg.has_result(): 

                # SpyGlass has results available, retrieve them now
                (lens, rects, labels) = self.sg.get_data()
                logging.debug(f"LensTasking lenstype {lens} result: {len(rects)} objects, motionRect {motionRect}, tick {self._tick}")

                if self.nextLens == Outpost.Lens_RESET:
                    # This is effectively a NOOP for the SpyGlass, clearing current results.
                    # A lens command = 0 insures that the next result set will be empty. 
                    self.nextLens = Outpost.Lens_MOTION
                    rects = []
                elif self.nextLens == Outpost.Lens_REDETECT:
                    # If requested, clear results and apply a new lens.
                    self.nextLens = Outpost.Lens_DETECT
                    rects = []

                # Based on the Outlook <-> SpyGlass protocol, any result set 
                # could be old, now very stale, and just received. In which case 
                # it's meaningless. So try to keep descision making in context.

                if len(rects) > 0:
                    # Have a non-empty result set back from the 
                    # SpyGlass, note the time and default to tracking. 
                    self.sg.lastUpdate = datetime.utcnow()
                    if self.object_tracking:
                        self.nextLens = Outpost.Lens_TRACK

                if self.nextLens != lens:
                    logging.debug(f"Changing lens from {lens} to {self.nextLens}, tick {self._tick}")

                # Send current frame on out to the SpyGlass for processing.
                # This handshake is based on a REQ/REP socket pair over ZMQ. 
                # More simply, for every send() there must always be a recv() 
                #
                # Always. Every time. Why the emphasis? 
                # -------------------------------------
                # The design of this protocol requires that whenever results are recieved, 
                # a request must follow. This is usually what's desired. Since it always takes
                # a while to get an answer back, just send the SpyGlass request now without delay.
                #
                # What does this mean for the Outpost?  Results are always evaluated only after
                # another request has already gone out. Thus that pending result set may linger 
                # quite a while before the next recv() if the Outpost determines there is nothing 
                # left to look at and loses interest.

                logging.debug(f"Sending '{self.nextLens}' to LensTasking, tick {self._tick}, look {self._looks}")
                self.sg.apply_lens(self.nextLens, image)

                # With current frame sent to the SpyGlass for analysis, there is now 
                # time to work through the result set from the prior request, if any.
                if self.spyGlassOnly and len(rects) > 0:
                    interestingTargetFound = self.sg.reviseTargetList(lens, rects, labels)

                # To do this correctly, we need more CPU power. Ideally, detection and tracking
                # should run in parallel to provide for a more responsive feedback loop to
                # tune the tracker. This technique would run the detector more often, but only on
                # selected regions of interest within the image, where we already think the object 
                # should be. This is how a country boy might try build something that attempts to
                # masquerade as a HyrdaNet. 

            else:
                # SpyGlass is busy. Skip this cycle and keep going. 
                pass

        if len(rects) > 0:
            # If tracking and any Target of interest has not moved, re-detect immediately
            # and take note of this. Maybe the tracker got lost, or perhaps the Target 
            # is just standing still. (TODO)

            targets = self.sg.get_count()
            logging.debug(f"Now tracking {targets} objects, tick {self._tick}")
            if targets == 0:
                # Finished processing results, and came up empty. Detection should run
                # again by default. Note the forced change in state for the next pass.
                self.nextLens = Outpost.Lens_REDETECT
            else:
                if self.status == Outpost.Status_INACTIVE:
                    if interestingTargetFound or (motionRect and self.motion_only):
                        # This is a new event, begin logging the tracking data
                        self.status = Outpost.Status_ACTIVE
                        ote = self.sg.new_event()
                        ote['fps'] = self._rate.fps()
                        logging.info(f"ote{json.dumps(ote)}")
                        self.event_start = self.sg.event_start

                if self.status == Outpost.Status_ACTIVE:
                    # event in progress
                    ote = self.sg.trackingLog('trk')
                    ote['lens'] = lens                   # just curious, can see when camwatcher logging=DEBUG
                    ote['looks'] = self._looks           # just curious, can see when camwatcher logging=DEBUG
                    for target in self.sg.get_targets():
                        if target.upd == self.sg.lastUpdate:
                            ote.update(target.toTrk())
                            logging.info(f"ote{json.dumps(ote)}")
        
        # outpost tick count 
        self._tick += 1  
        if self._tick % self.skip_factor == 0: 
            # Tracking threshold encountered? Run detection again. Should perhaps measure this 
            # based on both the number of successful tracking calls, as well as an elapsed time 
            # threshold. It might make sense to formulate based on the tick count if there is
            # an efficient way to gather metrics in-flight (something for the TODO list). 
            # As implemented, this is often out of phase from the surrounding logic, i.e. detection 
            # may have just completed. Formulating a one-size-fits-all solution is not simple. Much 
            # depends on the field and depth of view. Cameras in close proximity to the subject ROI 
            # need to be able to respond quickly to large changes in the images, such as when the 
            # subject is moving towards the camera. Correlation tracking may not even be effective
            # in these scenarios.
            if lens == Outpost.Lens_TRACK:
                logging.debug(f"tracking threshold reached, tick {self._tick}, look {self._looks}")
                self.nextLens = Outpost.Lens_REDETECT

        if self.status == Outpost.Status_ACTIVE:
            # If an event is in progress, is it time to end it?
            stayalive = True   # Assume there is still something going on
            if targets == 0:
                stayalive = False
                logging.debug(f"Event {self.sg.eventID} is tracking no targets")
            elif self._noMotion > 5:
                # This is more bandaid than correct. Mostly because the 
                # CentroidTracker is currently hanging on to objects longer 
                # than necessary
                self.status = Outpost.Status_QUIET
            else:
                # Fail safe kill switch, forced shutdown after 15 seconds. 
                # TODO: Design flexibility for this via ruleset in configuration?
                #  ----------------------------------------------------------
                event_elapsed = datetime.utcnow() - self.event_start
                if event_elapsed.seconds > 15:
                    self.status = Outpost.Status_QUIET

            if self.status == Outpost.Status_QUIET:
                # TODO: Placeholder for something more clever.
                # For now, just call it quits and go back to motion detection
                logging.debug(f"Status is quiet, ending event {self.sg.eventID}, targets {targets} noMotion {self._noMotion} tick {self._tick}")
                stayalive = False

            if not stayalive:
                logging.info(f"ote{json.dumps(self.sg.trackingLog('end'))}")
                # Ultimately, need more smarts around this. For now though,
                # this is quick, easy, and painless. Just erase the SpyGlass
                # memory and reset for a fresh start.
                self.sg.resetTargetList()
                self.nextLens = Outpost.Lens_RESET
                self.status = Outpost.Status_INACTIVE
        else:
            # temporary hack after DepthAI changes: shutdown a runaway SpyGlass?
            if targets > 0 and self._noMotion > 2:
                self.sg.resetTargetList()
                self.nextLens = Outpost.Lens_RESET

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
        self.object_detection = config["detectobjects"]
        self.object_tracking = config["tracker"] != "none"
        self.motion_only = self.object_detection == "motion"
        if 'skip_factor' in config:
            self.skip_factor = config["skip_factor"]
        else:
            self.skip_factor = 30
        self.depthAI = 'depthai' in config
        self.spyGlassOnly = not self.depthAI

    def setup_OAK(self, config) -> None:
        from sentinelcam.oak_camera import PipelineFactory
        self.oak = PipelineFactory(config["pipeline"]).device
        self.frameQ = self.oak.getOutputQueue(name=config["images"], maxSize=4, blocking=False)
        self.jpegQ = self.oak.getOutputQueue(name=config["jpegs"], maxSize=4, blocking=False)
        self.nnQs = []
        for net in config['neural_nets'].values():
            self.nnQs.append(self.oak.getOutputQueue(name=net, maxSize=50, blocking=False))
            logging.debug(f"DepthAI neural net queue '{net}' opened")
        Outpost.oakCameras[self.viewname] = self.frameQ

class OAKcamera:
    def __init__(self, view) -> None:
        self.frame = None
        self.frame_q = Outpost.oakCameras[view]
    def read(self) -> object:
        self.frame = self.frame_q.get()
        return self.frame.getCvFrame()
    def getImgFrame(self) -> object:
        return self.frame
    def stop(self) -> None:
        pass