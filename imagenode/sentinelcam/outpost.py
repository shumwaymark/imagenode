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
from sentinelcam.spyglass import SpyGlass, CentroidTracker

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

    def __init__(self, detector, config, nodename, viewname):
        self.nodename = nodename
        self.viewname = viewname
        self.detector = detector
        # configuration and setups
        self.cfg = config
        self.setups(config)
        # setup CentroidTracker and SpyGlass tooling 
        self._rate = FPS()
        self.ct = CentroidTracker(maxDisappeared=50, maxDistance=100)  # TODO: add parms to config
        self.sg = SpyGlass(viewname, self.dimensions, self.cfg)
        self.status = Outpost.Status_INACTIVE
        self.nextLens = Outpost.Lens_MOTION
        self.event_start = datetime.utcnow()
        self._lastPublished = 0
        self._heartbeat = 0
        self._noMotion = 0
        self._looks = 0
        self._tick = 0
        self.dropList = {}  # unwanted objects
        # when configured, start at most one instance each of log and image publishing
        if self.publish_log:
            if not Outpost.logger:
                Outpost.logger = self.start_logPublisher(self.publish_log)
        if self.publish_cam:
            if not Outpost.publisher:
                Outpost.publisher = imagezmq.ImageSender("tcp://*:{}".format(
                    self.publish_cam), 
                    REQ_REP=False)

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
    
    def _clear_CentroidTracker(self):
        trkdObjs = list(self.ct.objects.keys())
        for o in trkdObjs:
            self.ct.deregister(o)
        self.dropList = {}
    
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
            # Try not to over-publish, estimate throttling based on expected frame rate
            ns = time.time_ns()
            if (ns - self._lastPublished) > (1000000000 // camera.framerate):
                #ret_code, jpg_buffer = cv2.imencode(".jpg", image, 
                #    [int(cv2.IMWRITE_JPEG_QUALITY), camera.jpeg_quality])
                jpg_buffer = simplejpeg.encode_jpeg(image, 
                    quality=camera.jpeg_quality, 
                    colorspace='BGR')
                # TODO: Question: does the threshold above need to allow for
                # image compression overhead? Asked for 32 fps, getting about 
                # 24-25 per second for a (640,480) image size
                Outpost.publisher.send_jpg(camera.text, jpg_buffer)
                self._lastPublished = ns
                # Heartbeat as current publishing frame rate over the logger. 
                # TODO: Make this a configurable setting. Currently every 5 minutes.
                # This should probably go to the imagehub as a status item for the librarian.
                mm = self._rate.update().minute
                if mm % 5 == 0 and mm != self._heartbeat: 
                    logging.info(f"fps{self._rate.fps():.2f} at tick {self._tick}")
                    self._heartbeat = mm

        rects = []                     # fresh start here, no determinations made
        targets = self.sg.get_count()  # number of ojects tracked by the SpyGlass

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
            if self.status != Outpost.Status_ACTIVE: 
                # TODO: Still need support for motion-only mode where 
                # the setup for cfg["detectobject"] == 'motion'
                # With an aggregate area of motion in play, there would be no
                # need for the CentroidTracker. Just report the motion rectangle
                # as the event data.
                self._looks += 1
                if self.nextLens not in [Outpost.Lens_REDETECT, Outpost.Lens_RESET]:
                    self.nextLens = Outpost.Lens_DETECT

        if targets > 0 or motionRect:
            # Motion was detected or already have targets in view
            # Alternating between detection and tracking 
            # - object detection first, begin looking for characteristics on success
            # - initiate and run with tracking after objects detected
            # - re-deploy detection periodically, more frequently for missing expected attributes
            #   - such as persons without faces (after applying a specialized lens)
            #   - these results would add to the list of tracked objects
            #   - otherwise re-detect as configured, building a new list of trackers as an outcome
            # ----------------------------------------------------------------------------------------
            #  This a lot to ask of the imagenode module on a Raspberry Pi 4B. Minus some tricked-out 
            #  hardware provisioning, such as a USB-driven accelerator, most of this should be in 
            #  batch jobs on the Sentinel instead of out here on the edge.
            # ----------------------------------------------------------------------------------------

            if self.sg.has_result(): 

                # SpyGlass has results available, retrieve them now
                (lens, rects, labels) = self.sg.get_data()
                logging.debug(f"LensTasking lenstype {lens} result: {len(rects)} objects, tick {self._tick}")

                if self.nextLens == Outpost.Lens_RESET:
                    # This is effectively a NOOP for the SpyGlass. Clear the result set.
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
                    self.nextLens = Outpost.Lens_TRACK
                else:
                    # An active event could be ending, just send the NOOP?
                    #if self.status == Outpost.Status_ACTIVE:
                    #    self.nextLens = Outpost.Lens_MOTION
                    logging.debug(f"No result set during event, lens {lens} nextLens {self.nextLens}, tick {self._tick}")

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
                if len(rects) > 0:
                    # Centroid tracking algorithm courtesy of PyImageSearch.
                    # Using this to map tracked object centroids back to a  
                    # dictionary of targets managed by the SpyGlass
                    centroids = self.ct.update(rects)

                    # TODO: Need to validate CentroidTracker initilization and overall
                    # fit within the context of the Outpost use cases. Specifically the
                    # max disappeared limit. Also need to re-think how often detection 
                    # is redeployed given the sparse approach inherent in this design
                    for i, (objectID, centroid) in enumerate(centroids.items()):

                        # Ignore anything on the drop list
                        if objectID in self.dropList:
                            continue

                        # Grab the SpyGlass target via its object ID
                        target = self.sg.get_target(objectID)

                        # Create new targets for tracking as needed. This should never
                        # occur during tracking, since only detection produces new objects. 
                        if target is None:
                            if labels is None:
                                logging.error(f"How did we get here with obj {objectID}? Look {self._looks} Lens {lens} tick {self._tick}")
                                labels=[]
                            classname = labels[i].split(' ')[0][:-1] if i < len(labels) else 'mystery'
                            targetText = "_".join([classname, str(objectID)])
                            target = self.sg.new_target(objectID, classname, targetText)

                        rect = rects[i] if i<len(rects) else (0,0,0,0)  # TODO: fix this stupid hack? - maybe not needed anymore
                        target.update_geo(rect, centroid, lens, self.sg.lastUpdate)
                        logging.debug(f"update_geo:{target.toJSON()}")

                    for target in self.sg.get_targets():
                        # Drop vanished objects from SpyGlass 
                        if target.objectID not in self.ct.objects.keys():
                            logging.debug(f"Target {target.objectID} vanished")
                            self.sg.drop_target(target.objectID)
                        # Keep it simple for now, only track desired object classes?
                        if target.classname != "person":
                            logging.warning(f"Dropped [{target.classname}] obj {target.objectID} lens {lens} next {self.nextLens} tick {self._tick} look {self._looks}")
                            self.sg.drop_target(target.objectID)
                            # CentroidTracker still has this, ignore it for the 
                            # remainder of the event. This is admitedly, a bit clumsy.
                            self.dropList[target.objectID] = True  

                    # If tracking, and the Target has not moved... re-detect immediately and
                    # take note of this. Maybe the tracker got lost, or perhaps the Target 
                    # is just standing still. (TODO)

                    # To do this correctly, we need more CPU power. Ideally, detection and tracking
                    # should run in parallel, to provide for a more responsive feedback loop to
                    # tune the tracker. This technique would run the detector more often, but only on
                    # selected regions of interest within the image, where we already think the object 
                    # should be. This is how a country boy might try build something that attempts to
                    # masquerade as a HyrdaNet. 

                    targets = self.sg.get_count()
                    logging.debug(f"Now tracking {targets} objects, tick {self._tick}")
                    if targets == 0:
                        # Finished processing results, and came up empty. Detection should run
                        # again by default. Note the forced change in state for the next pass.
                        self.nextLens = Outpost.Lens_REDETECT
                        # Also wipe the memory of the CentroidTracker, just to be certain.
                        self._clear_CentroidTracker()
                    else:
                        if self.status == Outpost.Status_INACTIVE:
                            # This is a new event, begin logging the tracking data
                            self.status = Outpost.Status_ACTIVE
                            ote = self.sg.new_event()
                            ote['fps'] = self._rate.fps()        # TODO change this to tick count per second  
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
                else:
                    # Empty result set? Wipe the memory of the CentroidTracker; want a clean slate now.
                    #self._clear_CentroidTracker()
                    if lens == Outpost.Lens_DETECT and self.status == Outpost.Status_INACTIVE: 
                        # Current status is inactive, and object detection found nothing
                        # assume this was a false alarm, and resume motion detection?
                        logging.debug(f"revert to motion?, tick {self._tick}, look {self._looks}")
                        #self.nextLens = Outpost.Lens_MOTION
            else:
                # SpyGlass ia busy. Skip this cycle and keep going. 
                pass

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

        stayalive = True   # Assume there is still something going on
        # If an event is in progress, is it time to end it?
        if self.status == Outpost.Status_ACTIVE:
            if motionRect is None:
                self._noMotion += 1
            if targets == 0:
                stayalive = False
                logging.debug(f"Ending active event {self.sg.eventID}")
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
                for target in self.sg.get_targets():
                    self.sg.drop_target(target.objectID)   
                # Also, wipe the memory of the CentroidTracker.
                self._clear_CentroidTracker()
                self.nextLens = Outpost.Lens_RESET
                self.status = Outpost.Status_INACTIVE

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
        self.skip_factor = config["skip_factor"]
    