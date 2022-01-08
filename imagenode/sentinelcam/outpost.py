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
        self.state = "motion"
        self.lenstype = "motion"
        self.event_start = datetime.utcnow()
        self._lastPublished = 0
        self._heartbeat = 0
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

    def object_tracker(self, camera, image, send_q):
        """ Called as an imagnode Detector for each image in the pipeline.
        Using this code will deploy the SpyGlass for view and event managment.
        
        The SpyGlass provides for motion detection, followed by object detection,
        along with object tracking, in a cascading technique. Parallelization is
        implemented through multiprocessing with a data exchange in shared memory. 

        Parameters:
            camera (Camera object): current camera
            image (OpenCV image): current image
            send_q (Deque): where (text, image) tuples can be appended for
                            sending to the imagehub for Librarian processing

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
                #  TODO: Make this a configurable setting. 
                mm = self._rate.update().minute
                if mm != self._heartbeat: 
                    logging.info(f"fps{self._rate.fps():.2f} at tick {self._tick}")
                    self._heartbeat = mm

        rects = []                     # always start fresh, no determinations made
        targets = self.sg.get_count()  # number of ojects tracked by the SpyGlass

        if targets == 0:
            # No objects currently in view, apply background
            # subtraction model within ROI for motion detection
            x1, y1 = self.detector.top_left
            x2, y2 = self.detector.bottom_right
            ROI = image[y1:y2, x1:x2]
            # convert to grayscale and smoothen using gaussian kernel
            gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # motion detection
            motionRect = self.sg.detect_motion(gray)
            if motionRect: 
                # TODO: Still need support for motion-only mode where the 
                # setup for cfg["detectobject"] == 'motion'
                # With an aggregate area of motion in play, there would be no
                # need for the CentroidTracker. Just report the motion rectangle
                # as the the event data.
                self._looks += 1
                self.lenstype = "detect"
        else:
            motionRect = None

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
                (rects, labels) = self.sg.get_data()
                logging.debug(f"LensTasking '{self.state}' result: {len(rects)} objects, tick {self._tick}")

                if len(rects) > 0:
                    # Have a non-empty result set back from the 
                    # SpyGlass, note the time and default to tracking. 
                    self.sg.lastUpdate = datetime.utcnow()
                    self.lenstype = 'track'  

                    # run detection again if requested
                    if self.state == 'redetect':
                        self.lenstype = 'detect'

                else:
                    # Based on the Outlook <-> SpyGlass protocol, this could be an
                    # old, now very stale, empty result set just received. In which 
                    # case, it's meaningless. So look again. On the other hand, this 
                    # could just as easily represent an end to the current event.
                    
                    if self.state in ['motion', 'redetect']:
                        self.lenstype = 'detect'  # Keep looking as long as there is motion.

                    if self.sg.status == 'active':
                        self.lenstype = 'motion'  # Event could be ending, just send the NOOP.

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

                logging.debug(f"sending '{self.lenstype}' to LensTasking, tick {self._tick}, look {self._looks}")
                self.sg.apply_lens(self.lenstype, image)

                # Now work through current SpyGlass result set, if any 
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
                        # occur during tracking, since that does prodice new objects. 
                        if target is None:
                            if labels is None:
                                logging.debug("How did we get here?")
                                labels=[]
                            classname = labels[i].split(' ')[0][:-1] if i < len(labels) else 'mystery'
                            targetText = "_".join([classname, str(objectID)])
                            target = self.sg.new_target(objectID, classname, targetText)

                        rect = rects[i] if i<len(rects) else (0,0,0,0)  # TODO: fix this stupid hack
                        target.update_geo(rect, centroid, self.lenstype, self.sg.lastUpdate)
                        logging.debug(f"update_geo:{target.toJSON()}")

                    for target in self.sg.get_targets():
                        # Drop vanished objects from SpyGlass 
                        if target.objectID not in self.ct.objects.keys():
                            logging.debug(f"dropped Target {target.objectID}")
                            self.sg.drop_target(target.objectID)
                        # Keep it simple for now, only track desired object classes?
                        if target.classname != "person":
                            logging.warning(f"dropping unexpected [{target.classname}], objectID {target.objectID}")
                            self.sg.drop_target(target.objectID)
                            # CentroidTracker still has this, ignore it going forward
                            self.dropList[target.objectID] = True  

                    targets = self.sg.get_count()
                    logging.debug(f"Now tracking {targets} objects, tick {self._tick}")
                    if targets == 0:
                        # Finished processing results, and came up empty. Detection should run
                        # again by default. Note the forced change in state for the next pass.
                        self.lenstype = 'redetect' 
                        # Also wipe the memory of the CentroidTracker
                        for objID in self.ct.objects.keys():
                            self.ct.deregister(objID)
                    else:
                        if self.sg.status == 'inactive':
                            # This is a new event, begin logging the tracking data
                            self.sg.status = 'active'
                            ote = self.sg.new_event()
                            ote['fps'] = self._rate.fps()
                            logging.info(f"ote{json.dumps(ote)}")
                            self.event_start = self.sg.event_start

                        # Discard motion rectangle with targets in view?
                        motionRect = None 

                        if self.sg.status == 'active':
                            # event in progress
                            ote = self.sg.trackingLog()
                            for target in self.sg.get_targets():
                                if target.upd == self.sg.lastUpdate:
                                    ote.update(target.toTrk())
                                    logging.info(f"ote{json.dumps(ote)}")
                    
                elif self.lenstype == "detect" and self.sg.status == "inactive": 
                    # current status is inactive, and object detection found nothing
                    # assume this was a false alarm, and resume motion detection
                    logging.debug(f"revert to motion, tick {self._tick}, look {self._looks}")
                    self.lenstype = "motion"
                    self.state = "motion"
            else:
                # SpyGlass ia busy. Skip this cycle and return without further analysis. 
                pass

        # outpost tick count 
        self._tick += 1  
        if self._tick % self.skip_frames == 0:
            # tracking threshold encountered? run detection again
            if self.lenstype == "track":
                logging.debug(f"tracking threshold reached, tick {self._tick}, look {self._looks}")
                self.lenstype = "detect"

        stayalive = True   # assume there is still work to do
        # If an event is in progress, is it time to end it?
        if self.sg.status == 'active':
            if targets == 0:
                stayalive = False
                self.dropList = {}
                self.lenstype = "motion"
                self.sg.status = "inactive"
                logging.debug(f"Ending active event {self.sg.eventID}")
            else:
                # fail safe kill switch, forced shutdown after 15 seconds 
                # TODO: design flexibility for this via ruleset in configuration?
                #  ----------------------------------------------------------
                event_elapsed = datetime.utcnow() - self.event_start
                if event_elapsed.seconds > 15:
                    stayalive = False
                    self.sg.status = "quiet"
                    logging.debug(f"Status is quiet, ending event {self.sg.eventID}")
                    # TODO: placeholder for something more clever
                    # For now, just clear the SpyGlass and go back to motion detection
                    for target in self.sg.get_targets():
                        self.sg.drop_target(target.objectID)   
                    self.dropList = {}
                    self.lenstype = "motion"
                    self.sg.status = "inactive"

            if not stayalive:
                ote = self.sg.trackingLog()
                ote['evt'] = 'end'
                logging.info(f"ote{json.dumps(ote)}")

        if self.state != self.lenstype:
            logging.debug(f"State change from {self.lenstype} to {self.state}")
            self.state = self.lenstype
        if targets == 0:
            if motionRect:
                # Have motion but no object yet detected. Perhaps 
                # this should be logged for review and analysis?
                pass        

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
        self.skip_frames = config["skip_frames"]
    