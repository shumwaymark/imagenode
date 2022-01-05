"""outpost: sentinelcam integration with imagenode 
Support for video publishing and outpost functionality

Copyright (c) 2021 by Mark K Shumway, mark.shumway@swanriver.dev
License: MIT, see the sentinelcam LICENSE for more details.
"""

import cv2
import imutils
import logging
import json
import socket
import uuid
import sys
import zmq
import imagezmq
import simplejpeg
from ast import literal_eval
from datetime import datetime
from zmq.log.handlers import PUBHandler
from sentinelcam.utils import FPS
from sentinelcam.spyglass import SpyGlass, CentroidTracker

class Outpost:
    """ SentinelCam outpost functionality wrapped up as a Detector for the
    imagenode. Employs vision analysis to provide object detection and tracking 
    data with real-time event logging, and published image capture over ZMQ.

    Parameters:
        detector (object): reference to the ImageNode Detector instance 
        config (dict): configuration dictionary for this detector
        nodename (str): nodename to identify event messages and images sent
        viewnane (str): viewname to identify event messages and images sent
    """

    logger = None # zmq async log publisher  
    publisher = None # imagezmq video stream publishing

    def __init__(self, detector, config, nodename, viewname):
        self.nodename = nodename
        self.viewname = viewname
        self.detector = detector
        # configuration items
        self.cfg = config
        self.setups(config)
        # setup CentroidTracker and SpyGlass tooling 
        self._rate = FPS()
        self.ct = CentroidTracker(maxDisappeared=50, maxDistance=100)  # TODO: add parms to config
        self.sg = SpyGlass(self.dimensions, self.cfg)
        self.motionRect = None
        self.lenstype = "motion"
        self.event_start = datetime.utcnow()
        self._tick = 0
        self._looks = 0
        # when configured, start at most one instance each of log and video publishing
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
            self._rate.update()
            #ret_code, jpg_buffer = cv2.imencode(".jpg", image, 
            #    [int(cv2.IMWRITE_JPEG_QUALITY), camera.jpeg_quality])
            jpg_buffer = simplejpeg.encode_jpeg(image, 
                quality=camera.jpeg_quality, 
                colorspace='BGR')
            Outpost.publisher.send_jpg(camera.text, jpg_buffer)

        rects = []  # new frame, no determinations made
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
            self.motionRect = self.sg.detect_motion(gray)
            if self.motionRect:
                self._looks += 1
                self.lenstype = "detect"
                logging.debug(f"have motion, looks={self._looks}")
            self.sg.state = "inactive"
        else:
            self.motionRect = None

        if targets > 0 or self.motionRect:
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
                result = self.sg.get_data()
                rects = result[0]
                labels = result[1] if self.lenstype == "detect" else []
                logging.debug(f"LensTasking '{self.lenstype}' result: {len(rects)} objects, tick {self._tick}")
                # note the time whenever there is a result set to work with
                if len(rects) > 0:
                    self.lenstype = "track"
                    self.sg.lastUpdate = datetime.utcnow()     

                # send current frame on out to the SpyGlass for processing
                logging.debug(f"sending '{self.lenstype}' to LensTasking, tick {self._tick}, look {self._looks}")
                self.sg.apply_lens(self.lenstype, image)

                # work through current SpyGlass result set now, if any 
                if len(rects) > 0:
                    # Centroid tracking algorithm courtesy of PyImageSearch.
                    # Using this to map tracked object centroids back to a  
                    # dictionary of targets managed by the SpyGlass
                    centroids = self.ct.update(rects)
                    for i, (objectID, centroid) in enumerate(centroids.items()):

                        # grab the SpyGlass target via its object ID
                        target = self.sg.get_target(objectID)
                                    
                        # create new targets for tracking as needed
                        if target is None:
                            classname = labels[i].split(' ')[0][:-1] if i < len(labels) else 'unknown'
                            targetText = "_".join([classname, str(objectID)])
                            target = self.sg.new_target(objectID, classname, targetText)
                                    
                        rect = rects[i] if i<len(rects) else (0,0,0,0)  # TODO: fix this stupid hack
                        target.update_geo(rect, centroid, self.lenstype, self.sg.lastUpdate)
                        logging.debug(f"update_geo:{target.toJSON()}")
                                
                    # drop vanished objects from SpyGlass 
                    for target in self.sg.get_targets():
                        if target.objectID not in self.ct.objects.keys():
                            self.sg.drop_target(target.objectID)       
                            logging.debug(f"dropped Target {target.objectID}")
                    
                    targets = self.sg.get_count()
                    if targets > 0:
                        # SpyGlass has objects in view, discard motion rectangle ?
                        #self.motionRect = None
                        if self.sg.state == 'inactive':
                            # This is a new event, report and start logging
                            self.sg.state = 'active'
                            self.sg.eventID = uuid.uuid1().hex
                            self.ote = {'id': self.sg.eventID, 'evt': 'start',
                                'view': self.viewname, 'fps': self._rate.fps()}
                            logging.info(f"ote{json.dumps(self.ote)}")
                            self.event_start = datetime.utcnow()
                            del self.ote['fps']  # only needed for start event

                        if self.sg.state == 'active':
                            # event in progress
                            self.ote['evt'] = 'trk' 
                            for target in self.sg.get_targets():
                                if target.upd == self.sg.lastUpdate:
                                    self.ote['obj'] = target.objectID
                                    self.ote['class'] = target.classname
                                    self.ote['rect'] = [int(target.rect[0]), int(target.rect[1]), 
                                                        int(target.rect[2]), int(target.rect[3])]
                                    logging.info(f"ote{json.dumps(self.ote)}")

                elif self.lenstype == "detect" and self.sg.state == "inactive":
                    # current state is inactive, and object detection found nothing
                    # assume this was a false alarm, and resume motion detection
                    logging.debug(f"revert to motion, tick {self._tick}, look {self._looks}")
                    self.lenstype = "motion"

            else:
                # Have a new frame for review, but SpyGlass is busy. Ignore this instance,
                # releasing to imagenode pipeline without further analysis. No updates provided.
                logging.debug(f"frame skip, tick {self._tick}, look {self._looks}")

        # outpost tick count 
        self._tick += 1  
        if self._tick % self.skip_frames == 0:
            # tracking threshold encountered? run detection again
            if self.lenstype == "track":
                self.lenstype = "detect"

        stayalive = True   # assume there is still work to do
        # If an event is in progress, is it time to end it?
        if self.sg.state == 'active':
            if targets == 0:
                stayalive = False
                self.lenstype = "motion"
                self.sg.state = "inactive"
                logging.debug(f"Ending active event {self.sg.eventID}")
            else:
                # fail safe kill switch, forced shutdown after 15 seconds 
                # TODO: design flexibility for this via ruleset in configuration?
                #  ----------------------------------------------------------
                event_elapsed = datetime.utcnow() - self.event_start
                if event_elapsed.seconds > 15:
                    stayalive = False
                    self.sg.state = "quiet"
                    logging.debug(f"Status is quiet, ending event {self.sg.eventID}")
                    # TODO: placeholder for something more clever
                    # For now, just clear the SpyGlass and go back to motion detection
                    for target in self.sg.get_targets():
                        self.sg.drop_target(target.objectID)   
                    self.lenstype = "motion"
                    self.sg.state = "inactive"

            if not stayalive:
                self.ote['evt'] = 'end'
                #del self.ote['obj']
                #del self.ote['class']
                #del self.ote['rect']
                logging.info(f"ote{json.dumps(self.ote)}")

        if targets == 0:
            if self.motionRect:
                # Have motion but no object yet detected. Perhaps this
                # should be reported and captured for review and analysis?
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
    