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
from zmq.log.handlers import PUBHandler
from sentinelcam.utils import FPS
from sentinelcam.centroidtracker import CentroidTracker

class Outpost:
    """ Methods and attributes for the SentinelCam outpost, including 
    video publishing, and specialized trackers.

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
        # instantiate centroid tracker
        self.ct = CentroidTracker(maxDisappeared=15, maxDistance=100)
        # initialize the MOG foreground background subtractor
        self.mog = cv2.bgsegm.createBackgroundSubtractorMOG()
        self._rate = FPS()
        # configuration items
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
            handoff = {'node': self.nodename, 'view': self.viewname,  
                       'log': self.publish_log, 'video': self.publish_cam,
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
        """ Detect if ROI is 'moving' or 'still'; send _event message and images

        Parameters:
            camera (Camera object): current camera
            image (OpenCV image): current image
            send_q (Deque): where (text, image) tuples are appended to be sent

        """
        if self.publish_cam:
            self._rate.update()
            ret_code, jpg_buffer = cv2.imencode(".jpg", image, 
                [int(cv2.IMWRITE_JPEG_QUALITY), camera.jpeg_quality])
            Outpost.publisher.send_jpg(camera.text, jpg_buffer)

        # initialize a list to store the bounding box rectangles returned
        # by background subtraction model
        rects = []
        state = self.detector.current_state

        # crop to ROI
        x1, y1 = self.detector.top_left
        x2, y2 = self.detector.bottom_right
        ROI = image[y1:y2, x1:x2]

        # convert to grayscale and smoothen using gaussian kernel
        gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # apply the MOG background subtraction model
        mask = self.mog.apply(gray)

        # apply a series of erosions to break apart connected
        # components, then find contours in the mask
        erode = cv2.erode(mask, (7, 7), iterations=2)
        cnts = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over each contour
        for c in cnts:
            # if the contour area is less than the minimum area
            # required then ignore the object
            if cv2.contourArea(c) < 1000:
                continue

            # compute the bounding box coordinates of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            (startX, startY, endX, endY) = (x, y, x + w, y + h)

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))
        
        if len(rects) > 0:
            state = "moving"
            if state != self.detector.last_state:
                self._event = uuid.uuid1().hex
                self.ote = {'id': self._event, 'evt': 'start',
                    'view': camera.viewname, 'fps': self._rate.fps()}
                logging.info('ote' + json.dumps(self.ote))
                del self.ote['fps']  # only needed for start _event

        # update centroid tracker and loop over the tracked objects
        objects = self.ct.update(rects)
        if len(objects) > 0:
            self.ote['evt'] = 'trk' 
            for (objectID, centroid) in objects.items():
                self.ote['obj'] = int(objectID)
                self.ote['cent'] = [int(centroid[0] + x1), int(centroid[1] + y1)]
                logging.info('ote' + json.dumps(self.ote))
        
        if len(self.ct.objects) == 0:
            state = "still"
            if self.detector.last_state == "moving":
                self.ote['evt'] = 'end'
                logging.info('ote' + json.dumps(self.ote))

        self.detector.current_state = state
        # Now that current state has been sent, it becomes the last_state
        self.detector.last_state = self.detector.current_state
