"""spyglass: A concurrent image analysis pipeline for the SentinelCam Outpost

Copyright (c) 2021 by Mark K Shumway, mark.shumway@swanriver.dev
License: MIT, see the SentinelCam LICENSE for more details.
"""

import os
import json
import logging
import traceback
import uuid
import cv2
import numpy as np
import multiprocessing
from multiprocessing import sharedctypes
from datetime import datetime
from time import sleep
import dlib
import imagezmq
import msgpack
import zmq
from sentinelcam.lenses import LensMotion, LensYOLOv3, LensMobileNetSSD, CentroidTracker

class LensWire:
    def __init__(self, ipcname) -> None:
        self._wire = imagezmq.ImageHub(f"ipc://{ipcname}")
        self._poller = zmq.Poller()
        self._poller.register(self._wire.zmq_socket, zmq.POLLIN)
        self._send = self._wire.zmq_socket.send
        self._recv = self._wire.zmq_socket.recv

    def ready(self) -> bool:
        events = dict(self._poller.poll(0))
        if self._wire.zmq_socket in events:
            return events[self._wire.zmq_socket] == zmq.POLLIN
        else:
            return False
    
    def send(self, lenstype) -> None:
        self._send(msgpack.packb(lenstype))
    
    def recv(self) -> tuple:
        return msgpack.unpackb(self._recv(), use_list=False)

    def __del__(self) -> None:
        self._wire.close()

class LensTasking:

    FAIL_LIMIT = 2
    LENS_WIRE = "/tmp/SpyGlass306"

    Request_DETECT = 1
    Request_TRACK = 2

    OBJECT_DETECTORS = {
        'yolov3'       : LensYOLOv3,
        'mobilenetssd' : LensMobileNetSSD
    }
    def lens_factory(lenstype, cfg):
        if lenstype == LensTasking.Request_DETECT:
            detect = cfg["detectobjects"]
            return LensTasking.OBJECT_DETECTORS[detect](cfg[detect])

        elif lenstype == LensTasking.Request_TRACK:
            if cfg['tracker'] == 'dlib':
                # This conditional is required for operation under OpenVINO, which
                # does not include support for the legacy contributed trackers below
                return dlib.correlation_tracker()
            else:
                # This dictionary maps strings to their corresponding (now
                # legacy) OpenCV contributed object tracker implementations
                OPENCV_OBJECT_TRACKERS = {
                    "csrt": cv2.TrackerCSRT_create,
                    "kcf": cv2.TrackerKCF_create,
                    "boosting": cv2.TrackerBoosting_create,
                    "mil": cv2.TrackerMIL_create,
                    "tld": cv2.TrackerTLD_create,
                    "medianflow": cv2.TrackerMedianFlow_create,
                    "mosse": cv2.TrackerMOSSE_create
                }
                return OPENCV_OBJECT_TRACKERS[cfg["tracker"]]()

    def __init__(self, camsize, cfg) -> None:
        self._dlib = cfg['tracker'] == 'dlib'
        self._trkrs = None
        dtype = np.dtype('uint8')
        shape = (camsize[1], camsize[0], 3)
        self._frameBuffer = sharedctypes.RawArray('c', shape[0]*shape[1]*shape[2])
        self._wire = LensWire(LensTasking.LENS_WIRE)
        self.process = multiprocessing.Process(target=self._taskLoop, args=(
            self._frameBuffer, dtype, shape, cfg))
        self.process.start()
        handshake = self._wire.recv()  # wait on handshake from subprocess
        self._sharedFrame = np.frombuffer(self._frameBuffer, dtype=dtype).reshape(shape)
        self._wire.send(handshake)  # send it right back to prime the pmup

    def _trackers(self):
        if self._dlib:
            return []
        else:
            return cv2.MultiTracker_create()
    
    def _track_this(self, trkr, frame, x1, y1, x2, y2):
        if self._dlib:
            # convert the frame from BGR to RGB for dlib
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rect = dlib.rectangle(x1, y1, x2, y2)
            trkr.start_track(rgb, rect)
            self._trkrs.append(trkr)
        else:
            self._trkrs.add(trkr, frame, (x1, y1, x2-x1, y2-y1))

    def _update_trackers(self, frame):
        rects = []
        if self._dlib:
            # convert the frame from BGR to RGB for dlib
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # loop over the trackers
            for tracker in self._trkrs:
                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()
                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))
        else:
            # Update object trackers
            (success, boxes) = self._trkrs.update(frame)
            # Loop over the bounding boxes and convert to an (x1, y1, x2, y2) list
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                rects.append((x, y, x+w, y+h))
        # return results
        return rects

    def _taskLoop(self, framebuff, dtype, shape, cfg):
        try:
            exceptionCount = 0
            frame = np.frombuffer(framebuff, dtype=dtype).reshape(shape)
            outpost = imagezmq.ImageSender(f"ipc://{LensTasking.LENS_WIRE}")
            outpost_send = outpost.zmq_socket.send
            outpost_recv = outpost.zmq_socket.recv
            outpost_send(msgpack.packb(0))  # handshake
            
            if not cfg["detectobjects"] in ["none","motion"]:
                od = LensTasking.lens_factory(LensTasking.Request_DETECT, cfg)
                sleep(3.0)
            if cfg['tracker'] == "none":
                self._doTracking = False
            else:
                self._doTracking = True
                self._trkrs = self._trackers()
            print("LensTasking started.")
            
            # Ignoring the first exception, just for a little dev sanity. See syslog for traceback.
            while exceptionCount < LensTasking.FAIL_LIMIT:  
 
                # Task result is a tuple with the lens command, a list of rectangles, and a list of labels
                result = (0, [], [])
                try: 
                    # wait on a lens command from the Outpost
                    lens = msgpack.unpackb(outpost_recv())

                    if lens == LensTasking.Request_DETECT:
                        # Run object detection 
                        (rects, labels) = od.detect(frame)
                        rects_out = []
                        if self._doTracking:
                            # Populate new trackers with objects found, if any
                            self._trkrs = self._trackers()
                        for (x1, y1, x2, y2) in rects:
                            rects_out.append((int(x1), int(y1), int(x2), int(y2)))
                            if self._doTracking:
                                tracker = LensTasking.lens_factory(LensTasking.Request_TRACK, cfg)
                                self._track_this(tracker, frame, x1, y1, x2, y2)
                        result = (lens, rects_out, labels)
                
                    elif lens == LensTasking.Request_TRACK:
                        rects = self._update_trackers(frame)
                        result = (lens, rects, None)
                    
                except (KeyboardInterrupt, SystemExit):
                    print("LensTasking shutdown.")
                    exceptionCount = LensTasking.FAIL_LIMIT  # allow shutdown to continue
                except cv2.error as e:
                    print(f"OpenCV error trapped: {str(e)}")
                except Exception as ex:
                    exceptionCount += 1
                    print(f"LensTasking failure #{exceptionCount}.")
                    traceback.print_exc()
                finally:
                    # always reply to the Outpost
                    outpost_send(msgpack.packb(result))

        except (KeyboardInterrupt, SystemExit):
            print("LensTasking ending.")
        except Exception as ex:
            print("LensTasking failure.")
            traceback.print_exc()
        finally:
            print(f"LensTasking ended with exceptionCount={exceptionCount}.")
            outpost.close()

    def apply_lens(self, lens, frame) -> None:
        self._sharedFrame[:] = frame[:]  # np.copyto(self._sharedFrame, frame)
        self._wire.send(lens)

    def is_ready(self) -> bool:
        return self._wire.ready()
    
    def get_result(self) -> tuple:
        return self._wire.recv()
    
    def terminate(self) -> None:
        if self.process.is_alive():
            self.process.kill()
            self.process.join()

class Target:
	def __init__(self, objid, classname, label ) -> None:
		self.objectID = objid
		self.rect = (0,0,0,0)
		self.cent = (0,0)
		self.classname = classname
		self.source = 'lens'
		self.text = label
	def update_geo(self, rect, cent, source, wen) -> None:
		self.rect = rect 
		self.cent = cent 
		self.source = source
		self.upd = wen  # a datetime.utcnow() equivalent is expected here
	def toJSON(self) -> str:
		return json.dumps({
			'obj': self.objectID,
			'rect': (int(self.rect[0]),int(self.rect[1]),int(self.rect[2]),int(self.rect[3])),
			'cent': (int(self.cent[0]),int(self.cent[1])),
			'clas': self.classname,
			'src': self.source,
			'tag': self.text,
			'upd': self.upd.isoformat()
		})
	def toTrk(self) -> dict:
		return {'obj': self.objectID,
                'clas': self.classname,
                'rect': (int(self.rect[0]),int(self.rect[1]),int(self.rect[2]),int(self.rect[3]))
        }

class SpyGlass:
    """ The SpyGlass is a construct conceieved as an event and state
    management scratchpad for tracking objects within the current view,
    and directing the image analysis pipeline in use.

    A high-level wishlist of data collected for logging follows. Not all 
    the below has been implemented. Parts of this should be delegated to 
    batch processing by the Sentinel.

    - state/status (active/inactive/quiet/changing)
    - timestamp of last actitivy
    - current or last event id
    - current lens (detect/track)
    - objects tracked (aka Targets) 
        - object id as dictionary key
        - lens type (detectObjects/detectFaces/etc)
        - timestamp of last update
        - source of update (lens/tracker/dropped)
        - bounding rectangle within the view
        - Z-coordinate(s) within the view
        - geometric centroid within view
        - status? (still, vanished, in motion / direction of travel, velocity?)
        - classification
        - identification
        - confidence
        - label / text comment
        - color (for drawing rectangles on images)
    
    SpyGlass methods are primarly wrappers for LensTasking along with 
    convenience access to the list of Targets.

    Internal use only, one instance per Outpost view.

    Parameters
    ----------
    view : str  
        imagenode camera view name
    camsize : tuple
        image size (width, height) tuple
    cfg : dict
        configuration dictionary for LensTasking

    Methods
    -------
    has_result() -> bool
        spyglass has results avalable
    get_data() - > tuple
        retrieve results from spyglass
    apply_lens(lenstype, frame) -> None
        send frame to spyglass for analysis, with lens type to use
    new_target(objid, rect, classname, label) -> Target
        create a new trackable target
	get_target(objid) -> Target
		return spyglass target by object ID
    drop_target(objid) -> None
        delete a tracked object by object ID
    update_target_geo(objid, rect, cent, source, datetime) -> None
        update the tracking coordinates for a target by ID, with source of data
    get_count() -> int
        return total number of targets being tracked
	get_targets() -> list
		return list of targets being tracked
    detect_motion(image) -> list
        return a rectangle for aggregate area of motion from background subtraction model
    new_event() -> dict
        indicate start of new event and return dictionary with logging information
    trackingLog(type) -> dict
        return current event logging record dictionary for specifed type ['trk','end']
    terminate() -> None
        kill the LensTasking subprocess. Be courteous and call this as a part of imagenode shutdown
    """
    State_BUSY = 0
    State_RESULT = 1 

    State = ["SpyGlass is busy", "SpyGlass has result"]

    def __init__(self, view, camsize, cfg) -> None:
        self._tasking = LensTasking(camsize, cfg)
        self._motion = LensMotion()
        self._ct = CentroidTracker(maxDisappeared=3, maxDistance=100)  # TODO: add parms to config
        self._dropList = {}  # unwanted objects
        self._targets = {}   # dictionary of Targets by objectID
        self._logdata = {}   # tracking event data for logging
        self.eventID = None
        self.view = view
        self.state = SpyGlass.State_BUSY
        self.sgTime = datetime.utcnow()
        self.frametime = self.sgTime
        self.lastUpdate = self.sgTime
    
    def has_result(self) -> bool:
        if self._tasking.is_ready():
            self.state = SpyGlass.State_RESULT
        else:
            self.state = SpyGlass.State_BUSY
        return self.state == SpyGlass.State_RESULT
    
    def get_state(self) -> int:
        return self.state
    
    def get_data(self) -> tuple:
        self.frametime = self.sgTime
        return self._tasking.get_result()
    
    def get_frametime(self) -> datetime:
        return self.frametime

    def apply_lens(self, lenstype, image, frametime) -> None:
        self._tasking.apply_lens(lenstype, image)
        self.sgTime = frametime

    def new_target(self, item, classname, label) -> Target:
        self._targets[item] = Target(item, classname, label)
        return self._targets[item]

    def get_target(self, item) -> Target:
        return self._targets.get(item, None)

    def update_target_geo(self, item, rect, cent, source, wen) -> None:
        if item in self._targets:
            self._targets[item].update_geo(rect, cent, source, wen)
    
    def drop_target(self, item) -> None:
        if item in self._targets:
            del self._targets[item]

    def get_count(self) -> int:
        return len(self._targets)
    
    def get_targets(self) -> list:
        return list(self._targets.values())

    def detect_motion(self, image) -> tuple:
        return self._motion.detect(image)

    def new_event(self) -> dict:
        self.eventID = uuid.uuid1().hex
        self.event_start = datetime.utcnow()
        self._logdata = {'id': self.eventID, 'view': self.view, 'type': 'start', 'new': True}
        return self._logdata

    def trackingLog(self, type) -> dict:
        self._logdata = {'id': self.eventID, 'view': self.view, 'type': type}
        return self._logdata

    def terminate(self):
        self._tasking.terminate()

    def __del__(self) -> None:
        self.terminate()
    
    def resetTargetList(self):
        for target in self.get_targets():
            self.drop_target(target.objectID)   
        trkdObjs = list(self._ct.objects.keys())
        for o in trkdObjs:
            self._ct.deregister(o)
        self._dropList = {}

    def reviseTargetList(self, lens, rects, labels) -> tuple:
        # Centroid tracking algorithm courtesy of PyImageSearch.
        # Using this to map tracked object centroids back to a  
        # dictionary of targets managed by the SpyGlass
        centroids = self._ct.update(rects)

        # TODO: Need to validate CentroidTracker initilization and overall
        # fit within the context of the Outpost use cases. Specifically the
        # max disappeared limit. The real gap in thinking is likely the assumption
        # that an ocurrence number from the list of rects provides for a reliable 
        # mapping to objectID occurences coming out of the Centroid Tracker. 
        # Too many edge cases around this approach. Not a robust solution. 
        newTarget = False
        interestingTargetFound = False
        self.lastUpdate = datetime.utcnow()
        for i, (objectID, centroid) in enumerate(centroids.items()):

            # Ignore anything on the drop list
            if objectID in self._dropList:
                continue

            # Grab the SpyGlass target via its object ID
            target = self.get_target(objectID)

            # Create new targets for tracking as needed. 
            if target is None:
                newTarget = True
                classname = labels[i].split(' ')[0][:-1] if labels and i < len(labels) else 'mystery'
                targetText = "_".join([classname, str(objectID)])
                target = self.new_target(objectID, classname, targetText)

            rect = rects[i] if i<len(rects) else (0,0,0,0)  # TODO: fix this stupid hack? (serves as a fail-safe)
            target.update_geo(rect, centroid, lens, self.lastUpdate)
            logging.debug(f"update_geo:{target.toJSON()}")

        for target in self.get_targets():

            # Drop vanished objects from SpyGlass 
            if target.objectID not in self._ct.objects.keys():
                logging.debug(f"Target {target.objectID} vanished")
                self.drop_target(target.objectID)

            # when does it get interesting?
            elif target.classname == "person":
                interestingTargetFound = True

        return (newTarget, interestingTargetFound)