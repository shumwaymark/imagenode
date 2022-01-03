"""spyglass: image analysis pipeline for sentinelcam outpost

Fundamental components of the underlying object detection and tracking 
code within, most especially CentroidTracker, are courtesy of Dr. Adrian 
Rosebrock and the team at PyImageSearch.

Copyright (c) 2021 by Mark K Shumway, mark.shumway@swanriver.dev
License: MIT, see the sentinelcam LICENSE for more details.
"""

import os
import json
import traceback
import cv2
import imutils
import numpy as np
import multiprocessing
from datetime import datetime
from multiprocessing import sharedctypes
from collections import OrderedDict
from scipy.spatial import distance as dist

class ParseYOLOOutput:
	def __init__(self, conf):
		# store the configuration file
		self.conf = conf

	def parse(self, layerOutputs, LABELS, H, W):
		# initialize our lists of detected bounding boxes,
		# confidences, and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID
				scores = detection[5:]
				classID = np.argmax(scores)

				# check if the class detected should be considered,
				# if not, then skip this iteration
				if LABELS[classID] not in self.conf["consider"]:
					continue

				# retrieve the confidence (i.e., probability) of the
				# current object detection
				confidence = scores[classID]

				# filter out weak predictions by ensuring the
				# detected probability is greater than the minimum
				# probability
				if confidence > self.conf["confidence"]:
					# scale the bounding box coordinates back
					# relative to the size of the image, keeping in
					# mind that YOLO actually returns the center
					# (x, y)-coordinates of the bounding box followed
					# by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					box = box.astype("int")
					(centerX, centerY, width, height) = box

					# use the center (x, y)-coordinates to derive the
					# top and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates,
					# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# return the detected boxes and their corresponding
		# confidences and class IDs
		return (boxes, confidences, classIDs)

class LensMotion:
    def __init__(self) -> None:
        self.mog = cv2.bgsegm.createBackgroundSubtractorMOG()
    
    def detect(self, image) -> tuple:
        # apply the MOG background subtraction model
        mask = self.mog.apply(image)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (minX, minY) = (image.shape[1], image.shape[0])
        (maxX, maxY) = (0, 0)
        
        # if no contours were found, return False 
        if len(cnts) == 0:
            return None

		# otherwise, loop over the contours
        for c in cnts:
			# compute the bounding box of the contour and use it to
			# update the minimum and maximum bounding box of the region
            (x, y, w, h) = cv2.boundingRect(c)
            if w>20 and h>20:  # apply a size threshold for noise suppression
                (minX, minY) = (min(minX, x), min(minY, y))
                (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

		# return a tuple with the bounding box
        return (minX, minY, maxX, maxY)

class LensYOLOv3:
    def __init__(self, conf) -> None:
        self.conf = conf  # configuration dictionary
        (self.W, self.H) = (None, None)

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([conf["yolo_path"], "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([conf["yolo_path"], "yolov3.weights"])
        configPath = os.path.sep.join([conf["yolo_path"], "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        print("Loading YOLOv3 from disk...")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # initialize the YOLO output parsing object
        self.pyo = ParseYOLOOutput(conf)

    def detect(self, frame) -> tuple:
        # initialize output lists
        objs = []
        labls = []

        # if we do not already have the dimensions of the frame,
        # initialize it
        if self.H is None and self.W is None:
            (self.H, self.W) = frame.shape[:2]

		# construct a blob from the input frame and then perform
		# a forward pass of the YOLO object detector, giving us
		# our bounding boxes and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0,
			(416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)
        
        # parse YOLOv3 output, thanks to the PyImageSearch team
        (boxes, confidences, classIDs) = self.pyo.parse(layerOutputs,
            self.LABELS, self.H, self.W)

        # apply non-maxima suppression to suppress weak,
        # overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 
            self.conf["confidence"], self.conf["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
			# loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                # store the coordinates of the detected
                # object in (x1, y1, x2, y2) format
                objs.append((x, y, x + w, y + h))
                labls.append("{}: {:.4f}".format(
                    self.LABELS[classIDs[i]],
                    confidences[i]))

        return (objs, labls)

class LensMobileNetSSD:
    def __init__(self, conf) -> None:
        self.conf = conf  # configuration dictionary
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        (self.W, self.H) = (None, None)

        # load our serialized model from disk
        print("Loading MobileNetSSD model...")
        self.net = cv2.dnn.readNetFromCaffe(self.conf["prototxt_path"],
	        self.conf["model_path"])

        # check if the target processor is myriad, if so, then set the
        # preferable target to myriad
        if self.conf["target"] == "myriad":
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
        else:
            # set the preferable target processor to CPU and preferable
            # backend to OpenCV
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    def detect(self, frame) -> tuple:
        # initialize output lists
        objs = []
        labls = []
        
        # check to see if the frame dimensions are not set
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]
        
        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
        self.net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5,
            127.5, 127.5])
        detections = self.net.forward()
        
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > self.conf["confidence"]:
                # extract the index from the detections list
                idx = int(detections[0, 0, i, 1])
                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array(
                    [self.W, self.H, self.W, self.H])
                objs.append(box.astype("int"))
                labls.append("{}: {:.4f}".format(
                    self.CLASSES[idx],
					confidence))

        return (objs, labls)

class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

		# store the maximum distance between centroids to associate
		# an object -- if the distance is larger than this maximum
		# distance we'll start to mark the object as "disappeared"
		self.maxDistance = maxDistance

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				if row in usedRows or col in usedCols:
					continue

				# if the distance between centroids is greater than
				# the maximum distance, do not associate the two
				# centroids to the same object
				if D[row, col] > self.maxDistance:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects

class LensTasking:

    FAIL_LIMIT = 2

    OBJECT_DETECTORS = {
        'yolov3'       : LensYOLOv3,
        'mobilenetssd' : LensMobileNetSSD
    }
    def lens_factory(lensname, cfg):
        return LensTasking.OBJECT_DETECTORS[lensname](cfg)

    # Initialize a dictionary that maps strings to their corresponding
    # (now legacy) OpenCV contributed object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
	    "csrt": cv2.TrackerCSRT_create,
	    "kcf": cv2.TrackerKCF_create,
	    "boosting": cv2.TrackerBoosting_create,
	    "mil": cv2.TrackerMIL_create,
	    "tld": cv2.TrackerTLD_create,
	    "medianflow": cv2.TrackerMedianFlow_create,
	    "mosse": cv2.TrackerMOSSE_create
    }

    def __init__(self, camsize, cfg) -> None:
        # TODO: validate keys retrieving lens tasking dictionary setups (detectors/trackers)
        dtype = np.dtype('uint8')
        shape = (camsize[1], camsize[0], 3)
        #spyGlassContext = multiprocessing.get_context("spawn")
        self._signal = multiprocessing.Event()
        self._ready = multiprocessing.Event()
        self._control, self._worker = multiprocessing.Pipe()
        self._frameBuffer = sharedctypes.RawArray('c', shape[0]*shape[1]*shape[2])
        self.process = multiprocessing.Process(target=self._taskLoop, args=(
            self._signal, self._ready, self._control, self._frameBuffer, dtype, shape, cfg))
        self.process.start()
        self._sharedFrame = np.frombuffer(self._frameBuffer, dtype=dtype).reshape(shape)
    
    def _taskLoop(self, signal, ready, control, framebuff, dtype, shape, cfg):
        try:
            frame = np.frombuffer(framebuff, dtype=dtype).reshape(shape)
            detect = cfg["detectobjects"]
            od = LensTasking.lens_factory(detect, cfg[detect])
            mt = cv2.MultiTracker_create()
            exceptionCount = 0
            print("LensTasking ready.")
            
            while exceptionCount < LensTasking.FAIL_LIMIT:

                result = ([],[])  # task result is a tuple with a list of rectangles and a list of labels
                try:
                    signal.wait()
                    lens = control.recv()

                    if lens == "detect":
                        (rects, labels) = od.detect(frame)
                        # populate a new multi-tracker with objects found, if any
                        mt = cv2.MultiTracker_create()
                        for (x1, y1, x2, y2) in rects:
                            tracker = LensTasking.OPENCV_OBJECT_TRACKERS[cfg["tracker"]]()
                            mt.add(tracker, frame, (x1, y1, x2-x1, y2-y1))
                        result = (rects, labels)
                
                    elif lens == "track":
                        # update object trackers
                        (success, boxes) = mt.update(frame)
                        # loop over the bounding boxes and convert to an (x1, y1, x2, y2) list
                        rects = []
                        for box in boxes:
                            (x, y, w, h) = [int(v) for v in box]
                            rects.append((x, y, x+w, y+h))
                        result = (rects, [])

                except (KeyboardInterrupt, SystemExit):
                    print("LensTasking shutdown.")
                    exceptionCount = LensTasking.FAIL_LIMIT  # allow shutdown to continue
                except Exception as ex:
                    exceptionCount += 1
                    print(f"LensTasking failure #{exceptionCount+1}.")
                    traceback.print_exc()
                finally:
                    signal.clear()
                    control.send(result)
                    ready.set()

        except (KeyboardInterrupt, SystemExit):
            print("LensTasking ending.")
        except Exception as ex:
            print("LensTasking failure.")
            traceback.print_exc()
        finally:
            print(f"LensTasking ended with exceptionCount={exceptionCount}.")

    def apply_lens(self, lens, frame) -> None:
        np.copyto(self._sharedFrame, frame)
        self._worker.send(lens)
        self._signal.set()
    
    # is work still in progress?
    def is_busy(self) -> bool:
        return self._signal.is_set()

    # are results available?
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def get_result(self) -> tuple:
        lensresult = self._worker.recv()
        self._ready.clear()
        return lensresult
    
    def terminate(self) -> None:
        if self.process.is_alive():
            self.process.kill()
            self.process.join()

    def __del__(self) -> None:
        self.terminate()

class Target:
	def __init__(self, objid, classname, label ) -> None:
		self.objectID = objid
		self.rect = (0,0,0,0)
		self.cent = (0,0)
		self.classname = classname
		self.source = 'lens'
		self.text = label
		self.color = tuple(np.random.randint(256, size=3))  # cannot pass to OpenCV.rectangle() like this :-/
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

class SpyGlass:
    """ The SpyGlass is a construct conceieved as an event and state
    management scratchpad for tracking objects within the current view,
    and directing the image analysis pipeline in use.

    A high-level wishlist of data collected for logging follows. Not all 
    the below has been implemented. Parts of this should be delegated to 
    batch processing by the Sentinel.

    - state (active/inactive/quiet/changing)
    - timestamp of last actitivy
    - current or last event id
    - current lens (detect/track)
    - objects tracked (aka Targets) 
        - object id as dictionary key
        - lens type (detectObjects/detectFaces/etc)
        - timestamp of last update
        - source of update (lens/tracker/dropped)
        - bounding rectangle within the view
        - geometric centroid within view
        - status? (still, vanished, in motion / direction of travel, velocity?)
        - classification
        - identification
        - label / text comment
        - color (for drawing rectangles on images)
    
    SpyGlass methods are primarly wrappers for LensTasking along with 
    convenience access to the list of Targets

    Parameters
    ----------
    Not applicable. Internal use only, one instance per Outpost view.

    Methods
    -------
    new_target(objid, rect, classname, label) -> Target
        create a new trackable target
	get_target(objid) -> Target
		return spyglass target by object ID
    update_target_geo(objid, rect, cent, source, datetime) -> None
        update the tracking coordinates for a target by ID, with source of data
    get_count() -> int
        return total number of targets being tracked
	get_targets() -> list
		return list of targets being tracked
    detect_motion(image) -> list
        return rectangles for areas of motion within image from backgroumd subtraction
    """
    def __init__(self, camsize, cfg) -> None:
        self._tasking = LensTasking(camsize, cfg)
        self._motion = LensMotion()
        self._targets = {}
        self.eventID = None
        self.state = "inactive"
        self.lastUpdate = datetime.utcnow()

    def lens_busy(self) -> bool:
        return self._tasking.is_busy()
    
    def has_data(self) -> bool:
        return self._tasking.is_ready()
    
    def get_data(self) -> tuple:
        return self._tasking.get_result()

    def apply_lens(self, lenstype, image) -> None:
        self._tasking.apply_lens(lenstype, image)

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

    def terminate(self):
        self._tasking.terminate()

    def __del__(self) -> None:
        self.terminate()