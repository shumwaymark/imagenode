"""lenses: Lens definitions for the sentinel SpyGlass

Fundamental components of the underlying object detection code within are
courtesy of Dr. Adrian Rosebrock and the team at PyImageSearch.

Detection contract (§4.10): every object-detection lens returns a flat list of
``Detection`` tuples ``(class_name: str, bbox: (x1,y1,x2,y2) pixels, confidence:
float)``. This is the same shape the OAK device detections carry, so the picamera
and OAK intakes converge on one decode model. Display labels ("car: 0.96") are
built downstream from this structured form, not here.

Copyright (c) 2021 by Mark K Shumway, mark.shumway@swanriver.dev
License: MIT, see the sentinelcam LICENSE for more details.
"""

import os
import cv2
import imutils
import numpy as np
from PIL import Image

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
    def __init__(self, motion_params=None) -> None:
        # Use provided params or fall back to hardcoded defaults
        # motion_params can contain: varThreshold, detectShadows, history,
        # minContourW, minContourH, gaussianBlur
        if motion_params:
            varThreshold = motion_params.get('varThreshold', 128)
            detectShadows = motion_params.get('detectShadows', False)
            history = motion_params.get('history', 500)
            self.minContourW = motion_params.get('minContourW', 50)
            self.minContourH = motion_params.get('minContourH', 50)
            self.gaussianBlur = motion_params.get('gaussianBlur', 5)
        else:
            # Hardcoded defaults (original behavior)
            varThreshold = 128
            detectShadows = False
            history = 500
            self.minContourW = 50
            self.minContourH = 50
            self.gaussianBlur = 5

        self.mog = cv2.createBackgroundSubtractorMOG2(
            varThreshold=varThreshold,
            detectShadows=detectShadows,
            history=history)

    def detect(self, image) -> tuple:
        # Apply gaussian blur for noise reduction (configurable kernel size)
        blurred = cv2.GaussianBlur(image, (self.gaussianBlur, self.gaussianBlur), 0)

        # Apply the MOG background subtraction model
        mask = self.mog.apply(blurred)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            if w >= self.minContourW and h >= self.minContourH:
                (minX, minY) = (min(minX, x), min(minY, y))
                (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

        # return a tuple with the bounding box
        return (minX, minY, maxX, maxY)   # TODO: apply configurable filter on aggregate size?

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

    def detect(self, frame) -> list:
        # structured detections: list of (class_name, (x1,y1,x2,y2), confidence)
        dets = []

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

                # store the detection in (class_name, (x1,y1,x2,y2), conf) form
                dets.append((self.LABELS[classIDs[i]],
                             (int(x), int(y), int(x + w), int(y + h)),
                             float(confidences[i])))

        return dets

class LensMobileNetSSD:

    LENS_opencvDNN = 0
    LENS_edgetpu = 1

    def __init__(self, conf, accelerator="cpu") -> None:
        self.conf = conf  # configuration dictionary
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        (self.W, self.H) = (None, None)

        # load our serialized model from disk
        print("Loading MobileNetSSD model...")
        if accelerator == "coral":
            self.lens_type = LensMobileNetSSD.LENS_edgetpu
            from pycoral.utils.edgetpu import make_interpreter
            from pycoral.utils.dataset import read_label_file
            # create interpreter for EdgeTPU model
            self.net = make_interpreter(self.conf["edgetpu_model"])
            self.net.allocate_tensors()
            # load labels for model
            self.labels = read_label_file(self.conf["model_labels"])
            # import required modules for EdgeTPU inference
            from pycoral.adapters import common
            from pycoral.adapters import detect as edgetpu_detect
            # store imported modules for later use
            self.common = common
            self.edgetpu_detect = edgetpu_detect
        else:
            self.lens_type = LensMobileNetSSD.LENS_opencvDNN
             # load our serialized model from disk
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

    def detect(self, frame) -> list:
        # structured detections: list of (class_name, (x1,y1,x2,y2), confidence)
        dets = []

        if self.lens_type == LensMobileNetSSD.LENS_edgetpu:
            # prepare the frame for object detection
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            _, scale = self.common.set_resized_input(
                self.net, image.size, lambda size: image.resize(size, Image.LANCZOS))
            self.net.invoke()
            detections = self.edgetpu_detect.get_objects(self.net, self.conf["confidence"], scale)
            for detection in detections:
                # extract the bounding box coordinates
                bbox = detection.bbox
                dets.append((self.labels[detection.id],
                             (int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)),
                             float(detection.score)))
        else:
            H, W = frame.shape[:2]
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
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (x1, y1, x2, y2) = box.astype("int")
                    dets.append((self.CLASSES[idx],
                                 (int(x1), int(y1), int(x2), int(y2)),
                                 float(confidence)))

        return dets
