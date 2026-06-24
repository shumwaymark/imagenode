"""spyglass: A concurrent image analysis pipeline for the SentinelCam Outpost

Since the §4.10 closeout the SpyGlass is a DETECT-only inference engine: it runs
motion detection (the NN scheduler) and a single object-detection lens in a
non-blocking child process over a single-frame shared-memory buffer. Object
identity, tracking, and event lifecycle live in the persistent host-side tracker
(hosttracker.py) + event manager (eventmanager.py), NOT here. The legacy
correlation-tracking cascade (dlib/cv2 trackers, CentroidTracker, the Target /
new_event / trackingLog scene-management surface) was retired in §4.10.

Copyright (c) 2021 by Mark K Shumway, mark.shumway@swanriver.dev
License: MIT, see the SentinelCam LICENSE for more details.
"""

import traceback
import cv2
import numpy as np
import multiprocessing
from multiprocessing import sharedctypes
from datetime import datetime
from time import sleep
import imagezmq
import msgpack
import zmq
from sentinelcam.lenses import LensMotion, LensYOLOv3, LensMobileNetSSD

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

    OBJECT_DETECTORS = {
        'yolov3'       : LensYOLOv3,
        'mobilenetssd' : LensMobileNetSSD
    }
    @staticmethod
    def lens_factory(lenstype, cfg):
        if lenstype == LensTasking.Request_DETECT:
            detect = cfg["detectobjects"]
            accelerator = cfg.get("accelerator", "none")
            return LensTasking.OBJECT_DETECTORS[detect](cfg[detect], accelerator=accelerator)

    def __init__(self, camsize, cfg) -> None:
        dtype = np.dtype('uint8')
        shape = (camsize[1], camsize[0], 3)
        self._frameBuffer = sharedctypes.RawArray('c', shape[0]*shape[1]*shape[2])
        self._wire = LensWire(LensTasking.LENS_WIRE)
        self.process = multiprocessing.Process(target=self._taskLoop, args=(
            self._frameBuffer, dtype, shape, cfg))
        self.process.start()
        handshake = self._wire.recv()  # wait on handshake from subprocess
        self._sharedFrame = np.frombuffer(self._frameBuffer, dtype=dtype).reshape(shape)
        self._wire.send(handshake)  # send it right back to prime the pump

    def _taskLoop(self, framebuff, dtype, shape, cfg):
        outpost = None
        od = None
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
            print("LensTasking started.")

            # Ignoring the first exception, just for a little dev sanity. See syslog for traceback.
            while exceptionCount < LensTasking.FAIL_LIMIT:

                # Task result is a tuple: the lens command and a list of structured
                # detections, each (class_name, (x1,y1,x2,y2), confidence). The host
                # tracker owns identity/tracking, so SpyGlass only ever runs DETECT.
                result = (0, [])
                try:
                    # wait on a lens command from the Outpost
                    lens = msgpack.unpackb(outpost_recv())

                    if lens == LensTasking.Request_DETECT and od is not None:
                        # Run object detection -> structured detection list
                        result = (lens, od.detect(frame))

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
            print(f"LensTasking ended")
            if outpost is not None:
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

class SpyGlass:
    """ A DETECT-only inference engine for the SentinelCam Outpost.

    The SpyGlass wraps LensTasking (the forked detection child process) with
    motion detection and convenience access to the single-frame analysis IPC.
    Identity, tracking, and event lifecycle are owned by the persistent
    host-side tracker + event manager (§4.10), not by the SpyGlass.

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
        spyglass has results available
    get_data() -> tuple
        retrieve (lens, detections) from spyglass
    apply_lens(lenstype, image, frametime) -> None
        send frame to spyglass for analysis, with lens type to use
    detect_motion(image) -> tuple
        return a rectangle for the aggregate area of motion from the
        background subtraction model
    terminate() -> None
        kill the LensTasking subprocess. Be courteous and call this as a
        part of imagenode shutdown
    """
    State_BUSY = 0
    State_RESULT = 1

    State = ["SpyGlass is busy", "SpyGlass has result"]

    def __init__(self, view, camsize, cfg) -> None:
        self.CFG = cfg
        self._tasking = LensTasking(camsize, cfg)
        # Pass motion_params from config if available
        self._motion = LensMotion(cfg.get('motion_params'))
        self.view = view
        self.state = SpyGlass.State_BUSY
        self.sgTime = datetime.now()
        self.frametime = self.sgTime

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

    def detect_motion(self, image) -> tuple:
        return self._motion.detect(image)

    def terminate(self):
        self._tasking.terminate()

    def __del__(self) -> None:
        self.terminate()
