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
from time import monotonic, sleep
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

    def close(self) -> None:
        # Explicit, because a recycle must tear the wire down deterministically:
        # the REP socket is left mid-transaction (a request was sent, the reply
        # never came) and cannot be reused.
        self._wire.close()

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
        self._dtype = np.dtype('uint8')
        self._shape = (camsize[1], camsize[0], 3)
        self._cfg = cfg
        self._frameBuffer = sharedctypes.RawArray(
            'c', self._shape[0]*self._shape[1]*self._shape[2])
        self._wire = LensWire(LensTasking.LENS_WIRE)
        self._spawn()  # blocking handshake at startup, as it always was

    def _spawn(self, handshake_timeout=None) -> bool:
        """Fork the child and complete the priming handshake.

        The child publishes its handshake BEFORE loading the lens, so this returns
        promptly even when the model load behind it is slow. `handshake_timeout`
        bounds the wait for a recycle -- blocking the imagenode main loop on a
        child that may never come up would turn a detection outage into a total
        outpost stall, which is strictly worse than the fault being recovered.
        """
        self.process = multiprocessing.Process(target=self._taskLoop, args=(
            self._frameBuffer, self._dtype, self._shape, self._cfg))
        self.process.start()
        if handshake_timeout is not None:
            deadline = monotonic() + handshake_timeout
            while not self._wire.ready():
                if monotonic() > deadline:
                    return False
                sleep(0.05)
        handshake = self._wire.recv()  # wait on handshake from subprocess
        self._sharedFrame = np.frombuffer(
            self._frameBuffer, dtype=self._dtype).reshape(self._shape)
        self._wire.send(handshake)  # send it right back to prime the pump
        return True

    def recycle(self, handshake_timeout=10.0) -> bool:
        """Kill a wedged child and stand up a fresh one. Returns True on success.

        The only recovery available when the accelerator stops answering. Its
        blocking call sits in C holding the device fd, so nothing in-process can
        interrupt it -- killing the child is what releases the device, and the
        driver resets it on the next open. The shared frame buffer is plain memory
        and is reused; the wire is not, and must be rebuilt.
        """
        self.terminate()
        self._wire.close()
        self._wire = LensWire(LensTasking.LENS_WIRE)
        return self._spawn(handshake_timeout)

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
            self.process.kill()   # SIGKILL: a child blocked in a C call cannot
            self.process.join()   # service SIGTERM, so there is nothing gentler

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
        # Watchdog bookkeeping. _sent_at is monotonic on purpose: sgTime carries
        # the image CAPTURE time (anti-pattern #5) and is the wrong clock for
        # measuring how long a request has been outstanding.
        self._sent_at = monotonic()
        self._pending = True          # __init__'s handshake echo primes a request

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
        self._pending = False
        return self._tasking.get_result()

    def get_frametime(self) -> datetime:
        return self.frametime

    def apply_lens(self, lenstype, image, frametime) -> None:
        self._tasking.apply_lens(lenstype, image)
        self.sgTime = frametime
        self._sent_at = monotonic()
        self._pending = True

    def is_wedged(self, deadline) -> bool:
        """True when a request has gone unanswered past `deadline` seconds.

        Distinguishing a wedged child from an idle one is the whole trick. A
        healthy result may sit on the wire unconsumed for a long time -- while the
        scene is quiet the outpost never calls has_result(), and the pairing is
        deliberately left intact. So a bare 'time since apply_lens' would fire on
        a perfectly healthy idle camera. The discriminator is the wire itself: an
        answered request always shows POLLIN, whatever the caller does about it.
        Nothing on the wire, for this long, means the child never answered.

        The failure this catches leaves no trace anywhere else -- the accelerator
        stops responding with no USB reset, no kernel error, and no exception for
        the child to raise. There is nothing to catch; there is only a reply that
        never comes.
        """
        if not deadline or not self._pending:
            return False
        if self._tasking.is_ready():
            return False              # answered; merely not collected yet
        return (monotonic() - self._sent_at) > deadline

    def recycle(self) -> bool:
        """Replace a wedged child process. Returns True on success."""
        ok = self._tasking.recycle()
        # The fresh child's handshake echo primes a request, exactly as at startup.
        self._sent_at = monotonic()
        self._pending = ok
        self.state = SpyGlass.State_BUSY
        return ok

    def detect_motion(self, image) -> tuple:
        return self._motion.detect(image)

    def terminate(self):
        self._tasking.terminate()

    def __del__(self) -> None:
        self.terminate()
