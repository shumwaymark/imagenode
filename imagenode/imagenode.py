"""imagenode: capture, transform and transfer images to imagehub

Enables Raspberry Pi computers to capture images with the PiCamera, perform
image transformations and send the images to a central imagehub for further
processing. Can send other sensor data such as temperature data and GPIO data.
Works on other types of (non Raspberry Pi) computers with webcams.

Typically run as a service or background process. See README.rst for details.

Copyright (c) 2017 by Jeff Bass.
License: MIT, see LICENSE for more details.

19-Apr-2020, Mark.Shumway@swanriver.dev 
Added support for asynchronous logging over PyZMQ 
"""

import os
import sys
import signal
import logging
import logging.handlers
import traceback
import json
import socket
from zmq.log.handlers import PUBHandler
from tools.utils import clean_shutdown_when_killed, Patience
from tools.imaging import Settings, ImageNode

def main():
    # set up controlled shutdown when Kill Process or SIGTERM received
    signal.signal(signal.SIGTERM, clean_shutdown_when_killed)
    log = start_logging()
    try:
        log.info('Starting imagenode.py')
        settings = Settings()  # get settings for node cameras, ROIs, GPIO
        node = ImageNode(settings)  # start ZMQ, cameras and other sensors
        if settings.publish_log:
            log = start_logPublisher(node, settings)
            log.info('Camera startup complete')
        # forever event loop
        while True:
            # read cameras and run detectors until there is something to send
            while not node.send_q:
                node.read_cameras()
            while len(node.send_q) > 0:  # send frames until send_q is empty
                text, image = node.send_q.popleft()
                hub_reply = node.send_frame(text, image)
                node.process_hub_reply(hub_reply)
    except KeyboardInterrupt:
        log.warning('Ctrl-C was pressed.')
    except SystemExit:
        log.warning('SIGTERM was received.')
    except Exception as ex:  # traceback will appear in log
        log.exception('Unanticipated error with no Exception handler.')
    finally:
        if 'node' in locals():
            node.closeall(settings) # close cameras, GPIO, files
        log.info('Exiting imagenode.py')

def start_logging():
    log = logging.getLogger()
    handler = logging.handlers.RotatingFileHandler('imagenode.log',
        maxBytes=15000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s ~ %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    return log

def start_logPublisher(node, settings):
    log = logging.getLogger()
    log.info('Activating log publisher')
    zmq_log_handler = PUBHandler("tcp://*:{}".format(settings.publish_log))
    zmq_log_handler.setFormatter(logging.Formatter(fmt='{asctime}|{message}', style='{'))
    zmq_log_handler.root_topic = settings.nodename
    log.addHandler(zmq_log_handler)
    
    cams = {cam.viewname: cam.res_resized for cam in node.camlist if cam.video}
    handoff = {'node': settings.nodename, 'host': socket.gethostname(), 
               'log': settings.publish_log, 'video': settings.publish_cams,
               'cams': cams}
    msg = settings.nodename + "|$CameraUp|" + json.dumps(handoff)
    
    with Patience(settings.patience):
        hub_reply = node.send_frame(msg, node.tiny_jpg)
        if not hub_reply == b'OK':
            log.error('Unexpected reponse adding camwatcher: ' + hub_reply.decode())
            sys.exit()
    log.handlers.remove(log.handlers[0]) # OK, all logging over PUB socket only
    log.setLevel(logging.INFO)
    node.log = log
    return log

if __name__ == '__main__' :
    main()
