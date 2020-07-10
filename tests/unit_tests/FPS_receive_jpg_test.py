"""FPS_receive_test.py -- receive (text, image) pairs & print FPS stats

A test program to provide FPS statistics as different imagenode algorithms are
being tested. This program receives images OR images that have been jpg
compressed, depending on the setting of the JPG option.

It computes and prints FPS statistics.

1. Edit the options, such as the JPG option.

2. Set the yaml options on the imagenode sending RPi in the imagenode.yaml
   file at the home directory. Be sure that the jpg setting on the RPi matches
   the setting of JPG below.

2. Run this program in its own terminal window on the mac:
   python FPS_receive_test.py.

   This 'receive images' program must be running before starting
   the RPi image sending program.

2. Run the imagenode image sending program on the RPi:
   python imagenode.py

A cv2.imshow() window will only appear on the Mac that is receiving the
tramsmitted images if the "show_images" option is set to True.

The receiving program will run until the "number_of_images" option number is
reached or until Ctrl-C is pressed.

The imagenode program running on the RPi will end itself after a timeout or you
can end it by pressing Ctrl-C.

For details see the docs/FPS-tests.rst file.
"""

import cv2
import sys
import time
import imagezmq
import traceback
from imutils.video import FPS
from collections import defaultdict

#################################################
# set options
JPG = True  # or False if receiving images
SHOW_IMAGES = True
#################################################

def receive_image():
    text, image = image_hub.recv_image()
    return text, image

def receive_jpg():
    text, jpg_buffer = image_hub.recv_jpg()
    image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)

if JPG:
    receive_tuple = receive_jpg
else:
    receive_tuple = receive_image
# instantiate image_hub
image_hub = imagezmq.ImageHub()

image_count = 0
sender_image_counts = defaultdict(int)  # dict for counts by sender
first_image = True

try:
    while True:  # receive images until Ctrl-C is pressed
        sent_from, jpg_buffer = image_hub.recv_jpg()
        if first_image:
            fps = FPS().start()  # start FPS timer after first image is received
            first_image = False
        fps.update()
        image_count += 1  # global count of all images received
        sender_image_counts[sent_from] += 1  # count images for each RPi name
        if SHOW_IMAGES:
            cv2.imshow(sent_from, image)  # display images 1 window per sent_from
            cv2.waitKey(1)
        image_hub.send_reply(b'OK')  # REP reply
except (KeyboardInterrupt, SystemExit):
    pass  # Ctrl-C was pressed to end program; FPS stats computed below
except Exception as ex:
    print('Python error with no Exception handler:')
    print('Traceback error:', ex)
    traceback.print_exc()
finally:
    # stop the timer and display FPS information
    print()
    print('Test Program: ', __file__)
    print('Total Number of Images received: {:,g}'.format(image_count))
    if first_image:  # never got images from any RPi
        sys.exit()
    fps.stop()
    print('Number of Images received from each RPi:')
    for RPi in sender_image_counts:
        print('    ', RPi, ': {:,g}'.format(sender_image_counts[RPi]))
    compressed_size = len(jpg_buffer)
    print('Size of last jpg buffer received: {:,g} bytes'.format(compressed_size))
    image_size = image.shape
    print('Size of last image received: ', image_size)
    uncompressed_size = 1
    for dimension in image_size:
        uncompressed_size *= dimension
    print('    = {:,g} bytes'.format(uncompressed_size))
    print('Compression ratio: {:.2f}'.format(compressed_size / uncompressed_size))
    print('Elasped time: {:,.2f} seconds'.format(fps.elapsed()))
    print('Approximate FPS: {:.2f}'.format(fps.fps()))
    cv2.destroyAllWindows()  # closes the windows opened by cv2.imshow()
    image_hub.close()  # closes ZMQ socket and context
    sys.exit()
