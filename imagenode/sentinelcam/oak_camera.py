import logging
import depthai as dai

class PipelineFactory:

    def MobileNetSSD(self):
        NN_SIZE = (300,300)
        NN_PATH = '/home/pi/depthai/depthai-python/examples/models/mobilenet-ssd_openvino_2021.4_6shave.blob'

        # Create pipeline
        pipeline = dai.Pipeline()

        # Define nodes and outputs
        nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
        cam = pipeline.create(dai.node.ColorCamera)
        encoder = pipeline.create(dai.node.VideoEncoder)

        xoutFrames = pipeline.create(dai.node.XLinkOut)
        xoutJPEG = pipeline.create(dai.node.XLinkOut)
        xoutNN = pipeline.create(dai.node.XLinkOut)

        xoutFrames.setStreamName("frames")
        xoutJPEG.setStreamName("jpegs")
        xoutNN.setStreamName("nn")

        # Properties
        nn.setConfidenceThreshold(0.5)
        nn.setBlobPath(NN_PATH)

        cam.setPreviewSize(NN_SIZE)
        cam.setPreviewKeepAspectRatio(False)
        cam.setInterleaved(False)
        cam.setIspScale(1,3)
        cam.setFps(30)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        # scale collection down from 4K to just FullHD
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) 
        cam.setVideoSize(640,360) # reduce further for storage

        encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)

        # Linking
        cam.video.link(encoder.input)
        cam.video.link(xoutFrames.input)
        encoder.bitstream.link(xoutJPEG.input)
        cam.preview.link(nn.input)
        nn.out.link(xoutNN.input)

        # Connect to device and start pipeline
        device = dai.Device(pipeline)
        return device

    def __init__(self, pipeline) -> None:
        logging.debug(f"Starting DepthAI pipeline '{pipeline}'")
        PipeLines = {
            'MobileNetSSD' : self.MobileNetSSD
        }
        self.device = PipeLines[pipeline]()
