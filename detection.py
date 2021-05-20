from absl import app, flags, logging
from absl.flags import FLAGS
import detect_basic
from multiprocessing import Process
import time


def detect_camara(rtsp_link):
    """
    this function receives as a single parameter the connection link to the camera and other parameters can be added
    directly within the function.
    :param rtsp_link: IP camera link
    """
    FLAGS.output = './detections/' + rtsp_link[30:33] + '.avi'
    FLAGS.weights = './checkpoints/yolov4-tiny-drone-416'
    FLAGS.tracking = True
    FLAGS.video = rtsp_link
    try:
        app.run(detect_basic.main)
    except SystemExit:
        pass

if __name__ == '__main__':
      
    rtsp_link1 = "rtsp://admin:123456@192.168.1.168/sub"
    rtsp_link2 = "rtsp://admin:123456@192.168.1.169/sub"
    process1 = Process(target=detect_camara, args=(rtsp_link1,), name=f"proceso_{rtsp_link1}")
    process2 = Process(target=detect_camara, args=(rtsp_link2,), name=f"proceso_{rtsp_link2}")
    process1.start()
    # the delay is necessary for the two processes to be created correctly.
    time.sleep(0.5)
    process2.start()
