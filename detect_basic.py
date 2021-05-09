import os
import cv2
import numpy as np
import time
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from PIL import Image
# en utils se encuentran varias funciones necesarias para la deteccion, entre otros
import core.utils as utils
from core.yolov4 import filter_boxes
# en functions se encuentran especificamente funciones adicionales a la deteccion
from core.functions import *
##uncomment line below when using IPCamera 
from core.camera import *
##uncomment line below when using servos
#from core.pzt import *


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('follow', False, 'follow detected object')
flags.DEFINE_boolean('tracking', False, 'track detected object')


def main(_argv):
    # basic configurations
    config = ConfigProto()
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    # get video name by using split method
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)
  
    ##uncomment line below when using IPCamera
    capture = Camera(video_path)

    out = None

    if FLAGS.output:
        print('entro aca')
        print(FLAGS.output)
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        fps = 10
        #out = cv2.VideoWriter(FLAGS.output, codec, 10.0, (width, height))
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    boxes_t = []
    frame_num = 0
    initial_frame = 0
    tracking = False
    time.sleep(3)
    while True:
        start_time = time.time()
        # uncomment line below when using USBCamera
        #return_value, frame = vid.read()
        # uncomment line below when using IPCamera
        frame = capture.getFrame()
        #if return_value:
        if frame is not None:
            frame_num += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        # print(f"frame num: {frame_num}")
        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        ## Lo anterior es lo que mas consume tiempo

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        #print(valid_detections.numpy()[0])
        # print(bboxes)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to allow detections for only people)
        # allowed_classes = ['quadrotor', 'trirotor', 'hexarotor', 'octarotor', 'mavic', 'phantom']
        # allowed_classes = ['person']
        # allowed_classes = ['cell phone']

        # if follow flag is enabled, control servos and camera zoom
        if FLAGS.follow:
            follow_object(frame, pred_bbox, 0.1, allowed_classes)

        # if tracking flag is enabled, control servos and camera zoom
        if FLAGS.tracking and not tracking:
            # tracker, tracking = track_object(frame, pred_bbox, allowed_classes, tracking)
            trackers, tracking, tracking_data = track_several_objects(frame, pred_bbox, allowed_classes, tracking)
            initial_frame = frame_num

        # Aca entra solo si se quiere hacer tracking
        if FLAGS.tracking:
            # la siguiente funcion determina si se hace o no el tracking
            flag = compare_tracking(pred_bbox, tracking_data)
            # si existe una lista de tracking y la bandera lo indica, se grafican los objetos del tracking
            if tracking and flag:
                print('entro al tracking')
                (success, boxes_t) = trackers.update(frame)  # x0, y0, w h
                boxes, scores, classes, num_objects = tracking_data
                if success and (frame_num-initial_frame < 5):
                    for i in range(num_objects):
                        box = boxes_t[i]
                        class_name = classes[i]
                        (x, y, w, h) = [int(v) for v in box]
                        (x1, y1, x2, y2) = x, y, x+w, y+h
                        box = (x1, y1, x2, y2)
                        boxes[i] = box
                else:
                    tracking = False
                tracking_data = boxes, scores, classes, num_objects
                image = utils.draw_bbox(frame, tracking_data, FLAGS.info, allowed_classes=allowed_classes)
            else:
                image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes)
        else:
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes)

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result: " + FLAGS.video, cv2.WINDOW_AUTOSIZE)

        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        

        if not FLAGS.dont_show:
            cv2.imshow("result: " + FLAGS.video, result)

        if FLAGS.output:
            print('guardadno')
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
