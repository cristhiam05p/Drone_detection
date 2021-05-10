import os

# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from core.camera import *

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video_0', './data/video/video.mp4', '1th path to input video or set to 0 for webcam')
flags.DEFINE_string('video_1', './data/video/video.mp4', '2th path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('follow', False, 'follow detected object')
flags.DEFINE_boolean('stereo', False, 'show stereo calibrated images')


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path_0 = FLAGS.video_0
    video_path_1 = FLAGS.video_1
    # get video name by using split method
    video_name_0 = video_path_0.split('/')[-1]
    video_name_0 = video_name_0.split('.')[0]
    video_name_1 = video_path_1.split('/')[-1]
    video_name_1 = video_name_1.split('.')[0]

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid_0 = cv2.VideoCapture(int(video_path_0))
    except:
        vid_0 = cv2.VideoCapture(video_path_0)
##
    try:
        vid_1 = cv2.VideoCapture(int(video_path_1))
    except:
        vid_1 = cv2.VideoCapture(video_path_1)

    # for IP cameras
    capture_0 = Camera(video_path_0)
    capture_1 = Camera(video_path_1)

    out_0 = None
    out_1 = None

    if FLAGS.output:
        name_0 = FLAGS.output
        # by default VideoCapture returns float instead of int
        width_0 = int(vid_0.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_0 = int(vid_0.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_0 = int(vid_0.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out_0 = cv2.VideoWriter(name_0, codec, fps_0, (width_0, height_0))
        ##
        name_1 = FLAGS.output + "_1"
        # by default VideoCapture returns float instead of int
        width_1 = int(vid_1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_1 = int(vid_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_1 = int(vid_1.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out_1 = cv2.VideoWriter(name_1, codec, fps_1, (width_1, height_1))

    if FLAGS.stereo:
        square_size = 40  # mm size of square
        path = Path().absolute()
        path = os.path.join(path, 'stereo_param\ip_cameras\40mm')
        #path = os.path.join(path, str(square_size) + 'mm')
        map1L = np.loadtxt(os.path.join(path, "map1L.csv"), delimiter=",")
        map2L = np.loadtxt(os.path.join(path, "map2L.csv"), delimiter=",")
        map1R = np.loadtxt(os.path.join(path, "map1R.csv"), delimiter=",")
        map2R = np.loadtxt(os.path.join(path, "map2R.csv"), delimiter=",")
        #map1R = map1R.reshape((480, 640, 2))
        #map1L = map1L.reshape((480, 640, 2))

        map1R = map1R.reshape((576, 704, 2))
        map1L = map1L.reshape((576, 704, 2))

        data = square_size, path, map1L, map2L, map1R, map2R

    while True:
        #return_value, frame_0 = vid_0.read()
        #return_value_1, frame_1 = vid_1.read()
        # for IP cameras
        frame_0 = capture_0.getFrame()
        frame_1 = capture_1.getFrame()
        #time.sleep(0.01)
        #if return_value_1 and return_value:
        if (frame_0 is not None) and (frame_1 is not None):
            frame_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2RGB)
            frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
            #image = Image.fromarray(frame_0)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        start_time = time.time()

        # if stereo flag is enabled, calibrate stereo cameras
        if FLAGS.stereo:
            frame_0, frame_1 = stereo_vision(frame_0, frame_1, data)


        #frame_size_0 = frame_0.shape[:2]

        image_data_0 = cv2.resize(frame_0, (input_size, input_size))
        image_data_0 = image_data_0 / 255.
        image_data_0 = image_data_0[np.newaxis, ...].astype(np.float32)
        image_data_1 = cv2.resize(frame_1, (input_size, input_size))
        image_data_1 = image_data_1 / 255.
        image_data_1 = image_data_1[np.newaxis, ...].astype(np.float32)


        batch_data_0 = tf.constant(image_data_0)
        pred_bbox_0 = infer(batch_data_0)
        batch_data_1 = tf.constant(image_data_1)
        pred_bbox_1 = infer(batch_data_1)
        for key, value in pred_bbox_0.items():
            boxes_0 = value[:, :, 0:4]
            pred_conf_0 = value[:, :, 4:]
        for key, value in pred_bbox_1.items():
            boxes_1 = value[:, :, 0:4]
            pred_conf_1 = value[:, :, 4:]

        boxes_0, scores_0, classes_0, valid_detections_0 = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes_0, (tf.shape(boxes_0)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf_0, (tf.shape(pred_conf_0)[0], -1, tf.shape(pred_conf_0)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
            ##
        )
        boxes_1, scores_1, classes_1, valid_detections_1 = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes_1, (tf.shape(boxes_1)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf_1, (tf.shape(pred_conf_1)[0], -1, tf.shape(pred_conf_1)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # format bounding boxes_0 from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h_0, original_w_0, _ = frame_0.shape
        bboxes_0 = utils.format_boxes(boxes_0.numpy()[0], original_h_0, original_w_0)
        ##
        original_h_1, original_w_1, _ = frame_1.shape
        bboxes_1 = utils.format_boxes(boxes_1.numpy()[0], original_h_1, original_w_1)

        pred_bbox_0 = [bboxes_0, scores_0.numpy()[0], classes_0.numpy()[0], valid_detections_0.numpy()[0]]
        pred_bbox_1 = [bboxes_1, scores_1.numpy()[0], classes_1.numpy()[0], valid_detections_1.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to allow detections for only people)
        #allowed_classes = ['quadrotor', 'trirotor', 'hexarotor', 'octarotor', 'mavic', 'phantom']

        # if follow flag is enabled, control servos and camera zoom
        if FLAGS.follow:
            follow_object(frame_0, pred_bbox_0, 0.1, allowed_classes)
            # Aca hay que transformar los datos para llamar la funcion de control

        image_0 = utils.draw_bbox(frame_0, pred_bbox_0, FLAGS.info, allowed_classes=allowed_classes)
        image_1 = utils.draw_bbox(frame_1, pred_bbox_1, FLAGS.info, allowed_classes=allowed_classes)

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result_0 = np.asarray(image_0)
        cv2.namedWindow("result_0", cv2.WINDOW_AUTOSIZE)
        result_0 = cv2.cvtColor(image_0, cv2.COLOR_RGB2BGR)
        ##
        result_1 = np.asarray(image_1)
        cv2.namedWindow("result_0", cv2.WINDOW_AUTOSIZE)
        result_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("result_0", result_0)
            cv2.imshow("result_1", result_1)

        if FLAGS.output:
            out_0.write(result_0)
            out_1.write(result_1)
        if cv2.waitKey(1) & 0xFF == ord('q') : break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
