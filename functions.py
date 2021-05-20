import os
import cv2
import random
import numpy as np
import tensorflow as tf
import pytesseract
from core.utils import read_class_names
from core.config import cfg
from pathlib import Path
##uncomment line below when using servos
#from core.ptz import *


def compare_tracking(data, tracking_data):
    '''

    This function compares the detection lists with the tracking list, to decide whether to take 
    the detections or update the tracking. This is done, as updating the tracking is a process that 
    slows down the application. It returns a flag, which tells the system to update the tracking.

    :param data: list of detections
    :param tracking_data: list of objects to be tracked
    :return: flag that determines whether the tracking is updated
    '''
    flag = True
    boxes, scores, classes, num_objects = data
    boxes_t, scores_t, classes_t, num_objects_t = tracking_data
    # If the number of detections is less than the number of tracking, we do tracking
    if num_objects < num_objects_t:
        print('no hay deteccion')
    # If the number of detections is greater, we take the detections
    elif num_objects > num_objects_t:
        print('mas detecciones')
        flag = False
    # we compare each bbox (we still need to take into account the class and that it doesn't matter if it is in disorder)
    # i.e. the tracking list has a different order than the detections list.
    else:
        for i in range(num_objects):
            for j in range(4):
                if abs(boxes[i][j] - boxes_t[i][j]) < 10:
                    flag = False
                    # The objects in the tracking list are then updated.
                    boxes_t[i][j] = boxes[i][j]
    return flag


def track_several_objects(frame, data, allowed_classes, tracking):
    '''
    creates and returns a new instance of multitracking, as well as a list of the objects it contains 
    and the flag indicating that it is tracking.

    :param frame: frame to be tracked
    :param data: list of detections
    :param allowed_classes: classes allowed for tracking (all classes are supported by default)
    :param tracking: flag indicating whether tracking is in progress
    :return: trackers, tracking, tracking_data: trackers is the class that tracks multiple objects,
                                                tracking_data is the list of objects in tracking
    '''
    trackers = cv2.MultiTracker_create() 
    boxes, scores, classes, num_objects = data
    boxes_t = []
    class_names = read_class_names(cfg.YOLO.CLASSES)
    for i in range(num_objects):
        class_index = int(classes[i])
        class_name = class_names[class_index]
        print(f"class name: {class_name}, confidence: {scores[i]}")
        if class_name in allowed_classes:
            tracker = cv2.TrackerKCF_create()
            xmin, ymin, xmax, ymax = boxes[i]
            w = xmax - xmin
            h = ymax - ymin
            x = xmin
            y = ymin
            bbox = x, y, w, h
            boxes_t.append([int(x), int(y), int(w), int(h)])
            trackers.add(tracker, frame, bbox)
            tracking = True
    tracking_data = boxes, scores, classes, num_objects
    return trackers, tracking, tracking_data


def stereo_vision(frame_1, frame_2, data):
    '''
    function to change images based on stereo vision parameters and present 2 parallel images

    :param frame_1: image of camera 1
    :param frame_2: image of camera 2
    :param data: detections
    :return: each of the frames with the settings for stereo viewing
    '''
    square_size, path, map1L, map2L, map1R, map2R = data
    stereo_1 = frame_1
    stereo_2 = frame_2

    # Rectify the images on rotation and alignment
    stereo_1 = cv2.remap(stereo_1, np.int16(map1R), np.int16(map2R), cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    stereo_2 = cv2.remap(stereo_2, np.int16(map1L), np.int16(map2L), cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    return stereo_1, stereo_2


# function to follow detected object with servos and zoom (ptz)
def follow_object(img, data, threshold, allowed_classes = list(read_class_names(cfg.YOLO.CLASSES).values())):
    first = True
    height = img.shape[0]
    width = img.shape[1]
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    for i in range(num_objects):
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes and first:
            xmin, ymin, xmax, ymax = boxes[i]
            print(str(class_name) + ":  xmin: " + str(xmin) + ", ymin: " + str(ymin) + ", xmax: " + str(xmax) + ", ymax: " + str(ymax))
            control_zoom (width, height, boxes[i])
            move_camera (width, height, boxes[i], threshold)
            firts = False
        else:
            continue