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

# esta funcion compara las detecciones con los tracking. 
def compare_tracking(data, tracking_data):
    '''

    Esta funcion compara las listas de deteccion con la de tracking, para decidir si tomar las detecciones o actualizar
    el tracking. Esto se hace, ya que actualizar el tracking es un proceso que hace mas lent la aplicacion.
    Retorna una bandera, la cual le indica al sistema se actualiza el tracking.

    :param data: lista de detecciones
    :param tracking_data: lista de objetos en tracking
    :return: bandera que determina si se actualiza el tracking
    '''
    flag = True
    boxes, scores, classes, num_objects = data
    boxes_t, scores_t, classes_t, num_objects_t = tracking_data
    # si el numero de deteciones es menor al de tracking, hacemos tracking
    if num_objects < num_objects_t:
        print('no hay deteccion')
    # si el numero de detecciones es mayor, tomamos las detecciones
    elif num_objects > num_objects_t:
        print('mas detecciones')
        flag = False
    # comparamos cada bbox (aun falta tener en cuanta la clase y que no importe que este en desorden)
    # es decir, que la lista de tracking tenga otro orden a la de detecciones.
    else:
        for i in range(num_objects):
            for j in range(4):
                if abs(boxes[i][j] - boxes_t[i][j]) < 10:
                    flag = False
                    # A continuacion se actualiza los objetos en la lista de tracking
                    boxes_t[i][j] = boxes[i][j]
    return flag


def track_several_objects(frame, data, allowed_classes, tracking):
    '''

    :param frame: frame al que se le realizara tracking
    :param data: lista de detecciones
    :param allowed_classes: clases permitidas para realizar tracking (por defecto admite todas las clases)
    :param tracking: bandera que indica si se esta realizando tracking
    :return: trackers, tracking, tracking_data: trackers es la clase que hace tracking a multiples objetos,
                                                tracking_data es la lista de objetos en tracking
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

# function to change images based on stereo vision parameters and present 2 parallel images
def stereo_vision(frame_1, frame_2, data):
    '''

    :param frame_1: imagen de la camara 1
    :param frame_2: imagen de la camara 2
    :param data: detecciones
    :return: cada uno de los frames con los ajustes para vision estereo
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