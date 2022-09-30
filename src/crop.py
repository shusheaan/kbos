# Auto batch cropping


import os
import sys
import cv2
import time
import argparse

import pathlib
PATH_PARENT = pathlib.Path().cwd().parent

import numpy as np
import pandas as pd
import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

# official object_detection repo imports
PATH_RES = PATH_PARENT / 'models' / 'research'
PATH_OD  = PATH_RES / 'object_detection'
sys.path.append(str(PATH_OD))
sys.path.append(str(PATH_RES))
from object_detection.utils import label_map_util


# load-in pretrained model
NAME_MODEL = 'ssdlite_mobilenet_v2_coco_2018_05_09'
PATH_CKPT = str(PATH_PARENT / 'pretrained' / NAME_MODEL
    / 'frozen_inference_graph.pb')
PATH_LABL = str(PATH_OD / 'data' / 'mscoco_label_map.pbtxt')
N_CLASSES = 90

# process label maps and indexes
labl = label_map_util.load_labelmap(PATH_LABL)
cats = label_map_util.convert_label_map_to_categories(
    labl, max_num_classes=N_CLASSES, use_display_name=True)
cats_idx = label_map_util.create_category_index(cats)

# build session for the trained graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

# input and output tensors
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0') # input
detection_boxes   = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores  = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections    = detection_graph.get_tensor_by_name('num_detections:0')



# get path of raw
PATH_RAW = PATH_PARENT / 'raw' / 'dlsr'
PATH_EXP = PATH_PARENT / 'raw' / 'cropped'

# get image list
for file_name in PATH_RAW.iterdir():
    img = cv2.imread(str(file_name))

    # cv2.imshow('test', img)
    # cv2.waitKey(100)

    img_expanded = np.expand_dims(img, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
         detection_classes, num_detections],
        feed_dict={image_tensor: img_expanded})

    # simplified ver, prints
    identified = cats_idx[int(classes[0][0])]['name']
    score = np.round(scores[0][0], 2)
    boxar = boxes[0][0]

    # expand box identified
    expand_ratio = 0.30
    ymin_raw, ymax_raw = boxar[0], boxar[2]
    xmin_raw, xmax_raw = boxar[1], boxar[3]
    ymiddle = (ymax_raw + ymin_raw) / 2
    xmiddle = (xmax_raw + xmin_raw) / 2
    ylen = ymax_raw - ymin_raw
    xlen = xmax_raw - xmin_raw

    # adjusted box edges
    ymax = max(ymiddle + (1 + expand_ratio) * ylen / 2, 0)
    ymin = max(ymiddle - (1 + expand_ratio) * ylen / 2, 0)
    xmax = max(xmiddle + (1 + expand_ratio) * xlen / 2, 0)
    xmin = max(xmiddle - (1 + expand_ratio) * xlen / 2, 0)

    # crop image and show
    y, x, d = img.shape
    img_cropped = img[
        int(ymin * y):int(ymax * y),
        int(xmin * x):int(xmax * x)]

    # export 
    print(str(file_name))

    file_list = str(file_name).split('/')
    PATH_EXP_FILE = PATH_EXP / file_list[len(file_list)-1]

    # cv2.imshow(str(file_name), img_cropped)
    cv2.imwrite(str(PATH_EXP_FILE), img_cropped)
