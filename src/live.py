# DSLR Live Monitoring and OD


import os
import sys
import cv2
import time
import argparse

import pathlib
PATH_PARENT = pathlib.Path().cwd().parent

import numpy as np
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



# sentry recordings
# NAME_VIDEO = '2020-08-03-212223.webm'
NAME_VIDEO = '2020-08-02-100201.webm'
PATH_STREAM = str(PATH_PARENT / 'raw' / 'sentry' / NAME_VIDEO)

# dslr recordings
PATH_STREAM = "../raw/dlsr/MVI_9124.MOV"

# live from /dev/video0
PATH_STREAM = 0 # for webcam/DSLR
MONITOR = False # no streaming preview from cv2

# dslr handling stream forward to 
# assert "gphoto2 --auto-detect" has the right camera in the list
# TODO: set up a subprocess for enabling DSLR monitor mode
# TODO: end the monitor mode and capture in the script

# cropping params
# x_upper, x_lower, y_upper, y_lower = 0.9, 0.1, 0.7, 0.3
x_upper, x_lower, y_upper, y_lower = 1.0, 0.0, 1.0, 0.0



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



# images test
# img = cv2.imread("../assets/AF789.jpg")
# cv2.imshow('check', img)
# cv2.waitKey(0)

# pull the stream
stream = cv2.VideoCapture(PATH_STREAM)
ret = stream.set(3, 640) # length
ret = stream.set(4, 480)  # width
# ret = stream.set(10, 100)  # brightness

# cv2 package configs for monitoring
fps_observed = 1 # real-time fps
interf_sleep = 1 # extra inter-frame sleep
tick_freq = cv2.getTickFrequency()
font_disp = cv2.FONT_HERSHEY_SIMPLEX

while (True):
    t1 = cv2.getTickCount() # timestamp
    ret, img = stream.read() # same as cv2.imread(PATH)

    # display final/processed img 
    if ret: # if read exists
        y, x, d = img.shape

        # crop and expand in extra dim
        img_cropped = img[
            int(y_lower*y):int(y_upper*y),
            int(x_lower*x):int(x_upper*x)]
        img_expanded = np.expand_dims(img_cropped, axis=0)
        # extra processing
        # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blurry, etc

        # TODO: get the minimal model for successful inf!
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores,
             detection_classes, num_detections],
            feed_dict={image_tensor: img_expanded})

        # simplified ver, prints
        identified = cats_idx[int(classes[0][0])]['name']
        score = np.round(scores[0][0], 2)
        if score > 0.1: print('Identified:', identified, '| Scores:', score)
        if identified in ['airplane', 'keyboard']: break

        """
        # instead of drawing blocks, just print infos
        vis_util.visualize_boxes_and_labels_on_image_array(
            img_cropped,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            cats_idx,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.85)
        """

        if MONITOR: cv2.imshow('stream', img_cropped)

    # record stats
    t2 = cv2.getTickCount() # timestamp
    time = (t2 - t1) / tick_freq
    fps_observed = 1 / time
    # print(fps_observed)

    if cv2.waitKey(interf_sleep) == ord('q'):
        break # interval 10ms

# proper way to exit process
stream.release()
if MONITOR: cv2.destroyAllWindows()


