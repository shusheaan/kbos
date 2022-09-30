sudo apt-get update
sudo apt-get dist-upgrade
sudo apt-get install libatlas-base-dev python-tk
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install qt4-dev-tools libatlas-base-dev
sudo apt-get install protobuf-compiler gphoto2 v4l2loopback-utils ffmpeg

sudo pip3 install tensorflow pillow lxml jupyter matplotlib cython opencv-python

# MANUAL: git submodule setup - only once after clone
# git submodule init
# git submodule add https://github.com/tensorflow/models

# MANUAL, ref: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi
# cd .../tensorflow/models/research/object_detection
# protoc /protos/*.proto --python_out=. # convert .proto to .py

# MANUAL: download pre-trained models - md model zoo
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md

# MANUAL: get the fastest model
# wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
# tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz # unzip

# DEPRECATED: better/slower one
# wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
# wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz

# MANUAL: on RPi4B, need to build v4l2loopback manually
# git clone https://github.com/umlaeute/v4l2loopback
# cd v4l2loopback
# make && sudo make install
# sudo depmod -a
# sudo modprobe v4l2loopback
# ref: https://github.com/umlaeute/v4l2loopback
