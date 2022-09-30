# setup DSLR as a webcam, live stream to /dev/video0
# sudo apt install gphoto2 v4l2loopback-utils ffmpeg
# TODO: in rpi error using v4l2loopback, need to build from source

# load in v4l2 module, only once after reboot
sudo modprobe v4l2loopback

# TODO: get key time zone automatically and only activate monitoring when it's coming

# open monitor channel
gphoto2 --stdout --capture-movie | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video0 &
python3 ./live.py # monitoring input and running live object detection

# check timestamp when breaking
echo "* TIMESTAMP BREAKING"
echo $(date +%Y-%m-%d-%T.%N)
# TODO: tune sensitivity, crop image input
# TODO: check if preset parameters and focus can remain

kill $(pidof ffmpeg) # kill the only one ffmpeg live process 
sleep 0.05 # give sys short time to get back control of device
# gphoto2 --set-config /main/capturesettings/capturemode=1
# gphoto2 --set-config /main/capturesetting/burstnumber=2

# check timestamp when init capture
echo "* TIMESTAMP INIT CAPTURE"
echo $(date +%Y-%m-%d-%T.%N)
gphoto2 --capture-image-and-download # shot asap
# huge interval between capture and saving 
# MUST USE MF NOT AF!!!

# check timestamp when saving files
echo "* TIMESTAMP SAVING"
echo $(date +%Y-%m-%d-%T.%N)
mv capt0000.jpg $(date +%Y-%m-%d-%T).jpg
# TODO: run a separate cropping script for the image just created
python3 ./flight.py # show possible flight urls
# TODO: save this picture to another location in the project

# refs
# ref: https://github.com/umlaeute/v4l2loopback/issues/216
# ref: https://ubuntu.pkgs.org/16.04/ubuntu-universe-i386/v4l2loopback-dkms_0.9.1-4_all.deb.html
# ref: https://www.raspberrypi.org/forums/viewtopic.php?t=253875
# ref: https://askubuntu.com/questions/856460/using-a-digital-camera-canon-as-webcam
# ref: https://medium.com/@y1017c121y/use-a-digital-camera-in-opencv-instead-of-webcam-d8445898e6c8

