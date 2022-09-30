# load in v4l2 module, only once after reboot
sudo modprobe v4l2loopback

# open monitor channel
gphoto2 --stdout --capture-movie | ffmpeg -i - -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video0
