# kbos

<img src="./raw/misc/cover.gif" alt="drawing" style="width:400px;"/>

live dslr capture module for kbos 22l/r arrivals   

## notes

- recompile tflite model for edgetpu and test speed increase
- make sure Coral is unplugged when `sudo apt install libedgetpu1-std`
- saves from DSLR `gphoto2 --list-all-files; gphoto2 --get-files=XXX-XXX`
- [tensorflow cross-compilation](https://www.tensorflow.org/lite/guide/build_rpi)
- [tensorflow lite official guide](https://www.tensorflow.org/lite/guide/python)
- [objdetapp example](https://github.com/datitran/object_detector_app)
- [docker ref](https://draveness.me/docker/)
