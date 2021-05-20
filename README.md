# Drone_detection_tensorflow_anaconda

This repository contains the necessary code to convert a yolov4 network into tensorflow, in order to perform object detection on images and videos. Especially this code is oriented to the real time detection of images acquired by IP cameras within the framework of a Drone detection system.

The system is currently capable of detecting images from two cameras simultaneously. In order to make this possible and since IP cameras were used for image acquisition, it was necessary to use multithreading as well as multiprocessing. It is also possible to perform detection with one camera seamlessly.

The main script is detect_basic, which is responsible for performing detection on a camera and has different flags that indicate certain parameters and enable other functionalities, such as supporting detection with tracking.

The main objectives of this section are:

- [x] Detection using a customised network.
- [x] Implementation of tracking in the detection system.
- [x] Simultaneous detection over two cameras.
- [ ] Estimation of distance and location of the detected object.
- [ ] Graph of the 3D location of the detected object.


## Requirements installation

### Conda 

```
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu
```

```
# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```



## References

- [theAIGuysCode/yolov4-custom-functions](https://github.com/theAIGuysCode/yolov4-custom-functions).
