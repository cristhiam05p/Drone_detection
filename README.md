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


Inside this repository in the data folder there are 2 yolov4-tiny networks, the first one is the official one, trained on coco that can detect up to 80 classes, while the other one is the custom network, trained to detect different kinds of drones into the classes: 

- Quadrotor
- Trirotor
- Hexarotor
- Octarotor
- DJI Mavic
- DJI Phantom
- Airplane
- Bird

In case you want to perform the detection using different weights, it is possible to copy them into the same folder, and then convert them to tensorflow format.
The only change within the code needs to be make in order for a custom model to work is on line 14 of 'core/config.py' file. Update the code to point at the custom.names file as seen below. As well as changing the anchor boxes to better accommodate other objects of interest. 

![image](https://user-images.githubusercontent.com/64755713/119009397-b16e7980-b958-11eb-80e1-fc8dda73ffd7.png)

## Convert weights to tensorflow and run the detector

Then all that needs to be done is to convert the weights of the trained yolov4 or yolov4-tiny network to the corresponding tensorflow model.
```
# Convert darknet weights to tensorflow
## yolov4-tiny
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny

# Run yolov4-tiny tensorflow model
python detect.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --images ./data/images/kite.jpg --tiny

# Run yolov4-tiny on video
python detect_video.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --video ./data/video/video.mp4 --output ./detections/results.avi --tiny

# Run yolov4-tiny on webcam
python detect_video.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi --tiny

```



## References

- [theAIGuysCode/yolov4-custom-functions](https://github.com/theAIGuysCode/yolov4-custom-functions).
