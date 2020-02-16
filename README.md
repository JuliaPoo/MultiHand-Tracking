# Multi Hand Tracker on Python

Python wrapper for Google's [Mediapipe Multi-Hand Tracking](https://github.com/google/mediapipe/blob/master/mediapipe/docs/multi_hand_tracking_mobile_gpu.md) pipeline. I've added primitive hand instance tracking. It is not reliable and in actual usage it should be supplemented with other information.

This code is built upon [Metal Whale's](https://github.com/metalwhale/hand_tracking) python wrapper for [Mediapipe Hand Tracking](https://github.com/google/mediapipe/blob/master/mediapipe/docs/hand_tracking_mobile_gpu.md).

## Getting Started

Basic usage, processing a single image:
``` 
from PIL import Image
import numpy as np
import multi_hand_tracker as mht
import plot_hand

img_path = "./test_pic.jpg"
img = Image.open(img_path)
img = np.array(img)

palm_model_path = "./models/palm_detection_without_custom_op.tflite"
landmark_model_path = "./models/hand_landmark.tflite"
anchors_path = "./data/anchors.csv" 

# the independent flag makes the detector process each image independently
detector = mht.MultiHandTracker(palm_model_path, landmark_model_path, anchors_path, independent = True)
kp_list, box_list = detector(img)
plot_hand.plot_img(img, kp_list, box_list)
```

### Requirements

These are required to use the HandTracker module

```
numpy
opencv
tensorflow
shapely
scipy
```
To use the plotting functions you would need `matplotlib`

## Results

<img src="https://github.com/JuliaPoo/MultiHand-Tracking/blob/master/test2.gif" alt="Process video result" width="400">

Video from [Signing Savvy](https://www.signingsavvy.com/sign/HAVE%20A%20GOOD%20DAY/8194/1). Not very good when the hands occlude each other.

## Acknowledgments

This work is a study of models developed by Google and distributed as a part of the [Mediapipe](https://github.com/google/mediapipe) framework.   
@metalwhale for the Python Wrapper for single hand tracking and for removing custom operations for the Palm Detections model.
