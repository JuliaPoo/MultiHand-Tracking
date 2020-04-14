# Multi Hand Tracker on Python

Python wrapper for Google's [Mediapipe Multi-Hand Tracking](https://github.com/google/mediapipe/blob/master/mediapipe/docs/multi_hand_tracking_mobile_gpu.md) pipeline. There are 2 predictor classes available. ```MultiHandTracker``` which predicts 2D keypoints and ```MultiHandTracker3D``` which predicts 3D keypoints. Keypoints generated from ```MultiHandTracker3D``` can be fed into ```is_right_hand``` to determine handedness (distinguish between right and left hand). ```is_right_hand``` is not part of Mediapipe's pipeline but I thought it'll be useful.

This code is built upon [Metal Whale's](https://github.com/metalwhale/hand_tracking) python wrapper for [Mediapipe Hand Tracking](https://github.com/google/mediapipe/blob/master/mediapipe/docs/hand_tracking_mobile_gpu.md).

## Getting Started

Basic usage, processing a single image for 2D keypoint predictions:
``` python
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

# Initialise detector
# the independent flag makes the detector process each image independently
detector = mht.MultiHandTracker(palm_model_path, landmark_model_path, anchors_path, independent = True)

# Get predictions
kp_list, box_list = detector(img)

# Plot predictions
plot_hand.plot_img(img, kp_list, box_list)
```

Basic usage, processing a single image for 3D keypoint predictions and determining handedness:
```python
from PIL import Image
import numpy as np
import multi_hand_tracker as mht
import plot_hand

img_path = "./test_pic.jpg"
img = Image.open(img_path)
img = np.array(img)

palm_model_path = "./models/palm_detection_without_custom_op.tflite"
landmark_model_path = "./models/hand_landmark_3D.tflite"
anchors_path = "./data/anchors.csv" 

# Initialise detector
# independent flag not implemented for MultiHandTracker3D
detector = mht.MultiHandTracker3D(palm_model_path, landmark_model_path, anchors_path)

# Get predictions
kp_list, box_list = detector(img)

# Determine handedness of each prediction
is_right = [mht.is_right_hand(kp) for kp in kp_list]

# Plot predictions
plot_hand.plot_img(img, kp_list, box_list, is_right=is_right)
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

Predictions from ```MultiHandTracker```:

<img src="https://github.com/JuliaPoo/MultiHand-Tracking/blob/master/demo.gif" alt="Process video result" width="400">

Video from [Signing Savvy](https://www.signingsavvy.com/sign/HAVE%20A%20GOOD%20DAY/8194/1).

Predictions from ```MultiHandTracker3D```:

<img src="https://github.com/JuliaPoo/MultiHand-Tracking/blob/master/demo_3D.gif" alt="Process video result" width="400">

Video from [Signing Savvy](https://www.signingsavvy.com/search/happy%2Bbirthday%2Bto%2Byou).

Not very good when the hands occlude each other.


## Acknowledgments

This work is a study of models developed by Google and distributed as a part of the [Mediapipe](https://github.com/google/mediapipe) framework.   
@metalwhale for the Python Wrapper for single hand tracking and for removing custom operations for the Palm Detections model.
