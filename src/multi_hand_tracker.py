import csv
import cv2
import numpy as np
import tensorflow as tf
from shapely.geometry import Polygon

# TODO: 
#  Implement the realtime optimisation
#  Support 3D???
#  Make camera demo


class MultiHandTracker():
    
    r"""
    Class to use Google's Mediapipe MultiHandTracking pipeline from Python.
    Any any image size and aspect ratio supported.

    Args:
        palm_model: path to the palm_detection.tflite
        joint_model: path to the hand_landmark.tflite
        anchors_path: path to the csv containing SSD anchors
        
    Optional_Args:
        box_enlarge: bbox will be scalled by box_enlarge. Default 1.5
        box_shift: bbox will be shifted downwards by box_shift. Default 0.2
        max_hands: The maximum number of hands to detect. Default 2
        detect_hand_thres: Threshold for hand detections. Default 0.7
        detect_keypoints_thres: Threshold whereby keypoints detected will be considered a hand. Default 0.1
        iou_thres: non-maximum suppression threshold. Default 0.45
        
    Output:
        Tuple (keypoints list, bounding box list)
            -- keypoints list is a list of (21,2) array each representing the coordinates of keypoints for each hand
            -- bounding box list is a list of (4,2) array each representing the bounding box for each hand
        
        
    Examples::
        >>> det = HandTracker(path1, path2, path3)
        >>> input_img = np.random.randint(0,255, 256*256*3).reshape(256,256,3)
        >>> keypoints, bbox = det(input_img)
    """

    def __init__(self, 
                 palm_model, 
                 joint_model, 
                 anchors_path,
                 box_enlarge=1.5, 
                 box_shift=0.2, 
                 max_hands = 2,
                 detect_hand_thres = 0.7,
                 detect_keypoints_thres = 0.1,
                 iou_thres = 0.45
                 ):
        
        # BBox predictions parameters
        self.box_shift = box_shift
        self.box_enlarge = box_enlarge
        
        # HandLanmarks parameters (not used for now)
        self.max_hands = max_hands
        self.is_hands_list = [False]*max_hands
        
        # Thresholds init
        self.detect_hand_thres = detect_hand_thres
        self.detect_keypoints_thres = detect_keypoints_thres
        self.iou_thres = iou_thres

        # Initialise models
        self.interp_palm = tf.lite.Interpreter(palm_model)
        self.interp_palm.allocate_tensors()
        self.interp_joint = tf.lite.Interpreter(joint_model)
        self.interp_joint.allocate_tensors()
        
        # reading the SSD anchors
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]
        # reading tflite model paramteres
        output_details = self.interp_palm.get_output_details()
        input_details = self.interp_palm.get_input_details()
        
        # Getting tensor index for palm detection
        self.in_idx = input_details[0]['index']
        self.out_reg_idx = output_details[0]['index']
        self.out_clf_idx = output_details[1]['index']
        
        # Getting tensor index for hand landmarks
        self.in_idx_joint = self.interp_joint.get_input_details()[0]['index']
        self.out_idx_joint = self.interp_joint.get_output_details()[0]['index']
        self.out_idx_is_hand = self.interp_joint.get_output_details()[1]['index']

        # 90° rotation matrix used to create the alignment trianlge        
        self.R90 = np.r_[[[0,1],[-1,0]]]

        # trianlge target coordinates used to move the detected hand
        # into the right position
        self._target_triangle = np.float32([
                        [128, 128],
                        [128,   0],
                        [  0, 128]
                    ])
        self._target_box = np.float32([
                        [  0,   0, 1],
                        [256,   0, 1],
                        [256, 256, 1],
                        [  0, 256, 1],
                    ])
    
    def _get_triangle(self, kp0, kp2, dist=1):
        """get a triangle used to calculate Affine transformation matrix"""

        dir_v = kp2 - kp0
        dir_v /= np.linalg.norm(dir_v)

        dir_v_r = dir_v @ self.R90.T
        return np.float32([kp2, kp2+dir_v*dist, kp2 + dir_v_r*dist])

    @staticmethod
    def _triangle_to_bbox(source):
        # plain old vector arithmetics
        bbox = np.c_[
            [source[2] - source[0] + source[1]],
            [source[1] + source[0] - source[2]],
            [3 * source[0] - source[1] - source[2]],
            [source[2] - source[1] + source[0]],
        ].reshape(-1,2)
        return bbox
    
    @staticmethod
    def _im_normalize(img):
         return np.ascontiguousarray(
             2 * ((img / 255) - 0.5
        ).astype('float32'))
       
    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x) )
    
    @staticmethod
    def _pad1(x):
        return np.pad(x, ((0,0),(0,1)), constant_values=1, mode='constant')

    @staticmethod
    def _GetCandidateIdx(box_list, max_hands = 2, iou_thres = 0.45):
        '''Perform non max suppression'''
        
        def _IOU(poly1, poly2):
            return poly1.intersection(poly2).area / poly1.union(poly2).area
    
        box_groups = [[(box_list[0], 0)]]

        # Group BBOX according to IOU
        for idx, box in enumerate(box_list[1:]):
            idx += 1
            pbox = Polygon(box)
            new_group = True
            for group in box_groups:
                if _IOU(pbox, Polygon(group[0][0])) > iou_thres:
                    group.append((box, idx))
                    new_group = False
                    break
            if new_group: 
                box_groups.append([(box, idx)])

        # Sort bbox groups according to number of elements in group
        len_groups = [(len(group), idx) for idx, group in enumerate(box_groups)]
        len_groups = sorted(len_groups, reverse = True, key = lambda x: x[0])

        # Return candidate bbox groups (max max_hands groups)
        candidate_groups_idx = [len_idx[-1] for len_idx in len_groups[:max_hands]]
        candidate_groups = [box_groups[idx] for idx in candidate_groups_idx]

        # Return candidate bbox index based on maximum area
        candidate_groups_area = [[(Polygon(box).area, idx) for box, idx in group] for group in candidate_groups]
        candidate_idx = [max(group, key=lambda x: x[0])[-1] for group in candidate_groups_area]

        return candidate_idx
    
    def _source_to_bbox(self, scale, pad, source):
        '''
        From output of palm detections to bbox
        Also returns parameters used for transforming the 
        handlandmarks to prevent repeated computation
        '''
        
        Mtr = cv2.getAffineTransform(
                source * scale,
                self._target_triangle
                )
        Mtr_temp = self._pad1(Mtr.T).T
        Mtr_temp[2,:2] = 0
        Minv = np.linalg.inv(Mtr_temp)
        
        box_orig = (self._target_box @ Minv.T)[:,:2]
        box_orig -= pad[::-1]
        
        return box_orig, source, Mtr, Minv
    
    def predict_joints(self, img_norm, hand_thres = 0.):
        '''Returns the joints output and if a hand is present in img_norm'''
        
        self.interp_joint.set_tensor(
            self.in_idx_joint, img_norm.reshape(1,256,256,3))
        self.interp_joint.invoke()

        joints = self.interp_joint.get_tensor(self.out_idx_joint)
        is_hand = self.interp_joint.get_tensor(self.out_idx_is_hand)[0][0]*10**11
        
        return joints.reshape(-1,2), is_hand > hand_thres

    def detect_hand(self, img_norm):
        '''Perform palm hand detection'''
        
        assert -1 <= img_norm.min() and img_norm.max() <= 1,\
        "img_norm should be in range [-1, 1]"
        assert img_norm.shape == (256, 256, 3),\
        "img_norm shape must be (256, 256, 3)"

        # predict hand location and 7 initial landmarks
        self.interp_palm.set_tensor(self.in_idx, img_norm[None])
        self.interp_palm.invoke()

        out_reg = self.interp_palm.get_tensor(self.out_reg_idx)[0]
        out_clf = self.interp_palm.get_tensor(self.out_clf_idx)[0,:,0]
        
        # finding the best prediction
        detecion_mask = self._sigm(out_clf) > self.detect_hand_thres
        candidate_detect = out_reg[detecion_mask]
        candidate_anchors = self.anchors[detecion_mask]

        if candidate_detect.shape[0] == 0:
            return []
        
        candidate_idx = list(range(candidate_detect.shape[0]))

        # bounding box offsets, width and height
        bbox_dets_list = []
        center_wo_offst_list = []
        for idx in candidate_idx:
            dx,dy,w,h = candidate_detect[idx, :4]
            bbox_dets_list.append((dx,dy,w,h))
            
            center_wo_offst_list.append(candidate_anchors[idx,:2] * 256)
        
        # 7 initial keypoints
        keypoints_list = [center_wo_offst_list[i] + candidate_detect[idx,4:].reshape(-1,2) for i,idx in enumerate(candidate_idx)]
        side_list = [max(w,h) * self.box_enlarge for _,_,w,h in bbox_dets_list]
        
        # now we need to move and rotate the detected hand for it to occupy a
        # 256x256 square
        # line from wrist keypoint to middle finger keypoint
        # should point straight up
        # I'm sorry for the list comprehensions
        source_list = [self._get_triangle(keypoints[0], keypoints[2], side) for keypoints, side in zip(keypoints_list, side_list)]
        source_list = [source - (keypoints[0] - keypoints[2]) * self.box_shift for source, keypoints in zip(source_list, keypoints_list)]
        source_list = [np.array(source, dtype="float32") for source in source_list]
        
        return source_list

    def preprocess_img(self, img):
        # fit the image into a 256x256 square
        shape = np.r_[img.shape]
        pad = (shape.max() - shape[:2]).astype('uint32') // 2
        img_pad = np.pad(
            img,
            ((pad[0],pad[0]), (pad[1],pad[1]), (0,0)),
            mode='constant')
        img_small = cv2.resize(img_pad, (256, 256))
        img_small = np.ascontiguousarray(img_small)
        
        img_norm = self._im_normalize(img_small)
        return img_pad, img_norm, pad


    def __call__(self, img, get_kp = True):
        img_pad, img_norm, pad = self.preprocess_img(img)
        
        source_list = self.detect_hand(img_norm)
        if len(source_list) == 0:
            return [], []
        
        scale = max(img.shape) / 256
        bbox_source_Mtr_Minv_list = [self._source_to_bbox(scale, pad, source) for source in source_list]
        box_orig_list = [ele[0] for ele in bbox_source_Mtr_Minv_list]
        
        box_valid_idx = self._GetCandidateIdx(box_orig_list, max_hands = self.max_hands, iou_thres = self.iou_thres)
        box_orig_list = [box_orig_list[i] for i in box_valid_idx]
        bbox_source_Mtr_Minv_list = [bbox_source_Mtr_Minv_list[i] for i in box_valid_idx]
        
        if not get_kp: return [], box_orig_list
        
        kp_orig_list = []
        self.is_hands_list = []
        index = 0
        for bbox, source, Mtr, Minv in bbox_source_Mtr_Minv_list:

            img_landmark = cv2.warpAffine(
                self._im_normalize(img_pad), Mtr, (256,256)
            )

            
            joints, is_hand = self.predict_joints(img_landmark, hand_thres = self.detect_keypoints_thres)
            if not is_hand:
                box_orig_list.pop(index)
                continue

            # projecting keypoints back into original image coordinate space
            kp_orig = (self._pad1(joints) @ Minv.T)[:,:2]
            kp_orig -= pad[::-1]
            
            kp_orig_list.append(kp_orig)
            self.is_hands_list.append(is_hand)
            
            index += 1
        
        return kp_orig_list, box_orig_list