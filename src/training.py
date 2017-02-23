from bounding_boxes_utility import BoundingBoxUtility
from image_generator import ImageGenerator
from utils import create_prior_box
from XML_preprocessor import XML_preprocessor
import numpy as np

image_shape = (300, 300, 3)
box_configs = [
    {'layer_width': 38, 'layer_height': 38, 'num_prior': 3, 'min_size': 30.0,
     'max_size': None, 'aspect_ratios': [1.0, 2.0, 1/2.0]},
    {'layer_width': 19, 'layer_height': 19, 'num_prior': 6, 'min_size': 60.0,
     'max_size': 114.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
    {'layer_width': 10, 'layer_height': 10, 'num_prior': 6, 'min_size': 114.0,
     'max_size': 168.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
    {'layer_width':  5, 'layer_height':  5, 'num_prior': 6, 'min_size': 168.0,
     'max_size': 222.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
    {'layer_width':  3, 'layer_height':  3, 'num_prior': 6, 'min_size': 222.0,
     'max_size': 276.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
    {'layer_width':  1, 'layer_height':  1, 'num_prior': 6, 'min_size': 276.0,
     'max_size': 330.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
]
variances = [0.1, 0.1, 0.2, 0.2]
priors = create_prior_box(image_shape[0:2], box_configs, variances)
num_classes = 21
batch_size = 16
bounding_box_utils = BoundingBoxUtility(num_classes, priors)
data_path = '../datasets/VOCdevkit/VOC2007/'
ground_truth_data = XML_preprocessor(data_path+'Annotations/').data
image_generator = ImageGenerator(ground_truth_data, bounding_box_utils, batch_size,
                                                    data_path+'JPEGImages')
# tests
boxes = ground_truth_data['000009.jpg']
encode_box = bounding_box_utils.encode_box
#calculate_iou = bounding_box_utils.calculate_intersection_over_union
# returns a value for each box for every prior (num_boxes, num_priors)
#iou = np.apply_along_axis(calculate_iou, 1, boxes[:, :4])
#assign_mask = iou[0, :] > .5
num_priors = len(priors)
#encoded_box = np.zeros((num_priors, 4 + 1))
encoded_boxes = np.apply_along_axis(encode_box, 1, boxes[:, :4])
encoded_boxes = encoded_boxes.reshape(-1, num_priors, 5)
best_iou = encoded_boxes[:, :, -1].max(axis=0) #? shouldn't it be axis = 1
best_iou_indices = encoded_boxes[:, :, -1].argmax(axis=0) # ? same here
best_iou_mask = best_iou > 0
best_iou_indices2 = best_iou_indices[best_iou_mask]
num_assigned_boxes = len(best_iou_indices2) # ?
encoded_boxes2 = encoded_boxes[:, best_iou_mask, :]

num_classes = 21
assignment = np.zeros(shape=(num_priors,  4 + num_classes + 8))
assignment[:, 4] = 1.0 # is this the background?

assignment[:, 4][best_iou_mask] = 0
assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_indices2, 4:]
assignment[:, -8][best_iou_mask] = 1



