from bounding_boxes_utility import BoundingBoxUtility
from image_generator import ImageGenerator
from utils import create_prior_box
from XML_preprocessor import XML_preprocessor

# constants
image_shape = (300, 300, 3)
num_classes = 21
batch_size = 16
variances = [0.1, 0.1, 0.2, 0.2]
training_data_ratio = .8
data_path = '../datasets/VOCdevkit/VOC2007/'

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

priors = create_prior_box(image_shape[0:2], box_configs, variances)
bounding_box_utils = BoundingBoxUtility(num_classes, priors)
ground_truth_data = XML_preprocessor(data_path+'Annotations/').data

keys = sorted(ground_truth_data.keys())
num_train = int(round(training_data_ratio * len(keys)))
train_keys = keys[:num_train]
validation_keys = keys[num_train:]
num_val = len(validation_keys)

image_generator = ImageGenerator(ground_truth_data, bounding_box_utils,
                                batch_size, data_path+'JPEGImages/',
                                train_keys, validation_keys, image_shape[:2])

gen = image_generator.flow(True)
next(gen)

