import random

import numpy as np

from image_generator import ImageGenerator
#from models import SSD300
#from models2 import mini_SSD
from models import my_SSD
from utils.prior_box_creator import PriorBoxCreator
from utils.prior_box_manager import PriorBoxManager
from utils.box_visualizer import BoxVisualizer
from utils.XML_parser import XMLParser
from utils.utils import flatten_prior_boxes
from utils.utils import split_data
from utils.utils import read_image
from utils.utils import resize_image
from utils.utils import plot_images

#image_shape = (300, 300, 3)
num_classes = 21
#model =SSD300(image_shape)
model = my_SSD(num_classes)
image_shape = model.input_shape[1:]
box_creator = PriorBoxCreator(model)
prior_boxes = box_creator.create_boxes()

root_prefix = '../datasets/VOCdevkit/VOC2007/'
image_prefix = root_prefix + 'JPEGImages/'
classes_decoder = {0:'aeroplane', 1:'bicyle', 2:'bird', 3:'boat', 4:'bottle',
                   5:'bus', 6:'car', 7:'cat', 8:'chair', 9:'cow',
                   10:'diningtable', 11:'dog', 12:'horse', 13:'motorbike',
                   14:'person' ,15:'pottedplant', 16:'sheep', 17:'sofa',
                   18:'train', 19:'tvmonitor'}
box_visualizer = BoxVisualizer(image_prefix, image_shape[0:2],
                                                classes_decoder)
layer_scale, box_arg = 0, 780
box_coordinates = prior_boxes[layer_scale][box_arg, :, :]
box_visualizer.draw_normalized_box(box_coordinates)

prior_boxes = flatten_prior_boxes(prior_boxes)
box_coordinates = prior_boxes[780]
box_visualizer.draw_normalized_box(box_coordinates)

ground_data_prefix = root_prefix + 'Annotations/'
ground_truth_data = XMLParser(ground_data_prefix).get_data()
selected_key =  random.choice(list(ground_truth_data.keys()))
selected_data = ground_truth_data[selected_key]
selected_box_coordinates = selected_data[:, 0:4]

box_visualizer.draw_normalized_box(selected_data, selected_key)
train_keys, validation_keys = split_data(ground_truth_data, training_ratio=.8)

prior_box_manager = PriorBoxManager(prior_boxes)
# the assigne method should return the same amount of dimensions
# therefore it should also give back the classes
assigned_encoded_boxes = prior_box_manager.assign_boxes(selected_data)
positive_mask = assigned_encoded_boxes[:, 4] != 1
assigned_decoded_boxes = prior_box_manager.decode_boxes(assigned_encoded_boxes)
decoded_positive_boxes = assigned_decoded_boxes[positive_mask, 0:4]
box_visualizer.draw_normalized_box(decoded_positive_boxes, selected_key)

# possible IMPORTANT bug, it seems that the assigned boxes are being repeated
# with the maximum value
print(assigned_decoded_boxes[positive_mask])

batch_size = 10
image_generator = ImageGenerator(ground_truth_data,
                                 prior_box_manager,
                                 batch_size,
                                 image_shape[0:2],
                                 train_keys, validation_keys,
                                 image_prefix,
                                 vertical_flip_probability=0,
                                 horizontal_flip_probability=0.5)

transformed_image = next(image_generator.flow(mode='demo'))[0]['image_array']
transformed_image = np.squeeze(transformed_image[0]).astype('uint8')
original_image = read_image(image_prefix + validation_keys[0])
original_image = resize_image(original_image, image_shape[0:2])
plot_images(original_image, transformed_image)
