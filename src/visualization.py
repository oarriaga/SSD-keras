import random
import numpy as np

from image_generator import ImageGenerator
from models import SSD300
from utils.prior_box_creator import PriorBoxCreator
from utils.prior_box_manager import PriorBoxManager
from utils.box_visualizer import BoxVisualizer
from utils.XML_parser import XMLParser
from utils.utils import split_data
from utils.utils import read_image
from utils.utils import resize_image
from utils.utils import plot_images

# BUG fix the flatten function ti give a one to one map when flattened
# TODO: Incorporate the plot images into the box_visualizer class

model =SSD300()
image_shape = model.input_shape[1:3]
box_creator = PriorBoxCreator(model)
prior_boxes = box_creator.create_boxes()

root_prefix = '../datasets/VOCdevkit/VOC2007/'
ground_data_prefix = root_prefix + 'Annotations/'
image_prefix = root_prefix + 'JPEGImages/'

ground_truth_manager = XMLParser(ground_data_prefix, background_id=None)
ground_truth_data = ground_truth_manager.get_data()
class_decoder = ground_truth_manager.arg_to_class

# drawing any set of prior boxes at a given scale
box_visualizer = BoxVisualizer(image_prefix, image_shape, class_decoder)
# layer_scale, box_arg = 0, 777
# box_coordinates = prior_boxes[layer_scale][box_arg, :, :]
# box_visualizer.draw_normalized_box(box_coordinates)

# drawing ground truths
selected_key =  random.choice(list(ground_truth_data.keys()))
selected_data = ground_truth_data[selected_key]
selected_box_coordinates = selected_data[:, 0:4]
box_visualizer.draw_normalized_box(selected_data, selected_key)

# drawing encoded decoded boxes 
prior_box_manager = PriorBoxManager(prior_boxes)
assigned_encoded_boxes = prior_box_manager.assign_boxes(selected_data)
positive_mask = assigned_encoded_boxes[:, 4] != 1
encoded_positive_boxes = assigned_encoded_boxes[positive_mask]
box_visualizer.draw_normalized_box(encoded_positive_boxes, selected_key)
assigned_decoded_boxes = prior_box_manager.decode_boxes(assigned_encoded_boxes)
decoded_positive_boxes = assigned_decoded_boxes[positive_mask]
box_visualizer.draw_normalized_box(decoded_positive_boxes, selected_key)

# drawing generator output
train_keys, validation_keys = split_data(ground_truth_data, training_ratio=.8)
image_generator = ImageGenerator(ground_truth_data,
                                 prior_box_manager,
                                 1, image_shape,
                                 train_keys, validation_keys,
                                 image_prefix,
                                 vertical_flip_probability=0,
                                 horizontal_flip_probability=0.5)

generated_data = next(image_generator.flow(mode='demo'))
generated_input = generated_data[0]['input_1']
generated_output = generated_data[1]['predictions']
transformed_image = np.squeeze(generated_input[0]).astype('uint8')
validation_image_name = image_prefix + validation_keys[0]
original_image = read_image(validation_image_name)
original_image = resize_image(original_image, image_shape)
plot_images(original_image, transformed_image)

# finally draw the assigned boxes given by the generator
generated_encoded_boxes = np.squeeze(generated_output)
generated_boxes = prior_box_manager.decode_boxes(generated_encoded_boxes)
positive_mask = generated_boxes[:, 4] != 1
generated_positive_boxes = generated_boxes[positive_mask]
box_visualizer.draw_normalized_box(generated_positive_boxes, validation_keys[0])
