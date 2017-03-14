import numpy as np
import random

from image_generator import ImageGenerator
from models import SSD300
from utils.prior_box_creator import PriorBoxCreator
#from utils.prior_box_manager import PriorBoxManager
from utils.box_visualizer import BoxVisualizer
from utils.XML_parser import XMLParser
from utils.utils import split_data
from utils.utils import read_image
from utils.utils import resize_image
from utils.utils import plot_images

image_shape = (300, 300, 3)
model =SSD300(image_shape)
box_creator = PriorBoxCreator(model)
prior_boxes = box_creator.create_boxes()

root_prefix = '../datasets/VOCdevkit/VOC2007/'
image_prefix = root_prefix + 'JPEGImages/'
box_visualizer = BoxVisualizer(image_prefix, image_shape[0:2])

layer_scale, box_arg = 0, 780
box_coordinates = prior_boxes[layer_scale][box_arg, :, :]
box_visualizer.draw_normalized_box(box_coordinates)

ground_data_prefix = root_prefix + 'Annotations/'
ground_truth_data = XMLParser(ground_data_prefix).get_data()
random_key =  random.choice(list(ground_truth_data.keys()))
ground_truth_box_coordinates = ground_truth_data[random_key][:,0:4]
box_visualizer.draw_normalized_box(ground_truth_box_coordinates, random_key)

train_keys, validation_keys = split_data(ground_truth_data, training_ratio=.8)

batch_size =7
image_generator = ImageGenerator(ground_truth_data, batch_size,
                                 image_shape[0:2],
                                 train_keys, validation_keys,
                                 image_prefix,
                                 vertical_flip_probability=0,
                                 horizontal_flip_probability=0.5)

transformed_image = next(image_generator.flow(mode='demo'))[0]
transformed_image = np.squeeze(transformed_image[0]).astype('uint8')
original_image = read_image(image_prefix + validation_keys[0])
original_image = resize_image(original_image, image_shape[0:2])
plot_images(original_image, transformed_image)

