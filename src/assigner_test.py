import random
from models import my_SSD
from utils.prior_box_creator import PriorBoxCreator
from utils.prior_box_manager import PriorBoxManager
from utils.box_visualizer import BoxVisualizer
from utils.XML_parser import XMLParser
from utils.utils import flatten_prior_boxes

# parameters
root_prefix = '../datasets/VOCdevkit/VOC2007/'
image_prefix = root_prefix + 'JPEGImages/'
ground_data_prefix = root_prefix + 'Annotations/'
model = my_SSD(num_classes=21)
image_shape = model.input_shape[1:]
background_id = 0

ground_truth_manager = XMLParser(ground_data_prefix, background_id)
ground_truth_data = ground_truth_manager.get_data()
VOC2007_decoder = ground_truth_manager.arg_to_class

box_creator = PriorBoxCreator(model)
prior_boxes = box_creator.create_boxes()
vis = BoxVisualizer(image_prefix, image_shape[0:2], VOC2007_decoder)
prior_boxes = flatten_prior_boxes(prior_boxes)


selected_key =  random.choice(list(ground_truth_data.keys()))
selected_data = ground_truth_data[selected_key]
selected_box_coordinates = selected_data[:, 0:4]

prior_box_manager = PriorBoxManager(prior_boxes, background_id)
encoded_boxes = prior_box_manager.assign_boxes(selected_data)
positive_mask = encoded_boxes[:, 4 + background_id] != 1
# it seems to work but the dict is not counting the background level id
vis.draw_normalized_box(encoded_boxes[positive_mask], selected_key)
decoded_boxes = prior_box_manager.decode_boxes(encoded_boxes)
vis.draw_normalized_box(decoded_boxes[positive_mask], selected_key)

vis.draw_normalized_box(prior_boxes[positive_mask], selected_key)
