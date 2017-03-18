import random
from models import my_SSD
from utils.prior_box_creator import PriorBoxCreator
from utils.prior_box_manager import PriorBoxManager
from utils.box_visualizer import BoxVisualizer
from utils.XML_parser import XMLParser
from utils.utils import flatten_prior_boxes
from utils.utils import get_classes

# parameters
root_prefix = '../datasets/VOCdevkit/VOC2007/'
image_prefix = root_prefix + 'JPEGImages/'
ground_data_prefix = root_prefix + 'Annotations/'
model = my_SSD(num_classes=21)
image_shape = model.input_shape[1:]

box_creator = PriorBoxCreator(model)
prior_boxes = box_creator.create_boxes()
VOC2007_classes = get_classes(dataset='VOC2007')
vis = BoxVisualizer(image_prefix, image_shape[0:2], VOC2007_classes)
prior_boxes = flatten_prior_boxes(prior_boxes)

ground_truth_data = XMLParser(ground_data_prefix).get_data()
selected_key =  random.choice(list(ground_truth_data.keys()))
selected_data = ground_truth_data[selected_key]
selected_box_coordinates = selected_data[:, 0:4]


prior_box_manager = PriorBoxManager(prior_boxes)
# the assigne method should return the same amount of dimensions
# therefore it should also give back the classes
# new bug assign boxes mistakes the classes
encoded_boxes = prior_box_manager.assign_boxes(selected_data)
positive_mask = encoded_boxes[:, 4] != 1
vis.draw_normalized_box(encoded_boxes[positive_mask], selected_key)
#decoded_boxes = prior_box_manager.decode_boxes(encoded_boxes)
#decoded_positive_boxes = decoded_boxes[positive_mask]
#vis.draw_normalized_box(decoded_positive_boxes, selected_key)


