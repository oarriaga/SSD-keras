from models import SSD300
from utils.prior_box_creator import PriorBoxCreator
from utils.prior_box_assigner import PriorBoxAssigner
from utils.XML_parser import XMLParser

image_shape = (300, 300, 3)
model = SSD300(image_shape)
box_creator = PriorBoxCreator(model)
prior_boxes = box_creator.create_boxes()

layer_scale, box_arg = 0, 780
box_coordinates = prior_boxes[layer_scale][box_arg,:,:]
image_path = '../images/'
image_key = '007040.jpg'
box_creator.draw_boxes(image_path + image_key, box_coordinates)

data_path = '../datasets/VOCdevkit/VOC2007/'
ground_truths = XMLParser(data_path+'Annotations/').get_data()
prior_box_manager = PriorBoxAssigner(prior_boxes, ground_truths)
assigned_boxes = prior_box_manager.assign_boxes()
prior_box_manager.draw_assigned_boxes(image_path, image_shape[0:2], image_key)


