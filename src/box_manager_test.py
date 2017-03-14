from models import SSD300
from utils.prior_box_manager import PriorBoxManager
from utils.prior_box_creator import PriorBoxCreator
from utils.XML_parser import XMLParser

image_shape = (300, 300, 3)
overlap_threshold = .5
model = SSD300(image_shape)
box_creator = PriorBoxCreator(model)
prior_boxes = box_creator.create_boxes()


data_path = '../datasets/VOCdevkit/VOC2007/'
ground_truths = XMLParser(data_path+'Annotations/').get_data()

prior_box_manager = PriorBoxManager(prior_boxes,
                                    overlap_threshold,
                                    background_id=0)

a = prior_box_manager.assign_boxes(ground_truths['000009.jpg'])

encoded_boxes = prior_box_manager.encoded_boxes
best_iou_indices = prior_box_manager.best_iou_indices
num_assigned_boxes = prior_box_manager.num_assigned_boxes
best_iou_mask = prior_box_manager.best_iou_mask

