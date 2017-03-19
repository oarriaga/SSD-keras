from models import my_SSD
from image_generator import ImageGenerator
from utils.prior_box_creator import PriorBoxCreator
from utils.prior_box_manager import PriorBoxManager
from utils.XML_parser import XMLParser
from utils.utils import flatten_prior_boxes
from utils.utils import split_data

# parameters
root_prefix = '../datasets/VOCdevkit/VOC2007/'
image_prefix = root_prefix + 'JPEGImages/'
ground_data_prefix = root_prefix + 'Annotations/'
model = my_SSD(num_classes=21)
image_shape = model.input_shape[1:]
background_id = 0

ground_truth_manager = XMLParser(ground_data_prefix, background_id)
ground_truth_data = ground_truth_manager.get_data()
box_creator = PriorBoxCreator(model)
prior_boxes = box_creator.create_boxes()
prior_box_manager = PriorBoxManager(prior_boxes, background_id)

prior_boxes = flatten_prior_boxes(prior_boxes)

train_keys, validation_keys = split_data(ground_truth_data, training_ratio=.8)

batch_size = 10
image_generator = ImageGenerator(ground_truth_data,
                                 prior_box_manager,
                                 batch_size,
                                 image_shape[0:2],
                                 train_keys, validation_keys,
                                 image_prefix,
                                 vertical_flip_probability=0,
                                 horizontal_flip_probability=0.5)

data = next(image_generator.flow(mode='train'))




