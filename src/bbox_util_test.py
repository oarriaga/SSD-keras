from utils.ssd_utils import BBoxUtility
from utils.XML_parser import XMLParser
from utils.utils import add_variances
from utils.utils import flatten_prior_boxes
from utils.utils import split_data
from ssd import SSD300
from utils.prior_box_creator import PriorBoxCreator
from image_generator import ImageGenerator

num_classes = 21
model = SSD300()
image_shape = model.input_shape[1:]
box_creator = PriorBoxCreator(model)
prior_boxes = box_creator.create_boxes()

root_prefix = '../datasets/VOCdevkit/VOC2007/'
ground_data_prefix = root_prefix + 'Annotations/'
image_prefix = root_prefix + 'JPEGImages/'

ground_truth_manager = XMLParser(ground_data_prefix, background_id=None)
ground_truth_data = ground_truth_manager.get_data()

prior_boxes = flatten_prior_boxes(prior_boxes)
prior_boxes = add_variances(prior_boxes)
print('WTF')
bbox_util = BBoxUtility(num_classes, prior_boxes)

result = bbox_util.assign_boxes(ground_truth_data['000007.jpg'])
train_keys, val_keys = split_data(ground_truth_data, training_ratio=.8)
image_generator = ImageGenerator(ground_truth_data, bbox_util,
                                10, (300,300), train_keys, val_keys,
                                                    image_prefix)
data = next(image_generator.flow(mode='train'))


# test the differences here between you bbox_util
# why can't you train with this ?
