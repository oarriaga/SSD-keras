from data_augmentation import SSDAugmentation
from utils.datasets import DataManager
from utils.preprocessing import load_image
from utils.datasets import get_arg_to_class
from utils.visualizer import draw_image_boxes
import cv2
# import numpy as np

size = 300
dataset_name = 'VOC2007'
class_names = ['background', 'aeroplane']
class_to_arg = get_arg_to_class(class_names)
# transformer = SSDAugmentation(size, mean=(104, 117, 123))
transformer = SSDAugmentation(size, mean=(0, 0, 0))
data_manager = DataManager(dataset_name, class_names)
data = data_manager.load_data()
for image_key in data.keys():
    image_path = data_manager.image_prefix + image_key
    print(cv2.imread(image_path))
    image_array = load_image(image_path, (size, size))[0]
    original_image_array = load_image(image_path, None)[0]
    ground_truth_data = data[image_key]
    boxes = ground_truth_data[:, :4]
    labels = ground_truth_data[:, 4:]
    transformed_data = transformer(image_array, boxes, labels)
    image_array, boxes, labels = transformed_data
    # image_array = image_array[:, :, ::-1]
    image_array = image_array.astype('uint8')
    draw_image_boxes(boxes, image_array,
                     class_to_arg, normalized=True)
