import numpy as np
from utils.prior_box_manager import PriorBoxManager
from utils.XML_parser import XMLParser
from utils.utils import load_image
from utils.utils import preprocess_images

class Metrics(object):
    def __init__(self, model, dataset_path_prefix=None):
        self.model = model
        self.image_size = (300, 300)
        if dataset_path_prefix is None:
            dataset_path_prefix = '../datasets/VOCtest/VOCdevkit/VOC2007/Annotations/'
            ground_truth_manager = XMLParser(dataset_path_prefix)
            self.ground_truth_data = ground_truth_manager.get_data()

    def calculate_map(self):
        image_keys = sorted(self.ground_truth_data.keys())
        for image_key in image_keys:
            image_path = self.path_prefix + image_key
            image_array = load_image(image_path, False, self.image_size)
            image_array = np.expand_dims(image_array)
            image_array = preprocess_images(image_array, axis=0)
            predicted_boxes = self.model.predict(image_array)
            predicted_boxes = np.squeeze(predicted_boxes)

            ground_truth = self.ground_truth_data[image_key]



        self.box_manager = PriorBoxManager(predicted_boxes)


