import numpy as np
import os
from xml.etree import ElementTree
from utils.utils import get_class_names

class XMLParser(object):
    """ Preprocess the VOC2007 xml annotations data.

    # TODO: Add background label

    # Arguments
        data_path: Data path to VOC2007 annotations

    # Return
        data: Dictionary which keys correspond to the image names
        and values are numpy arrays of shape (num_objects, 4 + num_classes)
        num_objects refers to the number of objects in that specific image
    """

    def __init__(self, data_path, class_names=None, dataset_name='VOC2007'):
        self.path_prefix = data_path
        self.dataset_name = dataset_name
        self.class_names = class_names
        if self.class_names == None:
            self.class_names = get_class_names(self.dataset_name)
        self.num_classes = len(self.class_names)
        keys = np.arange(self.num_classes)
        self.arg_to_class = dict(zip(keys, self.class_names))
        self.class_to_arg = {value: key for key, value
                             in self.arg_to_class.items()}
        self.data = dict()
        self._preprocess_XML()

    def get_data(self):
        return self.data

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                class_name = object_tree.find('name').text
                if class_name in self.class_names:
                    one_hot_class = self._to_one_hot(class_name)
                    one_hot_classes.append(one_hot_class)
                    for bounding_box in object_tree.iter('bndbox'):
                        xmin = float(bounding_box.find('xmin').text) / width
                        ymin = float(bounding_box.find('ymin').text) / height
                        xmax = float(bounding_box.find('xmax').text) / width
                        ymax = float(bounding_box.find('ymax').text) / height
                    bounding_box = [xmin, ymin, xmax, ymax]
                    bounding_boxes.append(bounding_box)
            if len(one_hot_classes) == 0:
                continue
            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            if len(bounding_boxes.shape) == 1:
                image_data = np.expand_dims(image_data, axis=0)
            self.data[image_name] = image_data

    def _to_one_hot(self, class_name):
        one_hot_vector = [0] * self.num_classes
        class_arg = self.class_to_arg[class_name]
        one_hot_vector[class_arg] = 1
        return one_hot_vector

if __name__ == '__main__':
    data_path = '../../datasets/VOCdevkit/VOC2007/Annotations/'
    classes = ['bottle', 'sofa', 'tvmonitor', 'diningtable', 'chair']
    xml_parser = XMLParser(data_path, class_names=classes)
    ground_truths = xml_parser.get_data()
    print(len(ground_truths.keys()))
    print(xml_parser.arg_to_class)

