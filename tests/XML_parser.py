import numpy as np
import os
from xml.etree import ElementTree

class XMLParser(object):
    """ Preprocess the VOC2007 xml annotations data.

    # Arguments
        data_path: Data path to VOC2007 annotations

    # Return
        data: Dictionary which keys correspond to the image names
        and values are numpy arrays of shape (num_objects, 4 + num_classes)
        num_objects refers to the number of objects in that specific image
    """

    def __init__(self, data_path, background_id=None, class_names=None):
        self.path_prefix = data_path
        self.background_id = background_id
        if class_names == None:
            self.arg_to_class = self._use_VOC2007_classes()
        else:
            if background_id != None and background_id != -1:
                class_names.insert(background_id, 'background')
            elif background_id == -1:
                class_names.append('background')
            keys = np.arange(len(class_names))
            self.arg_to_class = dict(zip(keys, class_names))

        self.class_to_arg = {value: key for key, value
                             in self.arg_to_class.items()}
        self.data = dict()
        self._preprocess_XML()

    def _use_VOC2007_classes(self):
        class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                       'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                       'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor']
        if self.background_id != None and self.background_id != -1 :
            class_names.insert(self.background_id, 'background')
        elif self.background_id == -1:
            class_names.append('background')

        keys = np.arange(len(class_names))
        arg_to_class = dict(zip(keys, class_names))

        return arg_to_class

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
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text) / width
                    ymin = float(bounding_box.find('ymin').text) / height
                    xmax = float(bounding_box.find('xmax').text) / width
                    ymax = float(bounding_box.find('ymax').text) / height
                bounding_box = [xmin,ymin,xmax,ymax]
                bounding_boxes.append(bounding_box)
                class_name = object_tree.find('name').text
                one_hot_class = self._to_one_hot(class_name)
                one_hot_classes.append(one_hot_class)
            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            if len(bounding_boxes.shape) == 1:
                image_data = np.expand_dims(image_data, axis=0)
            self.data[image_name] = image_data

    def _to_one_hot(self, name):
        num_classes = len(self.class_to_arg)
        one_hot_vector = [0] * num_classes
        class_arg = self.class_to_arg[name]
        one_hot_vector[class_arg] = 1
        return one_hot_vector

if __name__ == '__main__':
    data_path = '../../datasets/VOCdevkit/VOC2007/Annotations/'
    xml_parser = XMLParser(data_path, background_id=0)
    ground_truths = xml_parser.get_data()
    print(xml_parser.arg_to_class)

