from xml.etree import ElementTree

import numpy as np

from .data_utils import get_class_names


class VOCParser(object):
    """ Preprocess the VOC2007 xml annotations data.

    # TODO: Add background label

    # Arguments
        data_path: Data path to VOC2007 annotations

    # Return
        data: Dictionary which keys correspond to the image names
        and values are numpy arrays of shape (num_objects, 4 + num_classes)
        num_objects refers to the number of objects in that specific image
    """

    def __init__(self, dataset_name='VOC2007', split='train',
                 class_names='all', with_difficult_objects=True,
                 dataset_path='../datasets/VOCdevkit/'):

        if dataset_name not in ['VOC2007', 'VOC2012']:
            raise Exception('Invalid dataset name.')

        if split not in ['train', 'val', 'trainval', 'test', 'all']:
            raise Exception('Invalid split name.')

        # creating data set prefix paths variables
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path + dataset_name + '/'
        self.split = split
        self.split_prefix = self.dataset_path + 'ImageSets/Main/'
        self.annotations_path = self.dataset_path + 'Annotations/'
        self.images_path = self.dataset_path + 'JPEGImages/'
        self.with_difficult_objects = with_difficult_objects

        self.class_names = class_names
        if self.class_names == 'all':
            self.class_names = get_class_names(self.dataset_name)
        self.num_classes = len(self.class_names)
        class_keys = np.arange(self.num_classes)
        self.arg_to_class = dict(zip(class_keys, self.class_names))
        self.class_to_arg = {value: key for key, value
                             in self.arg_to_class.items()}
        self.data = dict()
        self.difficult_objects = dict()
        self._preprocess_XML()

    def _load_filenames(self):
        split_file = self.split_prefix + self.split + '.txt'
        splitted_filenames = []
        for line in open(split_file):
            filename = line.strip() + '.xml'
            splitted_filenames.append(filename)
        return splitted_filenames

    def _preprocess_XML(self):
        filenames = self._load_filenames()
        for filename in filenames:
            filename_path = self.annotations_path + filename
            tree = ElementTree.parse(filename_path)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            difficulties = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                difficulty = int(object_tree.find('difficult').text)
                if difficulty == 1 and not(self.with_difficult_objects):
                    continue
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
                    difficulties.append(difficulty)
            if len(one_hot_classes) == 0:
                continue
            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            if len(bounding_boxes.shape) == 1:
                image_data = np.expand_dims(image_data, axis=0)
            self.data[self.images_path + image_name] = image_data
            self.difficult_objects[image_name] = difficulties

    def _to_one_hot(self, class_name):
        one_hot_vector = [0] * self.num_classes
        class_arg = self.class_to_arg[class_name]
        one_hot_vector[class_arg] = 1
        return one_hot_vector

    def load_data(self):
        return self.data
