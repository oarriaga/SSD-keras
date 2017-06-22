import glob
import os
from xml.etree import ElementTree

import numpy as np
try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

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

class COCOParser(object):
    def __init__(self, annotations_path, class_names='all'):
        self.coco = COCO(annotations_path)
        self.class_names = class_names
        if self.class_names == 'all':
            class_data = self.coco.loadCats(self.coco.getCatIds())
            self.class_names = [class_['name'] for class_ in class_data]
            coco_ids = [class_['id'] for class_ in class_data]
            one_hot_ids = list(range(1, len(coco_ids) + 1))
            self.coco_id_to_class_arg = dict(zip(coco_ids, one_hot_ids))
            self.class_names = ['background'] + self.class_names
            self.num_classes = len(self.class_names)
        elif len(self.class_names) > 1:
            raise NotImplementedError('Only one or all classes supported')
        self.data = dict()

    def get_data(self):
        self._get_data()
        return self.data

    def _get_data(self):
        image_ids = self.coco.getImgIds()
        for image_id in image_ids:
            image_data = self.coco.loadImgs(image_id)[0]
            image_file_name = image_data['file_name']
            width = float(image_data['width'])
            height = float(image_data['height'])
            annotation_ids = self.coco.getAnnIds(imgIds=image_data['id'])
            annotations = self.coco.loadAnns(annotation_ids)
            num_objects_in_image = len(annotations)
            if num_objects_in_image == 0:
                continue
            image_ground_truth = []
            for object_arg in range(num_objects_in_image):
                coco_id = annotations[object_arg]['category_id']
                class_arg = self.coco_id_to_class_arg[coco_id]
                one_hot_class = self._to_one_hot(class_arg)
                coco_coordinates = annotations[object_arg]['bbox']
                #print('coco_coordinates:', coco_coordinates)
                x_min = (coco_coordinates[0]) #/ width
                y_min = (coco_coordinates[1]) #/ height
                x_max = (x_min + coco_coordinates[2]) #/ width
                y_max = (y_min + coco_coordinates[3]) #/ height
                #print('transformed_coordinates:', [x_min, y_min, x_max, y_max])
                x_min = x_min / width
                y_min = y_min / height
                x_max = x_max / width
                y_max = y_max / height
                #print('normalized_coordinates:', [x_min, y_min, x_max, y_max])
                ground_truth = [x_min, y_min, x_max, y_max] + one_hot_class
                image_ground_truth.append(ground_truth)
            image_ground_truth = np.asarray(image_ground_truth)
            if len(image_ground_truth.shape) == 1:
                image_ground_truth = np.expand_dims(image_ground_truth, 0)
            self.data[image_file_name] = image_ground_truth

    def _to_one_hot(self, class_arg):
        one_hot_vector = [0] * self.num_classes
        one_hot_vector[class_arg] = 1
        return one_hot_vector


class DataManager(object):
    """Class for loading VOC2007 and COCO datasets or
    """
    def __init__(self, dataset_name='VOC2007', class_names=None,
                                    dataset_path_prefix=None,
                                    image_prefix=None):

        self.dataset_name = dataset_name
        self.dataset_path_prefix = dataset_path_prefix
        self.image_prefix = image_prefix
        self.class_names = class_names
        self.ground_truth_data = None
        self.parser = None
        if self.dataset_path_prefix == None:
            self.dataset_path_prefix = '../datasets/VOCdevkit/VOC2007/Annotations/'
        else:
            self.dataset_path_prefix = dataset_path_prefix

        if self.image_prefix == None:
            self.image_prefix = '../datasets/VOCdevkit/VOC2007/JPEGImages/'
        else:
            self.image_prefix = image_prefix

        if self.dataset_name == 'VOC2007':
            self._load_VOC2007()
        elif self.dataset_name == 'COCO':
            self.dataset_path_prefix = ('../datasets/COCO/annotations/' +
                                                'instances_train2014.json')
            self._load_COCO()
        elif self.dataset_name == 'all':
            raise NotImplementedError
        else:
            raise Exception('Incorrect dataset name:', self.dataset_name)

    def get_data(self):
        print('Deprecated function use function: load_data instead')
        return self.ground_truth_data

    def load_data(self):
        return self.ground_truth_data

    def _load_VOC2007(self):
        self.parser = XMLParser(self.dataset_path_prefix, self.class_names)
        self.ground_truth_data = self.parser.get_data()
        self.class_names = self.parser.class_names
        self.arg_to_class = self.parser.arg_to_class
        self.class_to_arg = self.parser.class_to_arg

    def _load_COCO(self):
        self.parser = COCOParser(self.dataset_path_prefix)
        self.ground_truth_data = self.parser.get_data()

def get_class_names(dataset_name='VOC2007'):
    if dataset_name == 'VOC2007':
        class_names = ['background','aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person',
                       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == 'COCO':
        class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle',
                        'airplane', 'bus', 'train', 'truck', 'boat',
                        'traffic light', 'fire hydrant', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog',
                        'horse', 'sheep', 'cow', 'elephant', 'bear',
                        'zebra', 'giraffe', 'backpack', 'umbrella',
                        'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat',
                        'baseball glove', 'skateboard', 'surfboard',
                        'tennis racket', 'bottle', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                        'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet',
                        'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                        'cell phone', 'microwave', 'oven', 'toaster',
                        'sink', 'refrigerator', 'book', 'clock', 'vase',
                        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    else:
        raise Exception('Invalid dataset', dataset_name)
    return class_names

def get_arg_to_class(class_names):
    return dict(zip(list(range(len(class_names))), class_names))

def list_files_in_directory(path_name='*'):
    return glob.glob(path_name)







