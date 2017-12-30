from pycocotools.coco import COCO
from .data_utils import get_class_names
import numpy as np


class COCODataManager(object):

    def __init__(self, annotations_path, images_path, class_names='all'):
        self.coco = COCO(annotations_path)
        self.images_path = images_path
        self.class_names = self._load_classes(class_names)
        self.num_classes = len(self.class_names)
        self._coco_id_to_class_arg = self._get_coco_id_to_class_arg()
        self._coco_ids = list(self._coco_id_to_class_arg.keys())
        self.arg_to_class = self._get_arg_to_class()

        self.data = dict()
        if class_names == 'all':
            self.image_ids = self.coco.getImgIds()
        else:
            self.image_ids = self._get_per_class_image_ids()

    def _load_classes(self, class_names):
        if class_names == 'all':
            class_names = get_class_names('COCO')
        elif 'background' not in class_names:
            class_names = ['background'] + class_names
        return class_names

    def _get_coco_id_to_class_arg(self):
        # check that all classes are valid for COCO API
        class_names_without_background = self.class_names.copy()
        if 'background' in class_names_without_background:
            class_names_without_background.remove('background')

        for class_name in class_names_without_background:
            coco_id = self.coco.getCatIds(class_name)
            if len(coco_id) == 0:
                raise Exception('Invalid class name:', class_name)

        coco_ids = self.coco.getCatIds(self.class_names)
        # the + 1 is to add the background class
        one_hot_ids = list(range(1, len(coco_ids) + 1))
        coco_id_to_class_arg = dict(zip(coco_ids, one_hot_ids))
        return coco_id_to_class_arg

    def _get_per_class_image_ids(self):
        image_ids = []
        class_names_without_background = self.class_names.copy()
        if 'background' in class_names_without_background:
            class_names_without_background.remove('background')
        for class_name in class_names_without_background:
            catIds = self.coco.getCatIds(catNms=[class_name])
            image_ids = image_ids + self.coco.getImgIds(catIds=catIds)
        return image_ids

    def _get_arg_to_class(self):
        class_names_without_background = self.class_names.copy()
        arg_to_class = dict()
        arg_to_class[0] = 'background'
        if 'background' in class_names_without_background:
            class_names_without_background.remove('background')
        for class_name in class_names_without_background:
            coco_id = self.coco.getCatIds(class_name)[0]
            class_arg = self._coco_id_to_class_arg[coco_id]
            arg_to_class[class_arg] = class_name
        return arg_to_class

    def _to_one_hot(self, class_arg):
        one_hot_vector = [0] * self.num_classes
        one_hot_vector[class_arg] = 1
        return one_hot_vector

    def load_data(self):
        for image_id in self.image_ids:
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
                if coco_id not in self._coco_ids:
                    continue
                class_arg = self._coco_id_to_class_arg[coco_id]
                one_hot_class = self._to_one_hot(class_arg)
                coco_coordinates = annotations[object_arg]['bbox']

                x_min = coco_coordinates[0] / width
                y_min = coco_coordinates[1] / height
                x_max = x_min + coco_coordinates[2] / width
                y_max = y_min + coco_coordinates[3] / height

                ground_truth = [x_min, y_min, x_max, y_max] + one_hot_class
                image_ground_truth.append(ground_truth)
            image_ground_truth = np.asarray(image_ground_truth)
            if len(image_ground_truth.shape) == 1:
                image_ground_truth = np.expand_dims(image_ground_truth, 0)
            image_full_path = self.images_path + image_file_name
            self.data[image_full_path] = image_ground_truth
        return self.data
