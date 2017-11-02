import numpy as np
from random import shuffle
import cv2

from .preprocessing import load_image
from .preprocessing import B_MEAN, G_MEAN, R_MEAN
# from .preprocessing import preprocess_images
from .boxes import assign_prior_boxes
from .pytorch_augmentations import SSDAugmentation


class ImageGenerator(object):
    """ Image generator with saturation, brightness, lighting, contrast,
    horizontal flip and vertical flip transformations. It supports
    bounding boxes coordinates.

    TODO:
        - Finish preprocess_images method.
        - Add random crop method.
        - Finish support for not using bounding_boxes.
    """
    # change ground_truth_data to ground_truth_data
    def __init__(self, train_data, val_data, prior_boxes,
                 batch_size=32, box_scale_factors=[.1, .1, .2, .2],
                 num_classes=21):

        # num_classes=21, path_prefix=None):
        self.train_data = train_data
        self.val_data = val_data
        self.num_classes = num_classes
        self.prior_boxes = prior_boxes
        self.box_scale_factors = box_scale_factors
        self.batch_size = batch_size
        # self.path_prefix = path_prefix
        self.transform = SSDAugmentation(300, (B_MEAN, G_MEAN, R_MEAN))

    def flow(self, mode='train'):
        if mode == 'train':
            train_keys = list(self.train_data.keys())
            shuffle(train_keys)
            keys = train_keys
            ground_truth_data = self.train_data
        elif mode == 'val':
            val_keys = list(self.val_data.keys())
            shuffle(val_keys)
            keys = val_keys
            ground_truth_data = self.val_data
        else:
            raise Exception('invalid mode: %s' % mode)

        while True:
            inputs = []
            targets = []
            for key in keys:
                # image_path = self.path_prefix + key
                image_path = key
                image_array = load_image(image_path, RGB=False).copy()
                box_data = ground_truth_data[key].copy()
                if mode == 'train':
                    data = (image_array, box_data[:, :4], box_data[:, 4:])
                    image_array, box_corners, labels = self.transform(*data)
                    # print('box_corners:', box_corners)
                    # print('labels:', labels)
                    box_data = np.concatenate([box_corners, labels], axis=1)
                    # print('box_data:', box_data)
                box_data = assign_prior_boxes(self.prior_boxes, box_data,
                                              self.num_classes,
                                              self.box_scale_factors)
                inputs.append(image_array)
                targets.append(box_data)
                if len(targets) == self.batch_size:
                    inputs = np.asarray(inputs)
                    targets = np.asarray(targets)
                    yield self._wrap_in_dictionary(inputs, targets)
                    inputs = []
                    targets = []

    def _wrap_in_dictionary(self, image_array, targets):
        return [{'input_1': image_array},
                {'predictions': targets}]
