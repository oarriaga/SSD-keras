import math
import random
import numpy as np
from keras.utils import Sequence

from .preprocessing import load_image
from .preprocessing import B_MEAN, G_MEAN, R_MEAN
from .boxes import assign_prior_boxes
from .data_augmentation import SSDAugmentation


class SequenceManager(Sequence):

    def __init__(self, data, mode, prior_boxes,
                 batch_size=32, box_scale_factors=[.1, .1, .2, .2],
                 num_classes=21, seed=777):

        if mode not in ['train', 'val']:
            raise Exception('Invalid mode:', mode)

        random.seed(seed)
        self.data = list(data.items())
        random.shuffle(self.data)

        self.mode = mode
        self.num_classes = num_classes
        self.prior_boxes = prior_boxes
        self.box_scale_factors = box_scale_factors
        self.batch_size = batch_size
        self.transform = SSDAugmentation(mode, 300, (B_MEAN, G_MEAN, R_MEAN))

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, batch_arg):
        start_arg, end_arg = self._get_args(batch_arg)
        inputs, targets = [], []
        for sample_arg, batch_data in enumerate(self.data[start_arg:end_arg]):
            image_path, box_data = batch_data[0], batch_data[1]
            image_array = load_image(image_path, RGB=False).copy()
            data = (image_array, box_data[:, :4], box_data[:, 4:])
            image_array, box_corners, labels = self.transform(*data)
            box_data = np.concatenate([box_corners, labels], axis=1)
            box_data = assign_prior_boxes(self.prior_boxes, box_data,
                                          self.num_classes,
                                          self.box_scale_factors)
            inputs.append(image_array)
            targets.append(box_data)
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        return self._wrap_in_dictionary(inputs, targets)

    def _get_args(self, batch_arg):
        start_arg = self.batch_size * batch_arg
        end_arg = self.batch_size * (batch_arg + 1)
        return start_arg, end_arg

    def _wrap_in_dictionary(self, image_array, targets):
        return [{'input_1': image_array},
                {'predictions': targets}]
