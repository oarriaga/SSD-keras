import numpy as np
from random import shuffle

from .preprocessing import load_image
from .preprocessing import B_MEAN, G_MEAN, R_MEAN
from .boxes import assign_prior_boxes
from .data_augmentation import SSDAugmentation
from .multiprocessing import threadsafe_generator


class DataGenerator(object):

    def __init__(self, train_data, prior_boxes, batch_size=32, num_classes=21,
                 val_data=None, box_scale_factors=[.1, .1, .2, .2]):

        self.train_data = train_data
        self.val_data = val_data
        self.num_classes = num_classes
        self.prior_boxes = prior_boxes
        self.box_scale_factors = box_scale_factors
        self.batch_size = batch_size

    @threadsafe_generator
    def flow(self, mode='train'):
        if mode == 'train':
            keys = list(self.train_data.keys())
            shuffle(keys)
            ground_truth_data = self.train_data
        elif mode == 'val':
            if self.val_data is None:
                raise Exception('Validation data not given in constructor.')
            keys = list(self.val_data.keys())
            ground_truth_data = self.val_data
        else:
            raise Exception('invalid mode: %s' % mode)

        self.transform = SSDAugmentation(mode, 300, (B_MEAN, G_MEAN, R_MEAN))

        while True:
            inputs = []
            targets = []
            for image_path in keys:
                image_array = load_image(image_path, RGB=False).copy()
                box_data = ground_truth_data[image_path].copy()

                data = (image_array, box_data[:, :4], box_data[:, 4:])
                image_array, box_corners, labels = self.transform(*data)
                box_data = np.concatenate([box_corners, labels], axis=1)

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
