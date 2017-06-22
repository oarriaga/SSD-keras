import numpy as np
from random import shuffle

#from .utils import load_image
#from .utils import preprocess_images
from .preprocessing import load_image
from .preprocessing import preprocess_images
from .boxes import assign_prior_boxes

class ImageGenerator(object):
    """ Image generator with saturation, brightness, lighting, contrast,
    horizontal flip and vertical flip transformations. It supports
    bounding boxes coordinates.

    TODO:
        - Finish preprocess_images method.
        - Add random crop method.
        - Finish support for not using bounding_boxes.
    """
    #change ground_truth_data to ground_truth_data
    def __init__(self, ground_truth_data, prior_boxes, num_classes,
                box_scale_factors,
                batch_size, image_size,
                train_keys, validation_keys, path_prefix=None,
                saturation_var=0.5,
                brightness_var=0.5,
                contrast_var=0.5,
                lighting_std=0.5,
                horizontal_flip_probability=0.5,
                vertical_flip_probability=0.5,
                do_crop=True,
                crop_area_range=[0.75, 1.0],
                aspect_ratio_range=[3./4., 4./3.]):

        self.ground_truth_data = ground_truth_data
        self.prior_boxes = prior_boxes
        self.num_classes = num_classes
        self.box_scale_factors = box_scale_factors
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.validation_keys = validation_keys
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range

    def _gray_scale(self, image_array):
        return image_array.dot([0.299, 0.587, 0.114])

    def saturation(self, image_array):
        gray_scale = self._gray_scale(image_array)
        alpha = 2.0 * np.random.random() * self.brightness_var
        alpha = alpha + 1 - self.saturation_var
        image_array = alpha * image_array + (1 - alpha) * gray_scale[:, :, None]
        return np.clip(image_array, 0, 255)

    def brightness(self, image_array):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha = alpha + 1 - self.saturation_var
        image_array = alpha * image_array
        return np.clip(image_array, 0, 255)

    def contrast(self, image_array):
        gray_scale = (self._gray_scale(image_array).mean() *
                        np.ones_like(image_array))
        alpha = 2 * np.random.random() * self.contrast_var
        alpha = alpha + 1 - self.contrast_var
        image_array = image_array * alpha + (1 - alpha) * gray_scale
        return np.clip(image_array, 0, 255)

    def lighting(self, image_array):
        covariance_matrix = np.cov(image_array.reshape(-1,3) /
                                    255.0, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigen_vectors.dot(eigen_values * noise) * 255
        image_array = image_array + noise
        return np.clip(image_array, 0 , 255)

    def horizontal_flip(self, image_array, box_corners):
        if np.random.random() < self.horizontal_flip_probability:
            image_array = image_array[:, ::-1]
            box_corners[:, [0, 2]] = 1 - box_corners[:, [2, 0]]
        return image_array, box_corners

    def vertical_flip(self, image_array, box_corners):
        if (np.random.random() < self.vertical_flip_probability):
            image_array = image_array[::-1]
            box_corners[:, [1, 3]] = 1 - box_corners[:, [3, 1]]
        return image_array, box_corners

    def transform(self, image_array, box_corners):
        shuffle(self.color_jitter)
        for jitter in self.color_jitter:
            image_array = jitter(image_array)

        if self.lighting_std:
            image_array = self.lighting(image_array)

        if self.horizontal_flip_probability > 0:
            image_array, box_corners = self.horizontal_flip(image_array,
                                                            box_corners)

        if self.vertical_flip_probability > 0:
            image_array, box_corners = self.vertical_flip(image_array,
                                                            box_corners)

        return image_array, box_corners

    def flow(self, mode='train'):
            while True:
                if mode =='train':
                    shuffle(self.train_keys)
                    keys = self.train_keys
                elif mode == 'val' or  mode == 'demo':
                    shuffle(self.validation_keys)
                    keys = self.validation_keys
                else:
                    raise Exception('invalid mode: %s' % mode)

                inputs = []
                targets = []
                for key in keys:
                    image_path = self.path_prefix + key
                    image_array = load_image(image_path, self.image_size)
                    box_corners = self.ground_truth_data[key].copy()
                    if mode == 'train' or mode == 'demo':
                        image_array, box_corners = self.transform(image_array,
                                                                box_corners)
                    box_corners = assign_prior_boxes(self.prior_boxes,
                                                    box_corners,
                                                    self.num_classes,
                                                    self.box_scale_factors)
                    inputs.append(image_array)
                    targets.append(box_corners)
                    if len(targets) == self.batch_size:
                        inputs = np.asarray(inputs)
                        targets = np.asarray(targets)
                        if mode == 'train' or mode == 'val':
                            inputs = preprocess_images(inputs)
                            yield self._wrap_in_dictionary(inputs, targets)
                        if mode == 'demo':
                            yield self._wrap_in_dictionary(inputs, targets)
                        inputs = []
                        targets = []

    def _wrap_in_dictionary(self, image_array, targets):
        return [{'input_1':image_array},
                {'predictions':targets}]
