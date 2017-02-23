import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from scipy.misc import imread, imresize

class ImageGenerator(object):
    """ Image generator with saturation, brightness, lighting, contrast,
    horizontal flip and vertical flip transformations. It supports
    bounding boxes coordinates.

    TODO:
        - Finish preprocess_images method.
        - Add random crop method.
        - Finish support for not using bounding_boxes.
    """
    def __init__(self, ground_truths=None, bounding_box_utils=None,
                 batch_size=None, path_prefix=None,
                 train_keys=None, validation_keys=None, image_size=None,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 horizontal_flip_probability=0.5,
                 vertical_flip_probability=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3./4., 4./3.]):

        self.ground_truths = ground_truths
        self.bounding_box_utils = bounding_box_utils
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
        return np.clip(image_array, 0 ,255)

    def horizontal_flip(self, image_array, box_corners=None):
        if np.random.random() < self.horizontal_flip_probability:
            image_array = image_array[:, ::-1]
            if box_corners != None:
                box_corners[:, [0, 2]] = 1 - box_corners[:, [2, 0]]
                return image_array, box_corners
        return image_array

    def vertical_flip(self, image_array, box_corners=None):
        if (np.random.random() < self.vertical_flip_probability):
            image_array = image_array[::-1]
            if box_corners != None:
                box_corners[:, [1, 3]] = 1 - box_corners[:, [3, 1]]
                return image_array, box_corners
        return image_array

    def apply_transformations(self, image_array, box_corners=None):
        shuffle(self.color_jitter)
        for jitter in self.color_jitter:
            image_array = jitter(image_array)
        if self.lighting_std:
            image_array = self.lighting(image_array)
        if box_corners != None:
            if self.horizontal_flip_probability > 0:
                image_array, box_corners = self.horizontal_flip(
                                                            image_array,
                                                            box_corners)
            if self.vertical_flip_probability > 0:
                image_array, box_corners = self.vertical_flip(image_array,
                                                            box_corners)
        else:
            if self.horizontal_flip_probability > 0:
                image_array = self.horizontal_flip(image_array)
            if self.vertical_flip_probability > 0:
                image_array = self.vertical_flip(image_array)

    def preprocess_images(image_array):
        return image_array

    def flow(self, train=True):
            for i in range(1):

                if train:
                    shuffle(self.train_keys)
                    keys = self.train_keys
                else:
                    shuffle(self.validation_keys)
                    keys = self.validation_keys

                inputs = []
                targets = []
                for key in keys:
                    image_path = self.path_prefix + key
                    image_array = self._imread(image_path)
                    image_array = self._imresize(image_array, self.image_size)
                    image_array = image_array.astype('float32')
                    box_corners = self.ground_truths[key].copy()
                    if train:
                        image_array, box_corners = self.apply_transformations(
                                                                image_array,
                                                                box_corners)
                        box_corners = self.bounding_box_utils.assign_boxes(
                                                                box_corners)
                        inputs.append(image_array)
                        targets.append(box_corners)

                        if len(targets) == self.batch_size:
                            inputs = np.asarray(inputs)
                            targets = np.asarray(targets)
                            yield self.preprocess_images(inputs), targets
                            inputs = []
                            targets = []

    def _imread(self, image_name):
        return imread(image_name)

    def _imresize(self, image_array, size):
        return imresize(image_array, size)

def test_saturation():
    image_name = 'image.jpg'
    generator = ImageGenerator()
    image_array = generator._imread(image_name)
    transformed_image_array = generator.saturation(image_array)
    transformed_image_array = transformed_image_array.astype('uint8')
    plt.imshow(transformed_image_array)
    plt.show()

def test_brightness(brightness_var=.5):
    image_name = 'image.jpg'
    generator = ImageGenerator(brightness_var=brightness_var)
    image_array = generator._imread(image_name)
    transformed_image_array = generator.brightness(image_array)
    transformed_image_array = transformed_image_array.astype('uint8')
    plt.imshow(transformed_image_array)
    plt.show()

def test_contrast():
    image_name = 'image.jpg'
    generator = ImageGenerator()
    image_array = generator._imread(image_name)
    transformed_image_array = generator.contrast(image_array)
    transformed_image_array = transformed_image_array.astype('uint8')
    plt.imshow(transformed_image_array)
    plt.show()

def test_lighting(lighting_std=.5):
    image_name = 'image.jpg'
    generator = ImageGenerator(lighting_std=lighting_std)
    image_array = generator._imread(image_name)
    transformed_image_array = generator.contrast(image_array)
    transformed_image_array = transformed_image_array.astype('uint8')
    plt.imshow(transformed_image_array)
    plt.show()

def test_horizontal_flip():
    image_name = 'image.jpg'
    generator = ImageGenerator()
    image_array = generator._imread(image_name)
    transformed_image_array = generator.horizontal_flip(image_array)
    transformed_image_array = transformed_image_array.astype('uint8')
    plt.imshow(transformed_image_array)
    plt.show()

def test_vertical_flip():
    image_name = 'image.jpg'
    generator = ImageGenerator()
    image_array = generator._imread(image_name)
    transformed_image_array = generator.vertical_flip(image_array)
    transformed_image_array = transformed_image_array.astype('uint8')
    plt.imshow(transformed_image_array)
    plt.show()





