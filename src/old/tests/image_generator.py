import numpy as np
from keras.applications.vgg16 import preprocess_input
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
    #change ground_truth_data to ground_truth_data
    def __init__(self, ground_truth_data, box_manager, batch_size, image_size,
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
        self.box_manager = box_manager
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

    def preprocess_images(self, image_array):
        return preprocess_input(image_array)

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
                    image_array = self._imread(image_path)
                    image_array = self._imresize(image_array, self.image_size)
                    image_array = image_array.astype('float32')
                    box_corners = self.ground_truth_data[key].copy()
                    if mode == 'train' or mode == 'demo':
                        image_array, box_corners = self.transform(image_array,
                                                                box_corners)
                    box_corners = self.box_manager.assign_boxes(box_corners)
                    inputs.append(image_array)
                    targets.append(box_corners)
                    if len(targets) == self.batch_size:
                        inputs = np.asarray(inputs)
                        targets = np.asarray(targets)
                        if mode == 'train' or mode == 'val':
                            inputs = self.preprocess_images(inputs)
                            yield self._wrap_in_dictionary2(inputs, targets)
                        if mode == 'demo':
                            yield self._wrap_in_dictionary2(inputs, targets)
                        inputs = []
                        targets = []

    def _wrap_in_dictionary2(self, image_array, targets):
        return [{'input_1':image_array},
                {'predictions':targets}]

    def _wrap_in_dictionary(self, image_array, targets):
        return [{'image_array':image_array},
                {'encoded_box':targets[:,:,:4],
                'classes':targets[:,:,4:]}]

    def _imread(self, image_name):
        return imread(image_name)

    def _imresize(self, image_array, size):
        return imresize(image_array, size)

if __name__ == '__main__':
    import random
    from models import my_SSD
    from utils.prior_box_creator import PriorBoxCreator
    from utils.prior_box_manager import PriorBoxManager
    #from utils.box_visualizer import BoxVisualizer
    from utils.XML_parser import XMLParser
    from utils.utils import split_data
    from utils.utils import read_image
    from utils.utils import resize_image
    #from utils.utils import plot_images

    #image_shape = (300, 300, 3)
    num_classes = 21
    #model =SSD300(image_shape)
    model = my_SSD(num_classes)
    image_shape = model.input_shape[1:]
    box_creator = PriorBoxCreator(model)
    prior_boxes = box_creator.create_boxes()

    root_prefix = '../datasets/VOCdevkit/VOC2007/'
    image_prefix = root_prefix + 'JPEGImages/'
    #box_visualizer = BoxVisualizer(image_prefix, image_shape[0:2])

    layer_scale, box_arg = 0, 780
    box_coordinates = prior_boxes[layer_scale][box_arg, :, :]
    #box_visualizer.draw_normalized_box(box_coordinates)

    ground_data_prefix = root_prefix + 'Annotations/'
    ground_truth_data = XMLParser(ground_data_prefix).get_data()
    random_key =  random.choice(list(ground_truth_data.keys()))
    selected_data = ground_truth_data[random_key]
    selected_box_coordinates = selected_data[:, 0:4]

    #box_visualizer.draw_normalized_box(selected_box_coordinates, random_key)
    train_keys, validation_keys = split_data(ground_truth_data, training_ratio=.8)

    prior_box_manager = PriorBoxManager(prior_boxes)
    assigned_encoded_boxes = prior_box_manager.assign_boxes(selected_data)
    positive_mask = assigned_encoded_boxes[:, -8] > 0
    assigned_decoded_boxes = prior_box_manager.decode_boxes(assigned_encoded_boxes)
    decoded_positive_boxes = assigned_decoded_boxes[positive_mask, 0:4]
    #box_visualizer.draw_normalized_box(decoded_positive_boxes, random_key)

    batch_size = 10
    image_generator = ImageGenerator(ground_truth_data,
                                     prior_box_manager,
                                     batch_size,
                                     image_shape[0:2],
                                     train_keys, validation_keys,
                                     image_prefix,
                                     vertical_flip_probability=0,
                                     horizontal_flip_probability=0.5)

    generated_data = next(image_generator.flow(mode='train'))
    transformed_image = generated_data[0]['image_array']
    transformed_image = next(image_generator.flow(mode='train'))[0]['image_array']
    transformed_image = np.squeeze(transformed_image[0]).astype('uint8')
    original_image = read_image(image_prefix + train_keys[0])
    original_image = resize_image(original_image, image_shape[0:2])
    #plot_images(original_image, transformed_image)
    import numpy as np
    classes = generated_data[1]['classes']
    encoded_boxes = generated_data[1]['encoded_box']
    masks = classes[:,:,0] != 1
    assigned_classes = classes[masks]
    assigned_boxes = encoded_boxes[masks]
    print(assigned_classes[0])



