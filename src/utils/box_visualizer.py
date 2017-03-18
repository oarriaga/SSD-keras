import matplotlib.pyplot as plt
import numpy as np
import random

from utils.utils import read_image
from utils.utils import resize_image
from utils.utils import list_files_in_directory

class BoxVisualizer(object):
    def __init__(self, image_prefix, image_size=(300, 300),
                 classes_decoder=None, seed=None):
        self.image_prefix = image_prefix
        self.image_size = image_size
        self.image_paths = list_files_in_directory(self.image_prefix + '*.jpg')
        self.classes_decoder = classes_decoder
        self.random_instance = random.Random()
        if seed != None:
            self.random_instance.seed(seed)

    def denormalize_box(self, box_coordinates):
        #num_objects_in_image = box_coordinates.shape[0]
        x_min = box_coordinates[:, 0]
        y_min = box_coordinates[:, 1]
        x_max = box_coordinates[:, 2]
        y_max = box_coordinates[:, 3]
        original_image_width, original_image_height = self.image_size
        x_min = x_min * original_image_width
        y_min = y_min * original_image_height
        x_max = x_max * original_image_width
        y_max = y_max * original_image_height
        return np.concatenate([x_min[:, None], y_min[:, None],
                               x_max[:, None], y_max[:, None]], axis=1)

    def draw_normalized_box(self, box_coordinates, image_key=None, color='r'):
        if len(box_coordinates.shape) == 1:
            box_coordinates = np.expand_dims(box_coordinates, 0)

        if image_key == None:
            image_path = self.random_instance.choice(self.image_paths)
        else:
            image_path = self.image_prefix + image_key

        image_array = read_image(image_path)
        image_array = resize_image(image_array, self.image_size)
        figure, axis = plt.subplots(1)
        axis.imshow(image_array)
        original_coordinates = self.denormalize_box(box_coordinates)
        x_min = original_coordinates[:, 0]
        y_min = original_coordinates[:, 1]
        x_max = original_coordinates[:, 2]
        y_max = original_coordinates[:, 3]
        width = x_max - x_min
        height = y_max - y_min
        if box_coordinates.shape[1] > 4:
            classes = box_coordinates[:, 4:]
            classes_flag = True
        else:
            classes_flag = False

        num_boxes = len(box_coordinates)
        for box_arg in range(num_boxes):
            x_min_box = x_min[box_arg]
            y_min_box = y_min[box_arg]
            box_width = width[box_arg]
            box_height = height[box_arg]
            x_text = x_min_box + (1 * box_width )
            y_text = y_min_box #+ (1 * box_height )

            rectangle = plt.Rectangle((x_min_box, y_min_box),
                            box_width, box_height,
                            linewidth=1, edgecolor=color, facecolor='none')
            axis.add_patch(rectangle)
            if self.classes_decoder != None and classes_flag:
                box_class = classes[box_arg]
                class_name = self.classes_decoder[np.argmax(box_class)]
                axis.text(x_text, y_text, class_name, style='italic',
                      bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
        plt.show()
