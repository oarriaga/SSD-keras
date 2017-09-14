import matplotlib.pyplot as plt
import numpy as np
import random

from utils.utils import load_image
from utils.utils import list_files_in_directory

class Visualizer(object):
    def __init__(self, image_prefix=None, image_size=(300, 300),
                 arg_to_class=None, seed=None, box_decoder=None):
        self.image_prefix = image_prefix
        self.image_size = image_size
        self.image_paths = None
        if self.image_prefix is not None:
            self.image_paths = list_files_in_directory(self.image_prefix + '*.jpg')
        self.arg_to_class = arg_to_class
        self.box_decoder = box_decoder
        self.random_instance = random.Random()
        if seed != None:
            self.random_instance.seed(seed)

        #new 
        self.num_classes = 21
        self.colors = plt.cm.hsv(np.linspace(0, 1, self.num_classes)).tolist()

    def _denormalize_box(self, box_coordinates, image_size):
        x_min = box_coordinates[:, 0]
        y_min = box_coordinates[:, 1]
        x_max = box_coordinates[:, 2]
        y_max = box_coordinates[:, 3]
        original_image_width, original_image_height = image_size
        x_min = x_min * original_image_width
        y_min = y_min * original_image_height
        x_max = x_max * original_image_width
        y_max = y_max * original_image_height
        return np.concatenate([x_min[:, None], y_min[:, None],
                               x_max[:, None], y_max[:, None]], axis=1)

    def draw_normalized_box(self, box_coordinates, image_array=None):
        image_array = np.squeeze(image_array)
        image_array = image_array.astype('uint8')
        image_size = image_array.shape[0:2]
        image_size = (image_size[1], image_size[0])
        figure, axis = plt.subplots(1)
        axis.imshow(image_array)
        original_coordinates = self._denormalize_box(box_coordinates, image_size)
        x_min = original_coordinates[:, 0]
        y_min = original_coordinates[:, 1]
        x_max = original_coordinates[:, 2]
        y_max = original_coordinates[:, 3]
        width = x_max - x_min
        height = y_max - y_min
        classes = box_coordinates[:, 4:]
        num_boxes = len(box_coordinates)
        for box_arg in range(num_boxes):
            x_min_box = x_min[box_arg]
            y_min_box = y_min[box_arg]
            box_width = width[box_arg]
            box_height = height[box_arg]
            box_class = classes[box_arg]
            label_arg = np.argmax(box_class)
            score = box_class[label_arg]
            class_name = self.arg_to_class[label_arg]
            color = self.colors[label_arg]
            rectangle = plt.Rectangle((x_min_box, y_min_box),
                            box_width, box_height, fill=False,
                            linewidth=2, edgecolor=color)
            axis.add_patch(rectangle)
            display_text = '{:0.2f}, {}'.format(score, class_name)
            axis.text(x_min_box, y_min_box, display_text, style='italic',
                      bbox={'facecolor':color, 'alpha':0.5, 'pad':10})
        plt.show()
