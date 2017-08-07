import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

from .utils import load_image
from .utils import list_files_in_directory
from .boxes import apply_non_max_suppression
from .boxes import denormalize_box

class BoxVisualizer(object):
    def __init__(self, image_prefix=None, image_size=(300, 300),
                 arg_to_class=None, seed=None, box_decoder=None):
        self.image_prefix = image_prefix
        self.image_size = image_size
        self.image_paths = list_files_in_directory(self.image_prefix + '*.jpg')
        self.arg_to_class = arg_to_class
        self.box_decoder = box_decoder
        self.random_instance = random.Random()
        if seed != None:
            self.random_instance.seed(seed)

    def denormalize_box(self, box_coordinates):
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

    def draw_normalized_box(self, box_coordinates, image_key=None, color='r',
                                                            image_array=None):
        if len(box_coordinates.shape) == 1:
            box_coordinates = np.expand_dims(box_coordinates, 0)

        if image_key == None:
            image_path = self.random_instance.choice(self.image_paths)
        else:
            image_path = self.image_prefix + image_key
        if image_array is None:
            #image_array = read_image(image_path)
            #image_array = resize_image(image_array, self.image_size)
            image_array = load_image(image_path, target_size=self.image_size)
        image_array = image_array.astype('uint8')
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
            if self.arg_to_class != None and classes_flag:
                box_class = classes[box_arg]
                class_name = self.arg_to_class[np.argmax(box_class)]
                axis.text(x_text, y_text, class_name, style='italic',
                      bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
        plt.show()

def draw_normalized_box(box_data, original_image_array, arg_to_class, colors, font):
    image_array = np.squeeze(original_image_array)
    image_array = image_array.astype('uint8')
    image_size = image_array.shape[0:2]
    image_size = (image_size[1], image_size[0])
    box_classes = box_data[:, 4:]
    box_coordinates = box_data[:, 0:4]
    original_coordinates = denormalize_box(box_coordinates, image_size)
    box_data = np.concatenate([original_coordinates, box_classes], axis=-1)
    box_data = apply_non_max_suppression(box_data)

    if len(box_data) == 0:
        return
    figure, axis = plt.subplots(1)
    axis.imshow(image_array)
    x_min = box_data[:, 0]
    y_min = box_data[:, 1]
    x_max = box_data[:, 2]
    y_max = box_data[:, 3]
    width = x_max - x_min
    height = y_max - y_min
    classes = box_data[:, 4:]
    num_boxes = len(box_data)
    for box_arg in range(num_boxes):
        x_min_box = int(x_min[box_arg])
        y_min_box = int(y_min[box_arg])
        box_width = int(width[box_arg])
        box_height = int(height[box_arg])
        box_class = classes[box_arg]
        label_arg = np.argmax(box_class)
        score = box_class[label_arg]
        class_name = arg_to_class[label_arg]
        color = colors[label_arg]
        display_text = '{:0.2f}, {}'.format(score, class_name)
        cv2.rectangle(original_image_array, (x_min_box, y_min_box),
                    (x_min_box + box_width, y_min_box + box_height),
                                                            color, 2)
        cv2.putText(original_image_array, display_text,
                    (x_min_box, y_min_box - 30), font,
                    .7, color, 1, cv2.LINE_AA)



