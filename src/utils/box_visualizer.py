import matplotlib.pyplot as plt
import random
from utils.utils import read_image
from utils.utils import resize_image
from utils.utils import list_files_in_directory

class BoxVisualizer(object):
    def __init__(self, image_prefix, image_size=(300, 300), seed=None):
        self.image_prefix = image_prefix
        self.image_size = image_size
        self.image_paths = list_files_in_directory(self.image_prefix + '*.jpg')
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
        return [x_min, y_min, x_max, y_max]

    def draw_normalized_box(self, box_coordinates, image_key=None, color='r'):
        if image_key == None:
            image_path = self.random_instance.choice(self.image_paths)
        else:
            image_path = self.image_prefix + image_key
        image_array = read_image(image_path)
        image_array = resize_image(image_array, self.image_size)
        figure, axis = plt.subplots(1)
        axis.imshow(image_array)
        num_boxes = len(box_coordinates)
        original_coordinates = self.denormalize_box(box_coordinates)
        x_min = original_coordinates[0]
        y_min = original_coordinates[1]
        x_max = original_coordinates[2]
        y_max = original_coordinates[3]
        box_width = x_max - x_min
        box_height = y_max - y_min
        for box_arg in range(num_boxes):
            rectangle = plt.Rectangle((x_min[box_arg], y_min[box_arg]),
                            box_width[box_arg], box_height[box_arg],
                            linewidth=1, edgecolor=color, facecolor='none')
            axis.add_patch(rectangle)
        plt.show()
