import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize, imread

class PriorBoxCreator(object):
    def __init__(self, model):
        self.model = model
        self.scale_min = .2
        self.scale_max = .9
        self.layer_scales = None
        self.model_configurations = None

    def create_prior_boxes(self):
        image_width, image_height = self.image_shape
        self.prior_boxes = []
        for layer_arg, configuration in enumerate(self.model_configurations):
            layer_size = configuration['layer_width']
            aspect_ratios = configuration['aspect_ratios']
            layer_scale = self.layer_scales[layer_arg]
            layer_prior_boxes = self._get_prior_boxes(layer_scale, aspect_ratios, layer_size)
            self.prior_boxes.append(layer_prior_boxes)

    def calculate_layer_scales(self):
        num_feature_maps = len(self.model_configurations)
        scales = []
        step = (self.scale_max - self.scale_min)/(num_feature_maps - 1)
        for k in range(1, num_feature_maps + 1):
            scale = self.scale_min + step*(k - 1)
            scales.append(scale)
        self.layer_scales = np.asarray(scales)

    def get_model_configurations(self):
        self.image_shape = self.model.input_shape[1:3]
        self.model_configurations = []
        for layer in self.model.layers:
            layer_type = layer.__class__.__name__
            if layer_type == 'PriorBox':
                layer_data = {}
                layer_data['layer_width'] = layer.input_shape[1]
                layer_data['layer_height'] = layer.input_shape[2]
                layer_data['min_size'] = layer.min_size
                layer_data['max_size'] = layer.max_size
                layer_data['aspect_ratios'] = layer.aspect_ratios
                layer_data['num_prior'] = len(layer.aspect_ratios)
                self.model_configurations.append(layer_data)

    def _get_prior_boxes(self, layer_scale, aspect_ratios, layer_size):
        box_widths = layer_scale * np.sqrt(aspect_ratios)
        box_heights = layer_scale / np.sqrt(aspect_ratios)
        box_centers_x = [(i + 0.5)/layer_size for i in range(layer_size)]
        box_centers_y = [(j + 0.5)/layer_size for j in range(layer_size)]
        box_centers_x = np.asarray(box_centers_x)
        box_centers_y = np.asarray(box_centers_y)
        num_boxes = len(box_centers_x) * len(box_centers_y)
        num_aspect_ratios = len(aspect_ratios)
        num_coordinates = 4 # x_min, y_min, x_max, y_max
        prior_boxes = np.zeros(shape=(num_boxes, num_aspect_ratios,
                                                num_coordinates))
        grid_x, grid_y = np.meshgrid(box_centers_x, box_centers_y)
        grid_x = grid_x.reshape(-1,1)
        grid_y = grid_y.reshape(-1,1)
        # TODO fix this with enumerate(zip)
        aspect_ratio_arg = 0
        for box_width, box_height in zip(box_widths, box_heights):
            x_min = grid_x - (.5 * box_width)
            x_max = grid_x + (.5 * box_width)
            y_min = grid_y - (.5 * box_height)
            y_max = grid_y + (.5 * box_height)
            x_min = np.clip(x_min, 0, 1)
            x_max = np.clip(x_max, 0, 1)
            y_min = np.clip(y_min, 0, 1)
            y_max = np.clip(y_max, 0, 1)
            coordinates = np.concatenate([x_min, y_min, x_max, y_max], axis=1)
            prior_boxes[:, aspect_ratio_arg, :] = coordinates
            aspect_ratio_arg = aspect_ratio_arg + 1
        return prior_boxes

    def create_boxes(self):
        self.get_model_configurations()
        self.calculate_layer_scales()
        self.create_prior_boxes()
        return self.prior_boxes

    def draw_boxes(self, image_path, box_coordinates):
        image_array = self.imread(image_path)
        image_array = self.resize_image(image_array, self.image_shape)
        figure, axis = plt.subplots(1)
        axis.imshow(image_array)
        box_coordinates = np.squeeze(box_coordinates)
        num_boxes = len(box_coordinates)
        decoded_coordinates = self.decode_box(box_coordinates)
        x_min = decoded_coordinates[0]
        y_min = decoded_coordinates[1]
        x_max = decoded_coordinates[2]
        y_max = decoded_coordinates[3]
        box_width = x_max - x_min
        box_height = y_max - y_min
        for box_arg in range(num_boxes):
            rectangle = plt.Rectangle((x_min[box_arg], y_min[box_arg]),
                            box_width[box_arg], box_height[box_arg],
                            linewidth=1, edgecolor='r', facecolor='none')
            axis.add_patch(rectangle)
        plt.show()

    def decode_box(self, box_coordinates):
        box_coordinates = np.squeeze(box_coordinates)
        x_min = box_coordinates[:, 0]
        y_min = box_coordinates[:, 1]
        x_max = box_coordinates[:, 2]
        y_max = box_coordinates[:, 3]
        original_image_width, original_image_height = self.image_shape
        x_min = x_min * original_image_width
        y_min = y_min * original_image_height
        x_max = x_max * original_image_width
        y_max = y_max * original_image_height
        return [x_min, y_min, x_max, y_max]

    def imread(self, image_path):
        return imread(image_path)

    def resize_image(self, image_array, size):
        return imresize(image_array, size)

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../')
    from models import SSD300
    model = SSD300((300,300,3))
    box_creator = PriorBoxCreator(model)
    prior_boxes = box_creator.create_boxes()
    layer_scale, box_arg = 0, 780
    box_coordinates = prior_boxes[layer_scale][box_arg,:,:]
    image_path = '../../images/007040.jpg'
    box_creator.draw_boxes(image_path, box_coordinates)
