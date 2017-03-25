import numpy as np

class PriorBoxCreator(object):
    def __init__(self, model, layer_type='PriorBox'):
        self.model = model
        self.image_shape = self.model.input_shape[1:3]
        self.layer_type = layer_type
        self.scale_min = .2
        self.scale_max = .9
        self.layer_scales = None
        self.model_configurations = None

    def _create_prior_boxes(self):
        image_width, image_height = self.image_shape
        self.prior_boxes = []
        for layer_arg, configuration in enumerate(self.model_configurations):
            layer_size = configuration['layer_width']
            aspect_ratios = configuration['aspect_ratios']
            layer_scale = self.layer_scales[layer_arg]
            layer_prior_boxes = self._get_prior_boxes(layer_scale, aspect_ratios, layer_size)
            self.prior_boxes.append(layer_prior_boxes)

    def _calculate_layer_scales(self):
        num_feature_maps = len(self.model_configurations)
        scales = []
        step = (self.scale_max - self.scale_min)/(num_feature_maps - 1)
        for k in range(1, num_feature_maps + 1):
            scale = self.scale_min + step*(k - 1)
            scales.append(scale)
        self.layer_scales = np.asarray(scales)

    def _get_model_configurations(self):
        self.model_configurations = []
        for layer in self.model.layers:
            layer_type = layer.__class__.__name__
            if self.layer_type == layer_type:
                layer_data = {}
                layer_data['layer_width'] = layer.input_shape[1]
                layer_data['layer_height'] = layer.input_shape[2]
                #layer_data['min_size'] = layer.min_size
                #layer_data['max_size'] = layer.max_size
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
        self._get_model_configurations()
        self._calculate_layer_scales()
        self._create_prior_boxes()
        return self.prior_boxes
