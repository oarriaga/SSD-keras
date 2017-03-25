import numpy as np

class PriorBoxCreator(object):
    def __init__(self, model, layer_type='PriorBox'):
        self.model = model
        self.image_shape = self.model.input_shape[1:3]
        self.layer_type = layer_type
        self.model_configurations = self._get_model_configurations(self.model)

    def _get_model_configurations(self, model):
        model_configurations = []
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            if self.layer_type == layer_type:
                layer_data = {}
                layer_data['layer_width'] = layer.input_shape[1]
                layer_data['layer_height'] = layer.input_shape[2]
                layer_data['min_size'] = layer.min_size
                layer_data['max_size'] = layer.max_size
                layer_data['aspect_ratios'] = layer.aspect_ratios
                layer_data['num_prior'] = len(layer.aspect_ratios)
                model_configurations.append(layer_data)
        return model_configurations

    def create_boxes(self):
        image_width, image_height = self.image_shape
        boxes_parameters = []
        for layer_config in self.model_configurations:
            layer_width = layer_config["layer_width"]
            layer_height = layer_config["layer_height"]
            # RENAME: to num_aspect_ratios
            num_priors = layer_config["num_prior"]
            aspect_ratios = layer_config["aspect_ratios"]
            min_size = layer_config["min_size"]
            max_size = layer_config["max_size"]

            # .5 is to locate every step in the center of the bounding box
            step_x = 0.5 * (float(image_width) / float(layer_width))
            step_y = 0.5 * (float(image_height) / float(layer_height))

            linspace_x = np.linspace(step_x, image_width - step_x, layer_width)
            linspace_y = np.linspace(step_y, image_height - step_y, layer_height)

            centers_x, centers_y = np.meshgrid(linspace_x, linspace_y)
            centers_x = centers_x.reshape(-1, 1)
            centers_y = centers_y.reshape(-1, 1)

            assert(num_priors == len(aspect_ratios))
            prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
            prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))

            box_widths = []
            box_heights = []
            for aspect_ratio in aspect_ratios:
                if aspect_ratio == 1 and len(box_widths) == 0:
                    box_widths.append(min_size)
                    box_heights.append(min_size)
                elif aspect_ratio == 1 and len(box_widths) > 0:
                    box_widths.append(np.sqrt(min_size * max_size))
                    box_heights.append(np.sqrt(min_size * max_size))
                elif aspect_ratio != 1:
                    box_widths.append(min_size * np.sqrt(aspect_ratio))
                    box_heights.append(min_size / np.sqrt(aspect_ratio))
            # we take half of the widths and heights since we are at the center
            box_widths = 0.5 * np.array(box_widths)
            box_heights = 0.5 * np.array(box_heights)

            # Normalize to 0-1
            prior_boxes[:, ::4] -= box_widths
            prior_boxes[:, 1::4] -= box_heights
            prior_boxes[:, 2::4] += box_widths
            prior_boxes[:, 3::4] += box_heights
            prior_boxes[:, ::2] /= image_width
            prior_boxes[:, 1::2] /= image_height
            prior_boxes = prior_boxes.reshape(-1, 4)

            # clip to 0-1
            layer_prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
            #prior_variances = np.tile(variances, (len(prior_boxes),1))
            #boxes_para = np.concatenate((prior_boxes, prior_variances), axis=1)
            boxes_parameters.append(layer_prior_boxes)

        return np.concatenate(boxes_parameters, axis=0)
"""
if __name__ == '__main__':
    from models import SSD300
    from utils.utils import add_variances
    import pickle
    model = SSD300()
    prior_box_creator = PriorBoxCreator(model)
    prior_boxes = prior_box_creator.create_boxes()
    prior_boxes = add_variances(prior_boxes)
    original_prior_boxes = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
    diff = prior_boxes - original_prior_boxes
    print("simi {}, max value {}, min value {}".format(diff.shape, diff.max(), diff.min()))
"""
