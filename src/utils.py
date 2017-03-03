import pickle
import numpy as np

"""
REFERENCES:
[1] SSD: Single Shot Multibox Detector [v5 29 Dec 2016]
"""

def create_prior_box(image_shape, box_configs, variances):
    """ QUESTIONS
    Q: What are the layer widths/heights?
    A: [1] Fig. 2 shows the that the feature maps are of size
    (layer_width,layer_height)

    Q: What are the min_size and max_size?
    A: [1] mentions that s_min = .2 and s_max = .9.as_integer_ratio.
    But it does not correspond to the values in the dictionaries
    max_size - min_size is always 54, except in the first layer.
    The value 270 (300*.9) 270/5 = 54. Probably they just divide this number
    and start adding from 60 (300*.2).

    Q: Why is the number of priors 3/6?
    A: [1] 6 corresponds to [1, 2, 3, 1/2, 1/3]. They probably
    represent the ratio between widths and heights.

    Q: From where is the step value taken from?
    A:
    """

    image_width, image_height = image_shape
    boxes_parameters = []
    for layer_config in box_configs:
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
        prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        prior_variances = np.tile(variances, (len(prior_boxes),1))
        boxes_para = np.concatenate((prior_boxes, prior_variances), axis=1)
        boxes_parameters.append(boxes_para)

    return np.concatenate(boxes_parameters, axis=0)


if __name__ == "__main__":

    box_configs = [
        {'layer_width': 38, 'layer_height': 38, 'num_prior': 3, 'min_size':  30.0,
         'max_size': None, 'aspect_ratios': [1.0, 2.0, 1/2.0]},
        {'layer_width': 19, 'layer_height': 19, 'num_prior': 6, 'min_size':  60.0,
         'max_size': 114.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
        {'layer_width': 10, 'layer_height': 10, 'num_prior': 6, 'min_size': 114.0,
         'max_size': 168.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
        {'layer_width':  5, 'layer_height':  5, 'num_prior': 6, 'min_size': 168.0,
         'max_size': 222.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
        {'layer_width':  3, 'layer_height':  3, 'num_prior': 6, 'min_size': 222.0,
         'max_size': 276.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
        {'layer_width':  1, 'layer_height':  1, 'num_prior': 6, 'min_size': 276.0,
         'max_size': 330.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
    ]
    variances = [0.1, 0.1, 0.2, 0.2]
    image_shape = (300,300)
    boxes_parameters = create_prior_box(image_shape, box_configs ,variances)
    priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
    diff = boxes_parameters - priors
    print("simi {}, max value {}, min value {}".format(diff.shape, diff.max(), diff.min()))
