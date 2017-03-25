from models import SSD300


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

def get_prior_parameters(model):
    box_configurations = []
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == 'PriorBox':
            layer_data = {}
            layer_data['layer_width'] = layer.input_shape[1]
            layer_data['layer_height'] = layer.input_shape[2]
            layer_data['min_size'] = layer.min_size
            layer_data['max_size'] = layer.max_size
            layer_data['aspect_ratios'] = layer.aspect_ratios
            box_configurations.append(layer_data)
    return box_configurations

image_shape = (300, 300, 3)
model = SSD300(image_shape)
box_configurations = get_prior_parameters(image_shape, )

