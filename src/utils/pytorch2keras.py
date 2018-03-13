import torch
import numpy as np
from models import SSD300


weights_path = '../trained_models/vgg16_reducedfc.pth'
vgg_weights = torch.load(weights_path)

biases = []
weights = []
for name, torch_weights in vgg_weights.items():
    print(name)
    if 'bias' in name:
        biases.append(torch_weights.numpy())
    if 'weight' in name:
        print(torch_weights.numpy().shape)
        conv_weights = torch_weights.numpy()
        conv_weights = np.rollaxis(conv_weights, 0, 4)
        conv_weights = np.rollaxis(conv_weights, 0, 3)
        weights.append(conv_weights)
vgg_weights = list(zip(weights, biases))

base_model = SSD300(return_base=True)
pytorch_layer_arg = 0
for layer in base_model.layers[1:]:
    conv_weights = vgg_weights[pytorch_layer_arg][0]
    bias_weights = vgg_weights[pytorch_layer_arg][1]
    if ('conv2d' in layer.name) or ('branch_2' in layer.name):
        print(layer.name)
        print('pre-trained_weigths:', conv_weights.shape)
        print('model weights:', layer.get_weights()[0].shape)
        layer.set_weights([conv_weights, bias_weights])
        pytorch_layer_arg = pytorch_layer_arg + 1

base_model.save_weights('../trained_models/VGG16_weights.hdf5')
