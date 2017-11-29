import torch
import numpy as np
from models.modified_VGG import modified_VGG16

weights_path = 'vgg16_reducedfc.pth'
vgg_weights = torch.load(weights_path)

biases = []
weights = []
for name, torch_weights in vgg_weights.items():
    print(name)
    if 'bias' in name:
        biases.append(torch_weights.numpy())
    if 'weight' in name:
        conv_weights = torch_weights.numpy()
        conv_weights = np.rollaxis(conv_weights, 0, 4)
        conv_weights = np.rollaxis(conv_weights, 0, 3)
        weights.append(conv_weights)
vgg_weights = list(zip(weights, biases))
model = modified_VGG16()

pytorch_layer_arg = 0
for layer in model.layers[1:]:
    conv_weights = vgg_weights[pytorch_layer_arg][0]
    bias_weights = vgg_weights[pytorch_layer_arg][1]
    if 'pool' not in layer.name:
        print('pre-trained_weigths:', conv_weights.shape)
        print('model weights:', layer.get_weights()[0].shape)
        layer.set_weights([conv_weights, bias_weights])
        pytorch_layer_arg = pytorch_layer_arg + 1

model.save_weights('VGG16_weights.hdf5')
