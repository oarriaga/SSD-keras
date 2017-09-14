import numpy as np
from ssdv2 import SSD300
import torch
import pickle

weights_path = '../../trained_models/pytorch/ssd_300_VOC0712.pth'
torch_model_weights = torch.load(weights_path)
torch_weights = []
for torch_layer_weights in torch_model_weights.values():
    numpy_layer_weights = torch_layer_weights.numpy()
    if len(numpy_layer_weights.shape) > 1:
        numpy_layer_weights = np.rollaxis(numpy_layer_weights, 0, 4)
        numpy_layer_weights = np.rollaxis(numpy_layer_weights, 0, 3)
    torch_weights.append(numpy_layer_weights)


#model = SSD300()
#model_layers = model.layers
#keras_weight_shape = []
#for layer in model_layers:
#    for weights in layer.get_weights():
#        keras_weight_shape.append(weights.shape)
#
#for weights in torch_weights[1:44]:
#    print(weights.shape)
#    print(keras_weight_shape[0:43])
#
#for weights in [torch_weights[0]]:
#    print(weights.shape)
#    print(keras_weight_shape[44])
#
#for weights in torch_weights[45:46]:
#    print(weights.shape)
#    print(keras_weight_shape[45:46])
#
#for weights in torch_weights[59:70]:
#    print(weights.shape)
#    print(keras_weight_shape[47:58])
#
#for weights in torch_weights[47:58]:
#    print(weights.shape)
#    print(keras_weight_shape[59:70])
#
keras_weights = (torch_weights[1:45] + [torch_weights[0]] +
                torch_weights[45:47] + torch_weights[59:71] +
                torch_weights[47:59])

pickle.dump(keras_weights, open('SSD300v2_weights.pkl', 'wb'))
#model = SSD300()
#model.set_weights(keras_weights)

"""
model_weights = 0
pytorch_weights = []
for layer_arg, layer_weights in enumerate(numpy_model_weights):
    if len(layer_weights.shape) > 1:
        layer_weights = np.rollaxis(layer_weights, 0, 4)
        layer_weights = np.rollaxis(layer_weights, 0, 3)
    print(layer_arg, layer_weights.shape)
    #print('Put in:', layer_weights.shape)
    num_weights = np.prod(layer_weights.shape)
    #print(num_weights)
    model_weights += num_weights
    #print('Space', model_weights[layer_arg].shape)
    #print('*' * 50)
    #model_layers[layer_arg].set_weights(layer_weights)

#keras_weights = []
#numpy_model_weights[0]
"""

