from models.ssd2 import SSD300


model = SSD300()
model_layers = model.layers
model_weights = model.get_weights()
a = -1
for layer_arg, layer in enumerate(model_layers):
    #print(layer_arg)
    #print('Name', layer.name)
    for weights in layer.get_weights():
        a = a + 1
        print(a)
        print('Shape:', weights.shape)
    print('*' * 50)
    #model_layers[layer_arg].set_weights(layer_weighs)

#for arg, a in enumerate(model_weights):
    #print(arg)
    #print(a.shape)
    #print('*' * 50)


