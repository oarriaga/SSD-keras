import pickle
from models.ssd import SSD300

tf_weights = pickle.load(open('tf_w.pkl', 'rb'))
model = SSD300()
for layer_arg, layer in enumerate(model.layers):
    print(layer_arg)
    layer_weights = tf_weights[layer_arg]
    layer.set_weights(layer_weights)


