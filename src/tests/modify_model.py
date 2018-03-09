from keras.models import load_model
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import concatenate
from keras.layers import Activation
from keras.layers import Reshape
import keras.backend as K

filepath = '../trained_models/300x300/weights.17-1.00.hdf5'
model = load_model(filepath)

input_layer = model.input

layer_1 = model.get_layer('concatenate_3').output
layer_2 = model.get_layer('concatenate_12').output
layer_3 = model.get_layer('concatenate_18').output

num_priors = [4, 6, 6]
num_classes = 21

boxes_1_class = Conv2D(num_priors[0] * num_classes, (3, 3))(layer_1)
boxes_1_flat_class = Flatten()(boxes_1_class)
boxes_2_class = Conv2D(num_priors[1] * num_classes, (3, 3))(layer_2)
boxes_2_flat_class = Flatten()(boxes_2_class)
boxes_3_class = Conv2D(num_priors[2] * num_classes, (3, 3))(layer_3)
boxes_3_flat_class = Flatten()(boxes_3_class)

boxes_1_loc = Conv2D(num_priors[0] * 4, (3, 3))(layer_1)
boxes_1_flat_loc = Flatten()(boxes_1_loc)
boxes_2_loc = Conv2D(num_priors[1] * 4, (3, 3))(layer_2)
boxes_2_flat_loc = Flatten()(boxes_2_loc)
boxes_3_loc = Conv2D(num_priors[2] * 4, (3, 3))(layer_3)
boxes_3_flat_loc = Flatten()(boxes_3_loc)


mbox_conf = concatenate([boxes_1_flat_class,
                         boxes_2_flat_class,
                         boxes_3_flat_class],
                        axis=1, name='concat_ssd_1')

mbox_loc = concatenate([boxes_1_flat_loc,
                        boxes_2_flat_loc,
                        boxes_3_flat_loc],
                       axis=1, name='concat_ssd_2')

num_boxes = K.int_shape(mbox_loc)[-1] // 4
mbox_loc = Reshape((num_boxes, 4))(mbox_loc)
mbox_conf = Reshape((num_boxes, num_classes))(mbox_conf)
mbox_conf = Activation('softmax', name='hola')(mbox_conf)
predictions = concatenate([mbox_loc, mbox_conf],
                          axis=2, name='predictions')

model2 = Model(inputs=input_layer, outputs=predictions)


