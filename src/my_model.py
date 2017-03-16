from keras.layers import Activation
from keras.layers import Convolution2D
from keras.layers import Dropout
#from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import merge
from keras.layers import Reshape
from keras.models import Model

from layers2 import PriorBox

"""
before any prior box you have to make sure that the size of the tensor
has to be reshaped for (-1, 25). One can achieved this by having multiples
of 25 in the number of kernels
"""



def mini_SSD300(input_shape, num_classes=21):
    input_tensor = Input(shape=input_shape)

    body = Convolution2D(32, 3, 3, border_mode='same')(input_tensor)
    body = Activation('relu')(body)
    body = Dropout(.5)(body)
    body = MaxPooling2D((2, 2))(body)
    body = Convolution2D(32, 3, 3, border_mode='same')(body)
    branch_1 = PriorBox()(body)

    body = Convolution2D(32, 3, 3, border_mode='same')(body)
    body = Activation('relu')(body)
    body = Dropout(.5)(body)
    body = MaxPooling2D((2, 2))(body)
    body = Convolution2D(32, 3, 3, border_mode='same')(body)
    branch_2 = PriorBox()(body)

    body = Convolution2D(64, 3, 3, border_mode='same')(body)
    body = Activation('relu')(body)
    body = Dropout(.5)(body)
    body = MaxPooling2D((3, 3))(body)
    body = Convolution2D(64, 3, 3, border_mode='same')(body)
    branch_3 = PriorBox()(body)

    branch_1 = Reshape((-1, 4 + num_classes))(branch_1)
    branch_2 = Reshape((-1, 4 + num_classes))(branch_2)
    branch_3 = Reshape((-1, 4 + num_classes))(branch_3)
    output_tensor = merge([branch_1, branch_2, branch_3], mode='concat', concat_axis=1)
    model = Model(input_tensor, output_tensor)
    return model

def mini_SSD(input_shape, num_classes=21):

    base_kernel_size = 4 + num_classes

    input_tensor = Input(shape=input_shape)

    body = Convolution2D(32, 5, 5, border_mode='same')(input_tensor)
    body = Activation('relu')(body)
    body = Dropout(.5)(body)
    body = MaxPooling2D((2, 2))(body)
    body = Convolution2D((base_kernel_size * 2), 3, 3, border_mode='same')(body)
    branch_1 = PriorBox()(body)

    body = Convolution2D(32, 3, 3, border_mode='same')(body)
    body = Activation('relu')(body)
    body = Dropout(.5)(body)
    body = MaxPooling2D((2, 2))(body)
    body = Convolution2D((base_kernel_size * 3), 3, 3, border_mode='same')(body)
    branch_2 = PriorBox()(body)

    body = Convolution2D(64, 3, 3, border_mode='same')(body)
    body = Activation('relu')(body)
    body = Dropout(.5)(body)
    body = MaxPooling2D((3, 3))(body)
    body = Convolution2D((base_kernel_size * 4), 3, 3, border_mode='same')(body)
    branch_3 = PriorBox()(body)

    branch_1 = Reshape((-1, 4 + num_classes))(branch_1)
    branch_2 = Reshape((-1, 4 + num_classes))(branch_2)
    branch_3 = Reshape((-1, 4 + num_classes))(branch_3)
    output_tensor = merge([branch_1, branch_2], mode='concat', concat_axis=1)
    output_tensor = branch_2
    model = Model(input_tensor, output_tensor)
    return model

if __name__ == '__main__':
    input_shape=(150, 150, 3)
    model = mini_SSD(input_shape)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print(model.summary())

