import keras.backend as K
from keras.layers import Activation
from keras.layers import Convolution2D
from keras.layers import Dropout
#from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import merge
from keras.layers import Reshape
from keras.models import Model
from keras.utils.visualize_util import plot

from layers2 import PriorBox

"""
before any prior box you have to make sure that the size of the tensor
has to be reshaped for (-1, 25). One can achieved this by having multiples
of 25 in the number of kernels
"""

def mini_SSD(input_shape, num_classes=21):

    base_kernel_size = 4 + num_classes
    #aspect_ratios = (1, 2, 3, 1/2, 1/3)
    aspect_ratios = (1, 2, 1/2)
    num_aspect_ratios = len(aspect_ratios)
    input_tensor = Input(shape=input_shape)

    body = Convolution2D(32, 5, 5, border_mode='same')(input_tensor)
    body = Activation('relu')(body)
    body = MaxPooling2D((2, 2))(body)
    body = Dropout(.5)(body)

    body = Convolution2D(32, 5, 5, border_mode='same')(input_tensor)
    body = Activation('relu')(body)
    body = MaxPooling2D((2, 2))(body)
    body = Dropout(.5)(body)

    body = Convolution2D((base_kernel_size * num_aspect_ratios), 3, 3,
                          border_mode='same')(body)
    branch_1 = PriorBox(aspect_ratios)(body)

    body = Convolution2D(32, 3, 3, border_mode='same')(branch_1)
    body = Activation('relu')(body)
    body = MaxPooling2D((2, 2))(body)
    body = Dropout(.5)(body)
    body = Convolution2D((base_kernel_size * num_aspect_ratios), 3, 3,
                          border_mode='same')(body)
    branch_2 = PriorBox(aspect_ratios)(body)

    body = Convolution2D(64, 3, 3, border_mode='same')(branch_2)
    body = Activation('relu')(body)
    body = MaxPooling2D((3, 3))(body)
    body = Dropout(.5)(body)
    body = Convolution2D((base_kernel_size * num_aspect_ratios), 3, 3,
                          border_mode='same')(body)
    branch_3 = PriorBox(aspect_ratios)(body)

    branch_1 = Reshape((-1, 4 + num_classes))(branch_1)
    local_1 = Lambda(lambda x: x[:, :, :4])(branch_1)
    class_1 = Lambda(lambda x: K.softmax(x[:, :, 4:]))(branch_1)

    branch_2 = Reshape((-1, 4 + num_classes))(branch_2)
    local_2 = Lambda(lambda x: x[:, :, :4])(branch_2)
    class_2 = Lambda(lambda x: K.softmax(x[:, :, 4:]))(branch_2)

    branch_3 = Reshape((-1, 4 + num_classes))(branch_3)
    local_3 = Lambda(lambda x: x[:, :, :4])(branch_3)
    class_3 = Lambda(lambda x: K.softmax(x[:, :, 4:]))(branch_3)

    classification_tensor = merge([class_1, class_2, class_3], mode='concat',
                                                            concat_axis=1)

    localization_tensor = merge([local_1, local_2, local_3], mode='concat',
                                                            concat_axis=1)
    output_tensor = merge([localization_tensor, classification_tensor],
                           mode='concat', concat_axis=-1)
    model = Model(input_tensor, output_tensor)
    return model

if __name__ == '__main__':
    input_shape=(100, 100, 3)
    model = mini_SSD(input_shape)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print(model.summary())
    plot(model, 'mini_SSD.png')
    from utils.prior_box_creator import PriorBoxCreator
    from utils.prior_box_manager import PriorBoxManager
    prior_box_creator = PriorBoxCreator(model)
    prior_boxes = prior_box_creator.create_boxes()
    prior_box_manager = PriorBoxManager(prior_boxes)
    print(prior_box_manager.prior_boxes.shape)
