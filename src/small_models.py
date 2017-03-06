from layers import Normalize, PriorBox
from keras.layers import Activation
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Merge
from keras.layers import Reshape
from keras.models import Model
import keras.backend as K


def simple_SSD(input_shape, num_classes, min_size, num_priors,
                max_size, aspect_ratios, variances):
    input_tensor = Input(shape=input_shape)

    body = Convolution2D(16, 7, 7)(input_tensor)
    body = Activation('relu')(body)
    body = MaxPooling2D(2, 2, border_mode='valid')(body)

    body = Convolution2D(32, 5, 5)(body)
    body = Activation('relu')(body)
    branch_1 = MaxPooling2D(2, 2, border_mode='valid')(body)

    body = Convolution2D(64, 3, 3)(branch_1)
    body = Activation('relu')(body)
    branch_2 = MaxPooling2D(2, 2, border_mode='valid')(body)

    # first branch
    norm_1 = Normalize(20)(branch_1)
    localization_1 = Convolution2D(num_priors * 4, 3, 3, border_mode='same')(norm_1)
    localization_1 = Flatten(localization_1)
    classification_1 = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same')(norm_1)
    classification_1 = Flatten()(classification_1)
    prior_boxes_1 = PriorBox(input_shape[0:2], min_size, max_size, aspect_ratios)

    # second branch
    norm_2 = Normalize(20)(branch_2)
    localization_2 = Convolution2D(num_priors * 4, 3, 3, border_mode='same')(norm_2)
    localization_2 = Flatten(localization_2)
    classification_2 = Convolution2D(num_priors * num_classes, 3, 3, border_mode='same')(norm_2)
    classification_2 = Flatten()(classification_2)
    prior_boxes_2 = PriorBox(input_shape[0:2], min_size, max_size, aspect_ratios)

    localization_head = Merge([localization_1, localization_2],
                        mode='concat', concat_axis=1)

    classification_head = Merge([classification_1, classification_2],
                            mode='concat', concat_axis=1)

    prior_boxes_head = Merge([prior_boxes_1, prior_boxes_2],
                            mode='concat', concat_axis=1)

    if hasattr(localization_head, '_keras_shape'):
        num_boxes = localization_head._keras_shape[-1] // 4
    elif hasattr(localization_head, 'int_shape'):
        num_boxes = K.int_shape(localization_head)[-1] // 4

    localization_head = Reshape((num_boxes, 4))(localization_head)
    classification_head = Reshape((num_boxes, num_classes))(classification_head)
    classification_head = Activation('softmax')(classification_head)
    predictions = Merge(localization_head,
                               classification_head,
                               prior_boxes_head,
                               mode='concat', concat_axis=2)

    model = Model(input_tensor, predictions)

    return model
















