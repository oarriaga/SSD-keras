from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.models import Model


def modified_VGG16(input_shape=(300, 300, 3), num_classes=21,
                   weights_path=None, frozen_layers=None, randomize_top=False):

    input_layer = Input(shape=input_shape)

    # Block 1 -----------------------------------------------
    conv1_1 = Conv2D(64, (3, 3),
                     name='conv1_1',
                     padding='same',
                     activation='relu')(input_layer)

    conv1_2 = Conv2D(64, (3, 3),
                     name='conv1_2',
                     padding='same',
                     activation='relu')(conv1_1)

    pool1 = MaxPooling2D(name='pool1',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same', )(conv1_2)

    # Block 2 ----------------------------------------------
    conv2_1 = Conv2D(128, (3, 3),
                     name='conv2_1',
                     padding='same',
                     activation='relu')(pool1)

    conv2_2 = Conv2D(128, (3, 3),
                     name='conv2_2',
                     padding='same',
                     activation='relu')(conv2_1)

    pool2 = MaxPooling2D(name='pool2',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(conv2_2)

    # Block 3 ----------------------------------------------
    conv3_1 = Conv2D(256, (3, 3),
                     name='conv3_1',
                     padding='same',
                     activation='relu')(pool2)

    conv3_2 = Conv2D(256, (3, 3),
                     name='conv3_2',
                     padding='same',
                     activation='relu')(conv3_1)

    conv3_3 = Conv2D(256, (3, 3),
                     name='conv3_3',
                     padding='same',
                     activation='relu')(conv3_2)

    pool3 = MaxPooling2D(name='pool3',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(conv3_3)

    # Block 4 ---------------------------------------------
    conv4_1 = Conv2D(512, (3, 3),
                     name='conv4_1',
                     padding='same',
                     activation='relu')(pool3)

    conv4_2 = Conv2D(512, (3, 3),
                     name='conv4_2',
                     padding='same',
                     activation='relu')(conv4_1)

    conv4_3 = Conv2D(512, (3, 3),
                     name='conv4_3',
                     padding='same',
                     activation='relu')(conv4_2)

    pool4 = MaxPooling2D(name='pool4',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(conv4_3)

    # Block 5 --------------------------------------------
    conv5_1 = Conv2D(512, (3, 3),
                     name='conv5_1',
                     padding='same',
                     activation='relu')(pool4)

    conv5_2 = Conv2D(512, (3, 3),
                     name='conv5_2',
                     padding='same',
                     activation='relu')(conv5_1)

    conv5_3 = Conv2D(512, (3, 3),
                     name='conv5_3',
                     padding='same',
                     activation='relu')(conv5_2)

    pool5 = MaxPooling2D(name='pool5',
                         pool_size=(3, 3),
                         strides=(1, 1),
                         padding='same')(conv5_3)

    # FC6 ------------------------------------------------
    pool5z = ZeroPadding2D(padding=(6, 6), name='pool5z')(pool5)

    fc6 = Conv2D(1024, (3, 3),
                 name='fc6',
                 dilation_rate=(6, 6),
                 padding='valid',
                 activation='relu'
                 )(pool5z)

    # FC7 ------------------------------------------------
    fc7 = Conv2D(1024, (1, 1),
                 name='fc7',
                 padding='same',
                 activation='relu'
                 )(fc6)

    return Model(input_layer, fc7)
