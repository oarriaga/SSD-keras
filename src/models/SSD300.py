from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Activation
from keras.models import Model

from .layers import Conv2DNormalization
from .ssd_utils import add_ssd_modules


def SSD300(input_shape=(300, 300, 3), num_classes=21,
           num_priors=[4, 6, 6, 6, 4, 4], weights_path=None,
           return_base=False):

    image = Input(shape=input_shape)

    # Block 1 -----------------------------------------------------------------
    conv1_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(image)
    conv1_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding='same', )(conv1_2)

    # Block 2 -----------------------------------------------------------------
    conv2_1 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
    conv2_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding='same')(conv2_2)

    # Block 3 -----------------------------------------------------------------
    conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
    conv3_2 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding='same')(conv3_3)

    # Block 4 -----------------------------------------------------------------
    conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
    conv4_2 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv4_2)
    conv4_3_norm = Conv2DNormalization(20, name='branch_1')(conv4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding='same')(conv4_3)

    # Block 5 -----------------------------------------------------------------
    conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool4)
    conv5_2 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                         padding='same')(conv5_3)

    # Dense 6/7 ------------------------------------------
    pool5z = ZeroPadding2D(padding=(6, 6))(pool5)
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), padding='valid',
                 activation='relu')(pool5z)
    fc7 = Conv2D(1024, (1, 1), padding='same', activation='relu',
                 name='branch_2')(fc6)

    # EXTRA layers in SSD -----------------------------------------------------
    # Block 6 -----------------------------------------------------------------
    conv6_1 = Conv2D(256, (1, 1), padding='same', activation='relu')(fc7)
    conv6_1z = ZeroPadding2D()(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid',
                     activation='relu', name='branch_3')(conv6_1z)

    # Block 7 -----------------------------------------------------------------
    conv7_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(conv6_2)
    conv7_1z = ZeroPadding2D()(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), padding='valid', strides=(2, 2),
                     activation='relu', name='branch_4')(conv7_1z)

    # Block 8 -----------------------------------------------------------------
    conv8_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), padding='valid', strides=(1, 1),
                     activation='relu', name='branch_5')(conv8_1)

    # Block 9 -----------------------------------------------------------------
    conv9_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), padding='valid', strides=(1, 1),
                     activation='relu', name='branch_6')(conv9_1)

    if return_base:
        output_tensor = fc7

    else:
        ssd_tenors = [conv4_3_norm, fc7, conv6_2, conv7_2, conv8_2, conv9_2]
        output_tensor = add_ssd_modules(
                ssd_tenors, num_classes, num_priors, with_batch_norm=False)

    model = Model(inputs=image, outputs=output_tensor)

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model
