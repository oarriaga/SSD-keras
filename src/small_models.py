from layers import Normalize, PriorBox
from keras.layers import Activation
#from keras.layers import AtrousConvolution2D
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
#from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
#from keras.layers import merge
#from keras.layers import Reshape
#from keras.layers import ZeroPadding2D
from keras.models import Model


def simple_SSD(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    x = Convolution2D(16, 7, 7)(input_tensor)
    x = Activation('relu')(x)
    x = MaxPooling2D(2, 2, border_mode='valid')(x)
    x = Dropout(.5)(x)
    x = Normalize(x)
    x = Convolution2D(32, 5, 5)(input_tensor)
    x = Activation('relu')(x)
    x = MaxPooling2D(2, 2, border_mode='valid')(x)
    x = Dropout(.5)(x)
    x = Normalize(x)




