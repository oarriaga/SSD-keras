import numpy as np
from PIL import Image as pil_image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as keras_image_preprocessor

def preprocess_images(image_array):
    return preprocess_input(image_array)

def load_image(image_path, target_size=None, grayscale=False):
    image = keras_image_preprocessor.load_img(image_path,
                                                grayscale,
                                    target_size=target_size)
    return keras_image_preprocessor.img_to_array(image)

def preprocess_images2(image_array, backend='tensorflow'):
    if backend == 'tensorflow':
        # 'RGB'->'BGR'
        image_array = image_array[:, :, :, ::-1]
    # Zero-center by mean pixel
    image_array[:, 0, :, :] -= 103.939
    image_array[:, 1, :, :] -= 116.779
    image_array[:, 2, :, :] -= 123.68
    return image_array

def load_image2(path, target_size=None):
    image = load_pil_image(path)
    image = resize_image(image, target_size)
    return image_to_array(image)

def load_pil_image(path):
    image = pil_image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def resize_image(image, target_size):
    height_width_tuple = (target_size[1], target_size[0])
    if image.size != height_width_tuple:
        image = image.resize(height_width_tuple)
    return image

def resize_image_array(image_array, target_size):
    image = array_to_image(image_array)
    image = resize_image(image, target_size)
    return image_to_array(image)

def image_to_array(image, backend='tensorflow'):
    image_array = np.asarray(image, dtype='float32')
    return image_array

def array_to_image(image_array, backend='tensorflow'):
    image_array = image_array.astype('uint8')
    return pil_image.fromarray(image_array, 'RGB')

def get_image_size(path):
    image = pil_image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image.size[::-1]
