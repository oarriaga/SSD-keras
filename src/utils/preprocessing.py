import numpy as np
import cv2

R_MEAN = 123
G_MEAN = 117
B_MEAN = 104


def substract_mean(image_array):
    image_array = image_array.astype(np.float32)
    image_array[:, :, 0] -= R_MEAN
    image_array[:, :, 1] -= G_MEAN
    image_array[:, :, 2] -= B_MEAN
    image_array = image_array[:, :, ::-1]
    return image_array


def load_image(image_path, target_size=None,
               return_original_shape=False, RGB=True):
    image_array = cv2.imread(image_path)
    if RGB:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    height, width = image_array.shape[:2]
    if target_size is not None:
        image_array = cv2.resize(image_array, target_size)
    if return_original_shape:
        return image_array, (height, width)
    else:
        return image_array


def get_image_size(image_path):
    image_array = cv2.imread(image_path)
    height, width = image_array.shape[:2]
    return (height, width)
