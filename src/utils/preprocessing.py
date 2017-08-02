import numpy as np
import cv2


def preprocess_images(image_array):
    image_array = image_array.astype(np.float32)
    image_array[:, :, 0] = image_array[:, :, 0] - 123.0
    image_array[:, :, 1] = image_array[:, :, 1] - 117.0
    image_array[:, :, 2] = image_array[:, :, 2] - 104.0
    image_array = image_array[:, :, ::-1]
    return image_array


def load_image(image_path, target_size=None):
    image_array = cv2.imread(image_path)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    height, width = image_array.shape[:2]
    image_array = cv2.resize(image_array, target_size)
    return image_array, (height, width)
