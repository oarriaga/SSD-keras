from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

from utils.utils import read_image
from utils.utils import resize_image



model_filename = '../trained_models/model_checkpoints/weights.09-0.25.hdf5'
model = load_model(model_filename)
image_size = model.input_shape[1:3]

image_filename = '../images/008745.jpg'
image_array = read_image(image_filename)
image_array = resize_image(image_array, image_size)
image_array = np.expand_dims(image_array, 0)
image_array = preprocess_input(image_array)
predictions = model.predict([image_array])


