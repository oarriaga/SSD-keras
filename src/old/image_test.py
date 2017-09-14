# import sys
import numpy as np
from utils.preprocessing import preprocess_images
from utils.boxes import create_prior_boxes
import cv2
from models.ssd import SSD300
import pickle

# my stuff
# image_path = '../images/cat.jpg'

class_threshold = .1
iou_threshold = .5
model_input = (300, 300)
weights_path = '../trained_models/SSD300_weights.hdf5'
model = SSD300(weights_path=weights_path)


rgb_image = pickle.load(open('rgb_image.pkl', 'rb'))
y1 = pickle.load(open('y1.pkl', 'rb'))
y2 = pickle.load(open('y2.pkl', 'rb'))
y3 = pickle.load(open('y3.pkl', 'rb'))
# image_array = load_image(image_path, model_input)[0]
image_array = cv2.resize(rgb_image, model_input)

image_array = preprocess_images(image_array)
image_array = np.expand_dims(image_array, axis=0)
predictions = model.predict(image_array)

keras_boxes = predictions[0, :, :4]
torch_boxes = y1[0]
keras_classes = np.argmax(predictions[0, :, 4:], axis=1)
torch_classes = np.argmax(y2[0], axis=1)
print('Number of different classifications')
print(np.sum(keras_classes != torch_classes))
print('Number of non-background classification for keras model')
print(np.sum(keras_classes != 0))
prior_boxes = create_prior_boxes()
print('Test for prior boxes')
print(np.all(np.isclose(prior_boxes, y3)))


"""
# his stuff
# image = cv2.imread(image_path)
image = rgb_image[:, :, ::-1]
# rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
x = cv2.resize(image, (300, 300)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
# x = torch.from_numpy(x).permute(2, 0, 1)

print(np.all(image_array[:, :, 0] == x[:, :, 0]))
print(np.all(image_array[:, :, 1] == x[:, :, 1]))
print(np.all(image_array[:, :, 2] == x[:, :, 2]))
"""
