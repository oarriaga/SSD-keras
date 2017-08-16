from pytorch_tests.pytorch_networks import build_ssd
from torch.autograd import Variable
import torch
import tensorflow as tf

from models.ssd import SSD300
from utils.preprocessing import preprocess_images
from utils.inference import predict
from utils.boxes import create_prior_boxes
import numpy as np
from scipy.misc import imread, imresize
from utils.datasets import DataManager
from utils.datasets import get_class_names
from tqdm import tqdm


# functions
def preprocess_pytorch_input(image):
    x = image.astype(np.float32)
    x -= (123.68, 116.779, 103.939)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    """
    if torch.cuda.is_available():
        xx = xx.cuda()
    """
    return xx


def softmax(x, axis=1):
    x = x.astype(float)
    num_samples, num_classes = x.shape
    new_x = np.zeros_like(x)
    for sample_arg in range(num_samples):
        x_sample = x[sample_arg]
        new_x[sample_arg] = np.exp(x_sample) / np.sum(np.exp(x_sample), axis=0)
    return new_x


def load_image(image_path, target_size=None):
    image_array = imread(image_path)
    height, width = image_array.shape[:2]
    if target_size is not None:
        image_array = imresize(image_array, target_size)
    return image_array, (height, width)


# parameters
trained_weights_path = '../trained_models/ssd_300_VOC0712.pth'
input_size = 300
num_classes = 21
iou_threshold = .5
lower_probability_threshold = .01
background_index = 0
dataset_name = 'VOC2007'
data_prefix = '../datasets/VOCdevkit/VOC2007/Annotations/'
image_prefix = '../datasets/VOCdevkit/VOC2007/JPEGImages/'
selected_classes = get_class_names(dataset_name)
prior_boxes = create_prior_boxes()

# loading pytorch model
pytorch_ssd = build_ssd('test', input_size, num_classes)
pytorch_ssd.load_weights(trained_weights_path)

# loading keras model
"""
weights_path = '../trained_models/SSD300_weights.hdf5'
with tf.device('/cpu:0'):
    model = SSD300(weights_path=weights_path)
"""

image_path = '../images/boys.jpg'
rgb_image, image_size = load_image(image_path, target_size=(300, 300))
pytorch_image = preprocess_pytorch_input(rgb_image)
pytorch_output = pytorch_ssd(pytorch_image)
pytorch_output = pytorch_output.data.numpy()

"""
data_manager = DataManager(dataset_name, selected_classes,
                           data_prefix, image_prefix)

ground_truth_data = data_manager.get_data()

# image_names = sorted(list(ground_truth_data.keys()))
image_names = list(ground_truth_data.keys())
for image_name in tqdm(image_names):
    ground_truth_sample = ground_truth_data[image_name]
    image_path = image_prefix + image_name

    rgb_image, image_size = load_image(image_path, target_size=(300, 300))
    pytorch_image = preprocess_pytorch_input(rgb_image)
    pytorch_output = pytorch_ssd(pytorch_image)
    p1 = pytorch_output[0].data.numpy()  # bounding boxes
    p2 = softmax(np.squeeze(pytorch_output[1].data.numpy()))  # classes
    p3 = pytorch_output[2].data.numpy()  # prior boxes
    # pytorch_detections = pytorch_output.data

    keras_image = preprocess_images(rgb_image)
    keras_image_input = np.expand_dims(keras_image, axis=0)
    keras_output = model.predict(keras_image_input)
    keras_detection = predict(model, keras_image, prior_boxes,
                              image_size, num_classes,
                              lower_probability_threshold,
                              iou_threshold,
                              background_index)

    keras_output = np.squeeze(keras_output)
    k1 = keras_output[:, :4]
    k2 = keras_output[:, 4:]
    k3 = prior_boxes

    diff = np.abs(p1 - k1)
    diff_mask = .0001 > diff
    all_good = np.all(diff_mask)
    print(all_good)
    if not all_good:
        print('*'*30)
        print(image_name)
        print(all_good)
    best_p2 = np.argmax(p2, axis=1)
    best_k2 = np.argmax(k2, axis=1)
    same_classes = np.all(best_p2 == best_k2)
    print(same_classes)
    if same_classes == False:
        print('*'*30)
    if not same_classes:
        print(image_name)
        print(np.sum(np.logical_not(same_classes)))
"""
