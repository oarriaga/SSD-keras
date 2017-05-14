import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
import glob

from keras.applications.vgg16 import preprocess_input

def split_data(ground_truths, training_ratio=.8):
    ground_truth_keys = sorted(ground_truths.keys())
    num_train = int(round(training_ratio * len(ground_truth_keys)))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys

def read_image(image_full_path):
    return imread(image_full_path)

def resize_image(image_array, shape):
    return imresize(image_array, shape)

def list_files_in_directory(path_name='*'):
    return glob.glob(path_name)

def plot_images(original_image, transformed_image):
    plt.figure(1)
    plt.subplot(121)
    plt.title('Original image')
    plt.imshow(original_image)
    plt.subplot(122)
    plt.title('Transformed image')
    plt.imshow(transformed_image)
    plt.show()

def flatten_prior_boxes(prior_boxes):
    prior_boxes = [layer_boxes.reshape(-1, 4)
                   for layer_boxes in prior_boxes]
    prior_boxes = np.concatenate(prior_boxes,axis=0)
    return prior_boxes

def get_class_names(dataset='VOC2007'):
    if dataset == 'VOC2007':
        class_names = ['background','aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                       'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                       'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor']
    else:
        raise Exception('Invalid dataset', dataset)
    return class_names

def scheduler(epoch, decay=0.9, base_learning_rate=3e-4):
    return base_learning_rate * decay**(epoch)

def preprocess_image(image_array):
    return preprocess_input(image_array)

def add_variances(prior_boxes, variances=[.1, .1, .2, .2]):
    num_prior_boxes = prior_boxes.shape[0]
    variances = np.asarray(variances) * np.ones((num_prior_boxes, 4))
    return np.concatenate([prior_boxes, variances], axis=1)

def add_classes(prior_boxes, variances=[.1, .1, .2, .2]):
    num_prior_boxes = prior_boxes.shape[0]
    variances = np.asarray(variances) * np.ones((num_prior_boxes, 4))
    return np.concatenate([prior_boxes, variances], axis=1)
