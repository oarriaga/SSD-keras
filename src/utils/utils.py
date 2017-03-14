from scipy.misc import imread
from scipy.misc import imresize
import glob

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


