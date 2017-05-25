import matplotlib.pyplot as plt
import glob
from keras.preprocessing import image as keras_image_preprocessor
from keras.applications.vgg16 import preprocess_input
import pickle


def split_data(ground_truths, training_ratio=.8):
    ground_truth_keys = sorted(ground_truths.keys())
    num_train = int(round(training_ratio * len(ground_truth_keys)))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys

def load_image(image_path, grayscale=False ,target_size=None):
    image = keras_image_preprocessor.load_img(image_path,
                                                grayscale ,
                                    target_size=target_size)
    return keras_image_preprocessor.img_to_array(image)

def list_files_in_directory(path_name='*'):
    return glob.glob(path_name)

def plot_images(original_image, transformed_image):
    plt.figure(1)
    plt.subplot(121)
    plt.title('Original image')
    plt.imshow(original_image.astype('uint8'))
    plt.subplot(122)
    plt.title('Transformed image')
    plt.imshow(transformed_image.astype('uint8'))
    plt.show()

def get_class_names(dataset_name='VOC2007'):
    if dataset_name == 'VOC2007':
        class_names = ['background','aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person',
                       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == 'COCO':
        # this is not working since the path has to be relative 
        class_names = pickle.load(open('utils/coco_classes.p', 'rb'))
    else:
        raise Exception('Invalid dataset', dataset_name)
    return class_names

def scheduler(epoch, decay=0.9, base_learning_rate=3e-4):
    return base_learning_rate * decay**(epoch)

def preprocess_images(image_array):
    return preprocess_input(image_array)

def get_arg_to_class(class_names):
    return dict(zip(list(range(len(class_names))), class_names))
