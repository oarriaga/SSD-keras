import matplotlib.pyplot as plt
from keras.preprocessing import image as keras_image_preprocessor
from keras.applications.vgg16 import preprocess_input
import glob
import numpy as np
from utils.boxes import decode_boxes
from utils.boxes import filter_boxes


def preprocess_images(image_array):
    return preprocess_input(image_array)

# remove this function
def list_files_in_directory(path_name='*'):
    return glob.glob(path_name)

def predict_boxes(model, image_array, prior_boxes,
                    class_threshold=.1,
                    box_scale_factors=[.1, .1, .2, .2],
                    num_classes=21, background_id=0):
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_images(image_array)
    predictions = model.predict(image_array)
    predictions = np.squeeze(predictions)
    predictions = decode_boxes(predictions, prior_boxes,
                                    box_scale_factors)
    predictions = filter_boxes(predictions, num_classes,
                                background_id, class_threshold)

def load_image(image_path, grayscale=False ,target_size=None):
    image = keras_image_preprocessor.load_img(image_path,
                                                grayscale ,
                                    target_size=target_size)
    return keras_image_preprocessor.img_to_array(image)

def scheduler(epoch, decay=0.9, base_learning_rate=3e-4):
    return base_learning_rate * decay**(epoch)

def split_data(ground_truths, training_ratio=.8):
    ground_truth_keys = sorted(ground_truths.keys())
    num_train = int(round(training_ratio * len(ground_truth_keys)))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys

def plot_images(original_image, transformed_image):
    plt.figure(1)
    plt.subplot(121)
    plt.title('Original image')
    plt.imshow(original_image.astype('uint8'))
    plt.subplot(122)
    plt.title('Transformed image')
    plt.imshow(transformed_image.astype('uint8'))
    plt.show()

# move this function to dataset module
def get_class_names(dataset_name='VOC2007'):
    if dataset_name == 'VOC2007':
        class_names = ['background','aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person',
                       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == 'COCO':
        class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle',
                        'airplane', 'bus', 'train', 'truck', 'boat',
                        'traffic light', 'fire hydrant', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog',
                        'horse', 'sheep', 'cow', 'elephant', 'bear',
                        'zebra', 'giraffe', 'backpack', 'umbrella',
                        'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat',
                        'baseball glove', 'skateboard', 'surfboard',
                        'tennis racket', 'bottle', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                        'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet',
                        'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                        'cell phone', 'microwave', 'oven', 'toaster',
                        'sink', 'refrigerator', 'book', 'clock', 'vase',
                        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    else:
        raise Exception('Invalid dataset', dataset_name)
    return class_names

def get_arg_to_class(class_names):
    return dict(zip(list(range(len(class_names))), class_names))
