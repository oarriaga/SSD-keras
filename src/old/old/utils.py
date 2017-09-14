#import matplotlib.pyplot as plt
#from keras.preprocessing import image as keras_image_preprocessor
#from keras.applications.vgg16 import preprocess_input
#import glob
#import numpy as np
#from utils.boxes import decode_boxes
#from utils.boxes import filter_boxes
#from PIL import Image as pil_image



# remove this function
#def list_files_in_directory(path_name='*'):
    #return glob.glob(path_name)


#def preprocess_images(image_array):
#    return preprocess_input(image_array)
#
#def predict_boxes(model, image_array, prior_boxes,
#                    class_threshold=.1,
#                    box_scale_factors=[.1, .1, .2, .2],
#                    num_classes=21, background_id=0):
#    image_array = np.expand_dims(image_array, axis=0)
#    image_array = preprocess_images(image_array)
#    predictions = model.predict(image_array)
#    predictions = np.squeeze(predictions)
#    predictions = decode_boxes(predictions, prior_boxes,
#                                    box_scale_factors)
#    predictions = filter_boxes(predictions, num_classes,
#                                background_id, class_threshold)
#
#def load_image(path, target_size=None):
#    image = load_pil_image(path)
#    image = resize_image(image, target_size)
#    return image_to_array(image)
#
#def array_to_image(image_array):
#    return pil_image.fromarray(image_array)
#
#def load_pil_image(path):
#    image = pil_image.open(path)
#    if image.mode != 'RGB':
#        image = image.convert('RGB')
#    return image
#
#def image_to_array(image, backend='tensorflow'):
#    image_array = np.asarray(image)
#    if backend == 'tensorflow':
#        image_array = image_array.transpose(2, 0, 1)
#    return pil_image
#
#def resize_image(image, target_size):
#    height_wdith_tuple = (target_size[1], target_size[0])
#    if image.size != height_wdith_tuple:
#        image = image.resize(height_wdith_tuple)
#    return image
#

#def scheduler(epoch, decay=0.9, base_learning_rate=3e-4):
#    return base_learning_rate * decay**(epoch)
#
#def split_data(ground_truths, training_ratio=.8):
#    ground_truth_keys = sorted(ground_truths.keys())
#    num_train = int(round(training_ratio * len(ground_truth_keys)))
#    train_keys = ground_truth_keys[:num_train]
#    validation_keys = ground_truth_keys[num_train:]
#    return train_keys, validation_keys
#
#def plot_images(original_image, transformed_image):
#    plt.figure(1)
#    plt.subplot(121)
#    plt.title('Original image')
#    plt.imshow(original_image.astype('uint8'))
#    plt.subplot(122)
#    plt.title('Transformed image')
#    plt.imshow(transformed_image.astype('uint8'))
#    plt.show()
#
# move this function to dataset module
#def get_class_names(dataset_name='VOC2007'):
#    if dataset_name == 'VOC2007':
#        class_names = ['background','aeroplane', 'bicycle', 'bird', 'boat',
#                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
#                       'diningtable', 'dog', 'horse', 'motorbike', 'person',
#                       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
#    elif dataset_name == 'COCO':
#        class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle',
#                        'airplane', 'bus', 'train', 'truck', 'boat',
#                        'traffic light', 'fire hydrant', 'stop sign',
#                        'parking meter', 'bench', 'bird', 'cat', 'dog',
#                        'horse', 'sheep', 'cow', 'elephant', 'bear',
#                        'zebra', 'giraffe', 'backpack', 'umbrella',
#                        'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#                        'snowboard', 'sports ball', 'kite', 'baseball bat',
#                        'baseball glove', 'skateboard', 'surfboard',
#                        'tennis racket', 'bottle', 'wine glass',
#                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
#                        'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#                        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#                        'potted plant', 'bed', 'dining table', 'toilet',
#                        'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#                        'cell phone', 'microwave', 'oven', 'toaster',
#                        'sink', 'refrigerator', 'book', 'clock', 'vase',
#                        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#    else:
#        raise Exception('Invalid dataset', dataset_name)
#    return class_names
#
#def get_arg_to_class(class_names):
#    return dict(zip(list(range(len(class_names))), class_names))
#
#
#
