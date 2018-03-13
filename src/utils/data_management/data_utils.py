import glob
import os
import cv2
import numpy as np
from tqdm import tqdm


def get_class_names(dataset_name='VOC2007'):

    if (set(dataset_name).issubset(['VOC2007', 'VOC2012'])
       or dataset_name == 'VOC2007' or dataset_name == 'VOC2012'):
        class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
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


def list_files_in_directory(path_name='*'):
    return glob.glob(path_name)


def merge_two_dictionaries(dict_1, dict_2):
    merged_dict = dict_1.copy()
    merged_dict.update(dict_2)
    return merged_dict


def crop_boxes(data, arg_to_class, dump_path='cropped_images/',
               min_box_area=2500):
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    for image_path, image_data in tqdm(data.items()):
        image_name = os.path.basename(image_path)
        image_array = cv2.imread(image_path)
        h, w = image_array.shape[:2]
        for image_arg, image_box_data in enumerate(image_data):
            x_min, y_min, x_max, y_max = image_box_data[:4]
            image_arg_class = np.argmax(image_box_data[4:])
            image_class_name = arg_to_class[image_arg_class]
            x_min = int(x_min * w)
            y_min = int(y_min * h)
            x_max = int(x_max * w)
            y_max = int(y_max * h)
            box_h = y_max - y_min
            box_w = x_max - x_min
            box_area = box_h * box_w
            if box_area < min_box_area:
                continue
            cropped_image = image_array[y_min:y_max, x_min:x_max, :]
            image_dump_path = (dump_path + image_name[:-4] + '_' +
                               str(image_arg) + '_' +
                               image_class_name + '.jpg')
            cv2.imwrite(image_dump_path, cropped_image)
