import numpy as np
# import matplotlib.pyplot as plt

# from utils.datasets import DataManager
# from utils.datasets import get_class_names
from datasets import DataManager
from datasets import get_class_names
from preprocessing import substract_mean
from preprocessing import load_image

from utils.inference import predict
from utils.boxes import create_prior_boxes
from utils.boxes import calculate_intersection_over_union
from utils.boxes import denormalize_box
from models.ssd import SSD300
from metrics import compute_average_precision
# from metrics import compute_average_precision_2
from metrics import compute_precision_and_recall
# from utils.visualizer import draw_image_boxes
from utils.datasets import get_arg_to_class
import cv2


dataset_name = 'VOC2007'
image_prefix = '../datasets/VOCdevkit/VOC2007/JPEGImages/'

weights_path = '../trained_models/SSD300_weights.hdf5'
model = SSD300(weights_path=weights_path)
prior_boxes = create_prior_boxes()
input_shape = model.input_shape[1:3]
class_threshold = .1
iou_nms_threshold = .45
iou_threshold = .5
num_classes = 21


average_precisions = []
for ground_truth_class_arg in range(1, num_classes):
    labels = []
    scores = []
    class_names = get_class_names(dataset_name)
    selected_classes = [class_names[0]] + [class_names[ground_truth_class_arg]]
    num_ground_truth_boxes = 0
    class_decoder = get_arg_to_class(class_names)
    num_classes = len(class_names)
    data_manager = DataManager(dataset_name, split='test',
                               class_names=selected_classes)
    ground_truth_data = data_manager.load_data()
    difficult_data_flags = data_manager.parser.difficult_objects

    image_names = sorted(list(ground_truth_data.keys()))
    for image_name in image_names:
        ground_truth_sample = ground_truth_data[image_name]
        image_prefix = data_manager.parser.images_path
        image_path = image_prefix + image_name
        original_image_array = cv2.imread(image_path)
        image_array, original_image_size = load_image(image_path, input_shape)
        image_array = substract_mean(image_array)
        predicted_data = predict(model, image_array, prior_boxes,
                                 original_image_size, num_classes,
                                 class_threshold, iou_nms_threshold)

        ground_truth_sample = denormalize_box(ground_truth_sample,
                                              original_image_size)
        difficult_objects = difficult_data_flags[image_name]
        difficult_objects = np.asarray(difficult_objects, dtype=bool)
        num_ground_truth_boxes += np.sum(np.logical_not(difficult_objects))
        if predicted_data is None:
            continue

        """
        draw_image_boxes(predicted_data, original_image_array,
                         class_decoder, normalized=False)

        draw_image_boxes(ground_truth_sample, original_image_array,
                         class_decoder, normalized=False)
        """

        for ground_truth_object_arg in range(len(ground_truth_sample)):
            ground_truth_object = ground_truth_sample[ground_truth_object_arg]
            ious = calculate_intersection_over_union(ground_truth_object,
                                                     predicted_data)
            detections = np.zeros_like(ious)
            max_iou = np.max(ious)
            max_iou_arg = np.argmax(ious)
            difficult = difficult_objects[ground_truth_object_arg]

            predicted_sample = predicted_data[max_iou_arg]
            predicted_probabilities = predicted_sample[4:]
            predicted_sample_arg = np.argmax(predicted_probabilities)
            predicted_score = np.max(predicted_probabilities)

            if max_iou > iou_threshold:
                if difficult == False:
                    if predicted_sample_arg == ground_truth_class_arg:
                        scores.append(predicted_score)
                        labels.append(True)

                    else:
                        scores.append(predicted_score)
                        labels.append(False)
            else:
                scores.append(predicted_score)
                labels.append(False)

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    precision, recall = compute_precision_and_recall(scores, labels,
                                                     num_ground_truth_boxes)
    average_precision = compute_average_precision(precision, recall)
    average_precisions.append(average_precision)
    print('Class:', selected_classes[-1])
    print('Number of ground_truth_boxes:', num_ground_truth_boxes)
    print('AP:', average_precision)

average_precisions = np.asarray(average_precisions)
mean_average_precision = np.mean(average_precisions)
print('mAP:', mean_average_precision)
