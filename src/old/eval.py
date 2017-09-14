import pickle
import numpy as np
from utils.datasets import get_class_names
from utils.boxes import calculate_intersection_over_union
from metrics import compute_average_precision
from metrics import compute_precision_and_recall


dataset_name = 'VOC2007'
class_names = get_class_names(dataset_name)
data_records = pickle.load(open('data_records.pkl', 'rb'))

image_ids = data_records['image_id']
detected_box_samples = data_records['detection_boxes']
detected_score_samples = data_records['detection_scores']
ground_truth_box_samples = data_records['groundtruth_boxes']
ground_truth_class_samples = data_records['groundtruth_classes']
difficulty_samples = data_records['difficult']
iou_treshold = .5
average_precisions = []
for class_arg, class_name in enumerate(class_names):
    num_images_in_class = 0
    labels = []
    scores = []
    for image_arg, image_id in enumerate(data_records['image_id']):
        ground_truth_classes = ground_truth_class_samples[image_arg]
        ground_truth_boxes = ground_truth_box_samples[image_arg]
        class_mask = ground_truth_classes == class_arg
        difficulties = difficulty_samples[image_arg]
        difficult_mask = np.logical_not(difficulties)
        mask = np.logical_or(class_mask, difficult_mask)
        if np.all(np.logical_not(mask)):
            continue
        # print(num_images_in_class)
        num_images_in_class = num_images_in_class + np.sum(mask)
        ground_truth_classes = ground_truth_classes[mask]
        ground_truth_boxes = ground_truth_boxes[mask]
        num_ground_truth_boxes = len(ground_truth_boxes)

        detected_boxes = detected_box_samples[image_arg]
        detected_scores = detected_score_samples[image_arg]
        num_detected_boxes = len(detected_scores)
        selected_ground_truth_boxes = np.zeros(num_ground_truth_boxes,
                                               dtype=bool)
        for detected_box_arg in range(num_detected_boxes):
            detected_box = detected_boxes[detected_box_arg]
            detected_score = detected_scores[detected_box_arg]
            scores.append(detected_score)
            ious = calculate_intersection_over_union(detected_box,
                                                     ground_truth_boxes)
            best_detected_box = np.argmax(ious)
            best_iou = np.max(ious)
            already_selected = selected_ground_truth_boxes[best_detected_box]
            if best_iou > iou_treshold:
                if not already_selected:
                    selected_ground_truth_boxes[best_detected_box] = True
                    labels.append(True)
                else:
                    labels.append(False)
            else:
                labels.append(False)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    precision, recall = compute_precision_and_recall(scores, labels,
                                                     num_images_in_class)
    average_precision = compute_average_precision(precision, recall)
    average_precisions.append(average_precision)
    print('Class:', class_name)
    print('Number of ground_truth_boxes:', num_images_in_class)
    print('AP:', average_precision)

average_precisions = np.asarray(average_precisions)
mean_average_precision = np.mean(average_precisions)
print('mAP:', mean_average_precision)
