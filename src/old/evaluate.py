import numpy as np
from tqdm import tqdm

from datasets import DataManager
from datasets import get_class_names
from preprocessing import get_image_size
from models import SSD300
from metrics import compute_average_precision
from metrics import compute_precision_and_recall

from utils.boxes import create_prior_boxes
from utils.boxes import calculate_intersection_over_union
from utils.boxes import denormalize_boxes
from utils.inference import infer_from_path


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

image_prefix = '../datasets/VOCdevkit/VOC2007/JPEGImages/'
with_difficult_objects = False
split = 'test'

class_names = get_class_names(dataset_name)
class_names = class_names[1:]
average_precisions = []
for class_name in class_names:
    selected_classes = ['background'] + [class_name]
    data_manager = DataManager(dataset_name, split, selected_classes,
                               with_difficult_objects)
    ground_truth_data = data_manager.load_data()
    difficult_data_flags = data_manager.parser.difficult_objects
    scores = []
    labels = []
    num_gt_boxes = 0
    for image_name, gt_sample in tqdm(ground_truth_data.items()):
        image_path = image_prefix + image_name
        reference_size = get_image_size(image_path)
        detections = infer_from_path(image_path, model, prior_boxes)
        gt_sample = denormalize_boxes(gt_sample, reference_size)
        num_gt_boxes = num_gt_boxes + len(gt_sample)
        already_detected = np.zeros(shape=len(gt_sample), dtype=bool)
        for detection in detections:
            ious = calculate_intersection_over_union(detection, gt_sample)
            score = np.max(detection[4:])
            best_iou = np.max(ious)
            best_iou_arg = np.argmax(ious)
            if best_iou > iou_threshold:
                if not already_detected[best_iou_arg]:
                    labels.append(True)
                    scores.append(best_iou)
                    already_detected[best_iou_arg] = True
                else:
                    labels.append(False)
                    scores.append(best_iou)
            else:
                labels.append(False)
                scores.append(best_iou)
    results = compute_precision_and_recall(scores, labels, num_gt_boxes)
    precision, recall = results
    average_precision = compute_average_precision(precision, recall)
    average_precisions.append(average_precision)
    print('Class:', class_name)
    print('Number of ground_truth_boxes:', num_gt_boxes)
    print('AP:', average_precision)

mean_average_precision = np.mean(average_precisions)
print('mAP:', mean_average_precision)
