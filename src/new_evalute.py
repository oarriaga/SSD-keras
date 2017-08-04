import numpy as np
# import matplotlib.pyplot as plt

from utils.datasets import DataManager
from utils.datasets import get_class_names
from utils.inference import predict
from utils.preprocessing import load_image
from utils.preprocessing import preprocess_images
from utils.boxes import create_prior_boxes
from utils.boxes import calculate_intersection_over_union
from utils.boxes import denormalize_box
from models.ssd import SSD300
from metrics import compute_average_precision
from metrics import compute_precision_and_recall
# from utils.visualizer import draw_image_boxes
from utils.datasets import get_arg_to_class



dataset_name = 'VOC2007'
data_prefix = '../datasets/VOCtest/VOCdevkit/VOC2007/Annotations/'
image_prefix = '../datasets/VOCtest/VOCdevkit/VOC2007/JPEGImages/'
weights_path = '../trained_models/SSD300_weights.hdf5'
model = SSD300(weights_path=weights_path)
prior_boxes = create_prior_boxes()
input_shape = model.input_shape[1:3]
class_threshold = .05
iou_nms_threshold = .45
iou_threshold = .5
num_classes = 21


average_precisions = []
# range(1, 21)
for ground_truth_class_arg in range(1, num_classes):
    labels = []
    scores = []
    class_names = get_class_names(dataset_name)
    selected_classes = [class_names[0]] + [class_names[ground_truth_class_arg]]
    num_ground_truth_boxes = 0
    class_decoder = get_arg_to_class(class_names)
    num_classes = len(class_names)
    data_manager = DataManager(dataset_name, selected_classes,
                               data_prefix, image_prefix)
    ground_truth_data = data_manager.load_data()
    # maybe the flags are wrong. They are assigned to different objects
    # How are order of the flags retained?
    difficult_data_flags = data_manager.parser.difficult_objects

    image_names = sorted(list(ground_truth_data.keys()))
    true_positives = []
    false_positives = []
    for image_name in image_names:
        ground_truth_sample = ground_truth_data[image_name]
        image_prefix = data_manager.image_prefix
        image_path = image_prefix + image_name
        image_array, original_image_size = load_image(image_path, input_shape)
        image_array = preprocess_images(image_array)
        predicted_data = predict(model, image_array, prior_boxes,
                                 original_image_size, num_classes,
                                 class_threshold, iou_nms_threshold)
        ground_truth_sample = denormalize_box(ground_truth_sample,
                                              original_image_size)
        difficult_objects = difficult_data_flags[image_name]
        difficult_objects = np.asarray(difficult_objects, dtype=bool)
        # num_ground_truth_boxes += np.sum(np.logical_not(difficult_objects))
        if predicted_data is None:
            continue
        """
        draw_image_boxes(predicted_data, original_image_array,
                         class_decoder, normalized=False)
        """
        # 
        num_predictions = len(predicted_data)
        for prediction_arg in range(num_predictions):
            predicted_sample = predicted_data[prediction_arg]
            predicted_sample_probabilities = predicted_sample[4:]
            predicted_sample_arg = np.argmax(predicted_sample_probabilities)
            predicted_score = np.max(predicted_sample_probabilities)
            print('predicted_class_arg', predicted_sample_arg)
            """
            you take an image an image it predicts.
            it predicts a lot of things
            this things might be correct since there might be a person
            but you are now only interested in airplanes
            you have to check which of all predictions has the max iou
            with respect to a ground truth box.
            Therefore the for should run for ground truth boxes
            you then select the predicted box with the highest iou
            then you check if the classes are the same.
            If the classes are the same you have tp
            if not you have a fp
            if the none of the predicted boxes matches the ground truth
            you have also a fp.
            """
            ious = calculate_intersection_over_union(predicted_sample,
                                                     ground_truth_sample)
            max_iou = np.max(ious)
            max_iou_arg = np.argmax(ious)
            difficult = difficult_objects[max_iou_arg]
            if difficult is True:
                continue
            scores.append(predicted_score)
            num_ground_truth_boxes = num_ground_truth_boxes + 1
            if max_iou >= iou_threshold:
                if predicted_sample_arg == ground_truth_class_arg:
                    labels.append(True)
                else:
                    labels.append(False)
            else:
                labels.append(False)

    """
            num_objects = len(ground_truth_sample)
            for object_arg in range(num_objects):
                # ground_truth_classes = ground_truth_sample[object_arg][4:]
                # ground_truth_class_arg = np.argmax(ground_truth_classes)
                difficult = difficult_objects[object_arg]
                if difficult:
                    # num_ground_truth_boxes = num_ground_truth_boxes - 1
                    continue
                iou = ious[object_arg]
                # i think this is bad you have to take the max
                if iou >= iou_threshold:
                    if predicted_class_arg == ground_truth_class_arg:
                        scores.append(predicted_score)
                        labels.append(True)
                    else:
                        scores.append(predicted_score)
                        labels.append(False)
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    precision, recall = compute_precision_and_recall(scores, labels,
                                                     num_ground_truth_boxes)
    average_precision = compute_average_precision(precision, recall)
    average_precisions.append(average_precision)
    print('Class:', selected_classes[-1])
    print('AP:', average_precision)
    print('Number of ground_truth_boxes:', num_ground_truth_boxes)

average_precisions = np.asarray(average_precisions)
mean_average_precision = np.mean(average_precisions)
print('mAP:', mean_average_precision)
