import numpy as np

from utils.datasets import DataManager
from utils.datasets import get_class_names
from utils.inference import predict
from utils.preprocessing import load_image
from utils.boxes import create_prior_boxes
from utils.preprocessing import image_to_array
from utils.preprocessing import load_pil_image
from utils.preprocessing import get_image_size
from utils.boxes import calculate_intersection_over_union
from utils.boxes import denormalize_box
from models.ssd import SSD300
#from utils.visualizer import draw_image_boxes
from utils.datasets import get_arg_to_class

def compute_precision_and_recall(scores, labels, num_gt):
    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    labels = labels.astype(int)

    true_positive_labels = labels[sorted_indices]
    false_positive_labels = 1 - true_positive_labels
    cum_true_positives = np.cumsum(true_positive_labels)
    cum_false_positives = np.cumsum(false_positive_labels)
    precision = cum_true_positives.astype(float) / (
                cum_true_positives + cum_false_positives)
    recall = cum_true_positives.astype(float) / num_gt
    return precision, recall


def compute_average_precision(precision, recall):
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # Preprocess precision to be a non-decreasing array
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
            (recall[indices] - recall[indices - 1]) * precision[indices])
    return average_precision

dataset_name = 'VOC2007'
data_prefix = '../datasets/VOCtest/VOCdevkit/VOC2007/Annotations/'
image_prefix = '../datasets/VOCtest/VOCdevkit/VOC2007/JPEGImages/'
weights_path = '../trained_models/weights_SSD300.hdf5'
model = SSD300(weights_path=weights_path)
prior_boxes = create_prior_boxes(model)
input_shape = model.input_shape[1:3]
class_threshold = .1
iou_threshold = .5


average_precisions = []
for ground_truth_class_arg in range(1, 21):
    labels = []
    scores = []
    class_names = get_class_names(dataset_name)
    #ground_truth_class_arg = class_arg
    selected_classes = [class_names[0]] + [class_names[ground_truth_class_arg]]
    num_ground_truth_boxes = 0
    class_decoder = get_arg_to_class(class_names)
    num_classes = len(class_names)
    data_manager = DataManager(dataset_name, selected_classes,
                                data_prefix, image_prefix)
    ground_truth_data = data_manager.load_data()
    difficult_data_flags = data_manager.parser.difficult_objects

    image_names = sorted(list(ground_truth_data.keys()))
    print('Number of images found:', len(image_names))
    for image_name in image_names:
        ground_truth_sample = ground_truth_data[image_name]
        image_prefix = data_manager.image_prefix
        image_path = image_prefix + image_name
        image_array = load_image(image_path, input_shape)
        original_image_array = image_to_array(load_pil_image(image_path))
        original_image_size = get_image_size(image_path)
        predicted_data = predict(model, image_array, prior_boxes, original_image_size,
                                        21, class_threshold, iou_threshold)
        ground_truth_sample = denormalize_box(ground_truth_sample, original_image_size)
        ground_truth_boxes_in_image = len(ground_truth_sample)
        difficult_objects = difficult_data_flags[image_name]
        difficult_objects = np.asarray(difficult_objects, dtype=bool)
        num_ground_truth_boxes += np.sum(np.logical_not(difficult_objects))
        if predicted_data is None:
            print('Zero predictions given for image:', image_name)
            continue
        #plt.imshow(original_image_array.astype('uint8'))
        #plt.show()
        #draw_image_boxes(predicted_data, original_image_array, class_decoder, normalized=False)
        num_predictions = len(predicted_data)
        for prediction_arg in range(num_predictions):
            predicted_box = predicted_data[prediction_arg]
            predicted_class_probabilities = predicted_box[4:]
            predicted_class_arg = np.argmax(predicted_class_probabilities)
            predicted_score = np.max(predicted_class_probabilities)
            ious = calculate_intersection_over_union(predicted_box, ground_truth_sample)
            num_objects = len(ground_truth_sample)
            for object_arg in range(num_objects):
                #ground_truth_classes = ground_truth_sample[object_arg][4:]
                #ground_truth_class_arg = np.argmax(ground_truth_classes)
                difficult = difficult_objects[object_arg]
                if difficult:
                    #num_ground_truth_boxes = num_ground_truth_boxes - 1
                    continue
                iou = ious[object_arg]
                if iou >= iou_threshold and predicted_class_arg == ground_truth_class_arg:
                    #print('True positive:', box_arg)
                    scores.append(predicted_score)
                    labels.append(True)

                if iou >= iou_threshold and predicted_class_arg != ground_truth_class_arg:
                    #print('False positive:', box_arg)
                    scores.append(predicted_score)
                    labels.append(False)

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


