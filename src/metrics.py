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

dataset_name = 'VOC2007'
data_prefix = '../datasets/VOCtest/VOCdevkit/VOC2007/Annotations/'
image_prefix = '../datasets/VOCtest/VOCdevkit/VOC2007/JPEGImages/'
weights_path = '../trained_models/weights_SSD300.hdf5'
class_names = get_class_names(dataset_name)
num_classes = len(class_names)
model = SSD300(weights_path=weights_path)
prior_boxes = create_prior_boxes(model)
input_shape = model.input_shape[1:3]
class_threshold = .1
iou_threshold = .5

#skip class name 0 for background
# use enumerate for the class argument
labels = []
scores = []
num_ground_truth_boxes = 0
class_name = ['background'] + [class_names[1]]
class_arg = 1

data_manager = DataManager(dataset_name, class_name,
                            data_prefix, image_prefix)
ground_truth_data = data_manager.load_data()

image_names = sorted(list(ground_truth_data.keys()))
print('Number of images found:', len(image_names))
for image_name in image_names:
    #image_name = image_names[0]

    ground_truth_sample = ground_truth_data[image_name]
    #print(ground_truth_sample.shape)

    image_prefix = data_manager.image_prefix
    image_path = image_prefix + image_name
    image_array = load_image(image_path, input_shape)
    original_image_array = image_to_array(load_pil_image(image_path))
    original_image_size = get_image_size(image_path)
    predicted_data = predict(model, image_array, prior_boxes, original_image_size,
                                    num_classes, class_threshold, iou_threshold)
    ground_truth_sample = denormalize_box(ground_truth_sample, original_image_size)
    #print('Predicted data', predicted_data)
    #print('Ground truth data', ground_truth_sample)
    num_ground_truth_boxes = num_ground_truth_boxes + len(ground_truth_sample)
    if predicted_data is None:
        continue
    num_predictions = len(predicted_data)
    for box_arg in range(num_predictions):
        predicted_box = predicted_data[box_arg]
        class_probabilities = predicted_box[4:]
        best_class_arg = np.argmax(class_probabilities)
        detection_score = np.max(class_probabilities)
        ious = calculate_intersection_over_union(predicted_box, ground_truth_sample)
        #print(ious)
        #print(ious.shape)
        #iou = ious
        for iou_arg in range(len(ious)):
            iou = ious[iou_arg]
            if iou >= iou_threshold and best_class_arg == class_arg:
                #print('True positive:', box_arg)
                scores.append(detection_score)
                labels.append(True)

            if iou >= iou_threshold and best_class_arg != class_arg:
                #print('False positive:', box_arg)
                scores.append(detection_score)
                labels.append(False)

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


def compute_cor_loc(num_gt_imgs_per_class,
                    num_images_correctly_detected_per_class):
  """Compute CorLoc according to the definition in the following paper.
  https://www.robots.ox.ac.uk/~vgg/rg/papers/deselaers-eccv10.pdf
  Returns nans if there are no ground truth images for a class.
  Args:
    num_gt_imgs_per_class: 1D array, representing number of images containing
        at least one object instance of a particular class
    num_images_correctly_detected_per_class: 1D array, representing number of
        images that are correctly detected at least one object instance of a
        particular class
  Returns:
    corloc_per_class: A float numpy array represents the corloc score of each
      class
  """
  return np.where(
      num_gt_imgs_per_class == 0,
      np.nan,
      num_images_correctly_detected_per_class / num_gt_imgs_per_class)


scores = np.asarray(scores)
labels = np.asarray(labels)
precision, recall = compute_precision_and_recall(scores, labels,
                                        num_ground_truth_boxes)
average_precision = compute_average_precision(precision, recall)
print(average_precision)




