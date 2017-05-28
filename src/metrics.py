import matplotlib.pyplot as plt
import numpy as np
from utils.prior_box_creator import PriorBoxCreator
from utils.utils import preprocess_images
from utils.utils import get_class_names
from utils.utils import load_image

from data_loader import DataLoader

class Metrics(object):
    """Class for evaluating VOC2007 metrics"""
    def __init__(self, model, test_keys, prior_boxes=None, dataset_name='VOC2007'):
        self.model = model
        box_creator = PriorBoxCreator(self.model)
        self.prior_boxes = box_creator.create_boxes()
        self.class_names = get_class_names(dataset_name)
        self.num_classes = len(self.class_names)
        self.colors = plt.cm.hsv(np.linspace(0, 1, self.num_classes)).tolist()
        self.colors = np.asarray(self.colors) * 255
        self.arg_to_class = dict(zip(list(range(self.num_classes)),
                                                self.class_names))
        self.overlap_threshold = .5
        self.background_id = 0
        self.test_keys = test_keys
        self.root_prefix = '../datasets/VOCdevkit/VOC2007/'
        self.ground_data_prefix = self.root_prefix + 'Annotations/'
        self.image_prefix = self.root_prefix + 'JPEGImages/'
        self.image_size = (300, 300)
        self.ground_truth_manager = DataLoader(dataset_name)
        self.ground_truth_data = self.ground_truth_manager.get_data()


    def _decode_boxes(self, predicted_boxes):
        prior_x_min = self.prior_boxes[:, 0]
        prior_y_min = self.prior_boxes[:, 1]
        prior_x_max = self.prior_boxes[:, 2]
        prior_y_max = self.prior_boxes[:, 3]

        prior_width = prior_x_max - prior_x_min
        prior_height = prior_y_max - prior_y_min
        prior_center_x = 0.5 * (prior_x_max + prior_x_min)
        prior_center_y = 0.5 * (prior_y_max + prior_y_min)

        # TODO rename to g_hat_center_x all the other variables 
        pred_center_x = predicted_boxes[:, 0]
        pred_center_y = predicted_boxes[:, 1]
        pred_width = predicted_boxes[:, 2]
        pred_height = predicted_boxes[:, 3]

        scale_center_x = self.box_scale_factors[0]
        scale_center_y = self.box_scale_factors[1]
        scale_width = self.box_scale_factors[2]
        scale_height = self.box_scale_factors[3]

        decoded_center_x = pred_center_x * prior_width * scale_center_x
        decoded_center_x = decoded_center_x + prior_center_x
        decoded_center_y = pred_center_y * prior_height * scale_center_y
        decoded_center_y = decoded_center_y + prior_center_y

        decoded_width = np.exp(pred_width * scale_width)
        decoded_width = decoded_width * prior_width
        decoded_height = np.exp(pred_height * scale_height)
        decoded_height = decoded_height * prior_height

        decoded_x_min = decoded_center_x - (0.5 * decoded_width)
        decoded_y_min = decoded_center_y - (0.5 * decoded_height)
        decoded_x_max = decoded_center_x + (0.5 * decoded_width)
        decoded_y_max = decoded_center_y + (0.5 * decoded_height)

        decoded_boxes = np.concatenate((decoded_x_min[:, None],
                                      decoded_y_min[:, None],
                                      decoded_x_max[:, None],
                                      decoded_y_max[:, None]), axis=-1)
        decoded_boxes = np.clip(decoded_boxes, 0.0, 1.0)
        if predicted_boxes.shape[1] > 4:
            decoded_boxes = np.concatenate([decoded_boxes,
                            predicted_boxes[:, 4:]], axis=-1)
        return decoded_boxes

    def _apply_non_max_suppression(self, boxes, iou_threshold=.2):
        if len(boxes) == 0:
                return []
        selected_indices = []
        x_min = boxes[:, 0]
        y_min = boxes[:, 1]
        x_max = boxes[:, 2]
        y_max = boxes[:, 3]
        classes = boxes[:, 4:]
        sorted_box_indices = np.argsort(y_max)
        while len(sorted_box_indices) > 0:
                last = len(sorted_box_indices) - 1
                i = sorted_box_indices[last]
                selected_indices.append(i)
                box = [x_min[i], y_min[i], x_max[i], y_max[i]]
                box = np.asarray(box)
                test_boxes = [x_min[sorted_box_indices[:last], None],
                         y_min[sorted_box_indices[:last], None],
                         x_max[sorted_box_indices[:last], None],
                         y_max[sorted_box_indices[:last], None]]
                test_boxes = np.concatenate(test_boxes, axis=-1)
                iou = self._calculate_intersection_over_unions(box, test_boxes)
                current_class = np.argmax(classes[i])
                box_classes = np.argmax(classes[sorted_box_indices[:last]], axis=-1)
                class_mask = current_class == box_classes
                overlap_mask = iou > iou_threshold
                delete_mask = np.logical_and(overlap_mask, class_mask)
                sorted_box_indices = np.delete(sorted_box_indices, np.concatenate(([last],
                        np.where(delete_mask)[0])))
        return boxes[selected_indices]

    def _calculate_intersection_over_unions(self, data_sample, prior_boxes):
        ground_truth_x_min = data_sample[0]
        ground_truth_y_min = data_sample[1]
        ground_truth_x_max = data_sample[2]
        ground_truth_y_max = data_sample[3]
        prior_boxes_x_min = prior_boxes[:, 0]
        prior_boxes_y_min = prior_boxes[:, 1]
        prior_boxes_x_max = prior_boxes[:, 2]
        prior_boxes_y_max = prior_boxes[:, 3]
        # calculating the intersection
        intersections_x_min = np.maximum(prior_boxes_x_min, ground_truth_x_min)
        intersections_y_min = np.maximum(prior_boxes_y_min, ground_truth_y_min)
        intersections_x_max = np.minimum(prior_boxes_x_max, ground_truth_x_max)
        intersections_y_max = np.minimum(prior_boxes_y_max, ground_truth_y_max)
        intersected_widths = intersections_x_max - intersections_x_min
        intersected_heights = intersections_y_max - intersections_y_min
        intersected_widths = np.maximum(intersected_widths, 0)
        intersected_heights = np.maximum(intersected_heights, 0)
        intersections = intersected_widths * intersected_heights
        # calculating the union
        prior_box_widths = prior_boxes_x_max - prior_boxes_x_min
        prior_box_heights = prior_boxes_y_max - prior_boxes_y_min
        prior_box_areas = prior_box_widths * prior_box_heights
        ground_truth_width = ground_truth_x_max - ground_truth_x_min
        ground_truth_height = ground_truth_y_max - ground_truth_y_min
        ground_truth_area = ground_truth_width * ground_truth_height
        unions = prior_box_areas + ground_truth_area - intersections
        intersection_over_unions = intersections / unions
        return intersection_over_unions

    def _filter_boxes(self, predictions):
        predictions = np.squeeze(predictions)
        predictions = self._decode_boxes(predictions)
        box_classes = predictions[:, 4:(4 + self.num_classes)]
        best_classes = np.argmax(box_classes, axis=-1)
        best_probabilities = np.max(box_classes, axis=-1)
        background_mask = best_classes != self.background_index
        lower_bound_mask = self.lower_probability_bound < best_probabilities
        mask = np.logical_and(background_mask, lower_bound_mask)
        selected_boxes = predictions[mask, :(4 + self.num_classes)]
        return selected_boxes

    def _decode_predictions(self, predictions, original_image_size):
        decoded_predictions = self._decode_boxes(predictions)
        selected_boxes = self._filter_boxes(decoded_predictions)
        if len(selected_boxes) == 0:
            return
        original_image_size = (original_image_size[1], original_image_size[0])
        box_classes = selected_boxes[:, 4:]
        box_coordinates = selected_boxes[:, 0:4]
        original_coordinates = self._denormalize_box(box_coordinates,
                                                    original_image_size)
        selected_boxes = np.concatenate([original_coordinates, box_classes],
                                                                    axis=-1)
        selected_boxes = self.apply_non_max_suppression_fast(selected_boxes)
        if len(selected_boxes) == 0:
            return
        return selected_boxes

    def _assign_boxes_to_object(self, ground_truth_box, return_iou=True):
        ious = self._calculate_intersection_over_unions(ground_truth_box)
        encoded_boxes = np.zeros((self.num_priors, 4 + return_iou))
        assign_mask = ious > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[ious.argmax()] = True
        if return_iou:
            encoded_boxes[:, -1][assign_mask] = ious[assign_mask]
        assigned_prior_boxes = self.prior_boxes[assign_mask]
        self.assigned_prior_boxes.append(assigned_prior_boxes)
        assigned_encoded_priors = self._encode_box(assigned_prior_boxes,
                                                   ground_truth_box)
        encoded_boxes[assign_mask, 0:4] = assigned_encoded_priors
        return encoded_boxes.ravel()

    def assign_boxes(self, ground_truth_data):
        self.assigned_prior_boxes = []
        assignments = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignments[:, 4 + self.background_id] = 1.0
        num_objects_in_image = len(ground_truth_data)
        if num_objects_in_image == 0:
            return assignments
        encoded_boxes = np.apply_along_axis(self._assign_boxes_to_object,
                                            1, ground_truth_data[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_indices = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_indices = best_iou_indices[best_iou_mask]
        num_assigned_boxes = len(best_iou_indices)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignments[best_iou_mask, :4] = encoded_boxes[best_iou_indices,
                                                np.arange(num_assigned_boxes),
                                                :4]

        assignments[:, 4][best_iou_mask] = 0
        assignments[:, 5:-8][best_iou_mask] = ground_truth_data[best_iou_indices, 5:]
        assignments[:, -8][best_iou_mask] = 1
        return assignments

    def calculate_MAP(self, model):
        for test_key in self.test_keys:
            image_path = self.path_prefix + test_key
            image_array = load_image(image_path, False, self.image_size)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_images(image_array)
            predicted_data = self.model.predict(image_array)
            predicted_data = np.squeeze(predicted_data)
            original_image_size = None
            predicted_boxes = self._decode_predictions(predicted_data, original_image_size)
            ground_truth_box_data = self.ground_truth_data[test_key].copy()


