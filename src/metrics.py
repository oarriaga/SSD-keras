import matplotlib.pyplot as plt
import numpy as np
from utils.prior_box_creator import PriorBoxCreator
from utils.utils import preprocess_images
from utils.utils import get_class_names
import cv2

class Metrics(object):
    """Class for evaluating VOC2007 metrics"""
    def __init__(self, model, prior_boxes=None, dataset_name='VOC2007'):
        self.model = model
        box_creator = PriorBoxCreator(self.model)
        self.prior_boxes = box_creator.create_boxes()
        self.class_names = get_class_names(dataset_name)
        self.num_classes = len(self.class_names)
        self.colors = plt.cm.hsv(np.linspace(0, 1, self.num_classes)).tolist()
        self.colors = np.asarray(self.colors) * 255
        self.arg_to_class = dict(zip(list(range(self.num_classes)),
                                                self.class_names))
        self.font = cv2.FONT_HERSHEY_SIMPLEX

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

    def _find_boxes(self, predictions, original_image_array):
        decoded_predictions = self._decode_boxes(predictions)
        selected_boxes = self._filter_boxes(decoded_predictions)
        if len(selected_boxes) == 0:
            return
        image_array = np.squeeze(original_image_array)
        image_array = image_array.astype('uint8')
        image_size = image_array.shape[0:2]
        image_size = (image_size[1], image_size[0])
        box_classes = selected_boxes[:, 4:]
        box_coordinates = selected_boxes[:, 0:4]
        original_coordinates = self._denormalize_box(box_coordinates,
                                                            image_size)
        selected_boxes = np.concatenate([original_coordinates, box_classes],
                                                                    axis=-1)
        selected_boxes = self.apply_non_max_suppression_fast(selected_boxes)
        if len(selected_boxes) == 0:
            return
        return selected_boxes

    def _compare_predictions(self, selected_boxes, ground_truth_boxes):
        figure, axis = plt.subplots(1)
        x_min = selected_boxes[:, 0]
        y_min = selected_boxes[:, 1]
        x_max = selected_boxes[:, 2]
        y_max = selected_boxes[:, 3]
        width = x_max - x_min
        height = y_max - y_min
        classes = selected_boxes[:, 4:]
        num_boxes = len(selected_boxes)
        for box_arg in range(num_boxes):
            x_min_box = int(x_min[box_arg])
            y_min_box = int(y_min[box_arg])
            box_width = int(width[box_arg])
            box_height = int(height[box_arg])
            box_class = classes[box_arg]
            label_arg = np.argmax(box_class)
            score = box_class[label_arg]
            class_name = self.arg_to_class[label_arg]


    def _draw_boxes(self, selected_boxes, original_image_array):
        figure, axis = plt.subplots(1)
        axis.imshow(original_image_array)
        x_min = selected_boxes[:, 0]
        y_min = selected_boxes[:, 1]
        x_max = selected_boxes[:, 2]
        y_max = selected_boxes[:, 3]
        width = x_max - x_min
        height = y_max - y_min
        classes = selected_boxes[:, 4:]
        num_boxes = len(selected_boxes)
        for box_arg in range(num_boxes):
            x_min_box = int(x_min[box_arg])
            y_min_box = int(y_min[box_arg])
            box_width = int(width[box_arg])
            box_height = int(height[box_arg])
            box_class = classes[box_arg]
            label_arg = np.argmax(box_class)
            score = box_class[label_arg]
            class_name = self.arg_to_class[label_arg]
            color = self.colors[label_arg]
            display_text = '{:0.2f}, {}'.format(score, class_name)
            cv2.rectangle(original_image_array, (x_min_box, y_min_box),
                        (x_min_box + box_width, y_min_box + box_height),
                                                                color, 2)
            cv2.putText(original_image_array, display_text,
                        (x_min_box, y_min_box - 30), self.font,
                        .7, color, 1, cv2.LINE_AA)

    def start_video(self, image):
        image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_array = image_array.astype('float32')
        image_array = cv2.resize(image_array, (300, 300))
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_images(image_array)
        predictions = self.model.predict(image_array)
        predictions = np.squeeze(predictions)
        self.draw_boxes_in_video(predictions, image)
        cv2.imshow('webcam', image)
