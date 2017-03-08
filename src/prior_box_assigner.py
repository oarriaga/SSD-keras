import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
import random

class PriorBoxAssigner(object):
    def __init__(self, prior_boxes, ground_truths):
        self.ground_truth_boxes = ground_truths
        self.prior_boxes = self._flatten_prior_boxes(prior_boxes)
        self.assigned_boxes = self.ground_truth_boxes.copy()
        self.intersection_over_unions = []

    def _flatten_prior_boxes(self, prior_boxes):
        prior_boxes = [boxes.reshape(-1, 4) for boxes in prior_boxes]
        prior_boxes = np.concatenate(prior_boxes, axis=0)
        return prior_boxes

    def _flatten_ground_truths(self, ground_truths):
        ground_truth_boxes = []
        self.ground_truth_keys = ground_truths.keys()
        for image_key in self.ground_truth_keys:
            image_ground_truths = ground_truths[image_key]
            ground_truth_boxes.append(image_ground_truths)
        return np.concatenate(ground_truth_boxes, axis=0)

    def assign_boxes(self):
        for image_key in self.ground_truth_boxes.keys():
            num_objects_in_image = self.ground_truth_boxes[image_key].shape[0]
            for object_arg in range(num_objects_in_image):
                ground_truth = self.ground_truth_boxes[image_key][object_arg][0:4]
                ious = self._calculate_intersection_over_unions(ground_truth)
                best_iou = np.max(ious)
                self.intersection_over_unions.append(best_iou)
                best_iou_arg = np.argmax(ious)
                best_coordinates = self.prior_boxes[best_iou_arg]
                self.assigned_boxes[image_key][object_arg][0:4] = best_coordinates
        return self.assigned_boxes

    def _calculate_intersection_over_unions(self, ground_truth):
        prior_boxes_x_min = self.prior_boxes[:, 0]
        prior_boxes_y_min = self.prior_boxes[:, 1]
        prior_boxes_x_max = self.prior_boxes[:, 2]
        prior_boxes_y_max = self.prior_boxes[:, 3]
        ground_truth_x_min = ground_truth[0]
        ground_truth_y_min = ground_truth[1]
        ground_truth_x_max = ground_truth[2]
        ground_truth_y_max = ground_truth[3]
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

    def imread(self, image_path):
        return imread(image_path)

    def resize_image(self, image_array, size):
        return imresize(image_array, size)

    def draw_assigned_boxes(self, image_prefix, image_shape=(300,300),
                                                        image_key=None):
        if image_key == None:
            random_key = random.choice(list(self.assigned_boxes.keys()))
        box_coordinates = self.assigned_boxes[random_key][0:4]
        image_path = image_prefix + str(random_key)
        image_array = self.imread(image_path)
        image_array = self.resize_image(image_array, image_shape)
        figure, axis = plt.subplots(1)
        axis.imshow(image_array)
        box_coordinates = np.squeeze(box_coordinates)
        num_boxes = len(box_coordinates)
        decoded_coordinates = self.decode_box(box_coordinates, image_shape)
        x_min = decoded_coordinates[0]
        y_min = decoded_coordinates[1]
        x_max = decoded_coordinates[2]
        y_max = decoded_coordinates[3]
        box_width = x_max - x_min
        box_height = y_max - y_min
        print(num_boxes)
        if len(box_coordinates.shape) == 1:
            rectangle = plt.Rectangle((x_min, y_min), box_width, box_height,
                            linewidth=1, edgecolor='r', facecolor='none')
            axis.add_patch(rectangle)
        else:
            for box_arg in range(num_boxes):
                rectangle = plt.Rectangle((x_min[box_arg], y_min[box_arg]),
                                box_width[box_arg], box_height[box_arg],
                                linewidth=1, edgecolor='r', facecolor='none')
                axis.add_patch(rectangle)
        plt.show()

    def decode_box(self, box_coordinates, image_shape):
        box_coordinates = np.squeeze(box_coordinates)
        if len(box_coordinates.shape) == 2:
            x_min = box_coordinates[:, 0]
            y_min = box_coordinates[:, 1]
            x_max = box_coordinates[:, 2]
            y_max = box_coordinates[:, 3]
        else:
            x_min = box_coordinates[0]
            y_min = box_coordinates[1]
            x_max = box_coordinates[2]
            y_max = box_coordinates[3]
        original_image_width, original_image_height = image_shape
        x_min = x_min * original_image_width
        y_min = y_min * original_image_height
        x_max = x_max * original_image_width
        y_max = y_max * original_image_height
        return [x_min, y_min, x_max, y_max]

if __name__ == '__main__':
    from models import SSD300
    from prior_box_creator import PriorBoxCreator
    from XML_parser import XMLParser

    model = SSD300((300,300,3))
    box_creator = PriorBoxCreator()
    box_creator.create_boxes(model)
    prior_boxes = box_creator.prior_boxes
    data_path = '../datasets/VOCdevkit/VOC2007/'
    ground_truths = XMLParser(data_path+'Annotations/').data
    prior_box_manager = PriorBoxAssigner(prior_boxes, ground_truths)
    assigned_boxes = prior_box_manager.assign_boxes()
    prior_box_manager.draw_assigned_boxes(data_path+'JPEGImages/')








