import numpy as np

class BoundingBoxUtility(object):
    def __init__(self, prior_boxes, background_index=0, overlap_threshold=0.5):
        self.prior_boxes = self._flatten_prior_boxes(prior_boxes)
        self.background_index = background_index
        self.overlap_threshold = overlap_threshold
        self.num_priors = len(self.prior_boxes)

    def _flatten_prior_boxes(self, prior_boxes):
        prior_boxes = [layer_boxes.reshape(-1, 4) for layer_boxes in prior_boxes]
        prior_boxes = np.concatenate(prior_boxes, axis=0)
        return prior_boxes

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

    def _encode_box(self, assigned_box_coordinates,
                    ground_truth_box_coordinates):
        d_box_values = assigned_box_coordinates
        d_box_coordinates = d_box_values[0:4]
        d_x_min = d_box_coordinates[0]
        d_y_min = d_box_coordinates[1]
        d_x_max = d_box_coordinates[2]
        d_y_max = d_box_coordinates[3]
        d_center_x = 0.5 * (d_x_min + d_x_max)
        d_center_y = 0.5 * (d_y_min + d_y_max)
        d_width =  d_x_max - d_x_min
        d_height = d_y_max - d_y_min

        g_box_coordinates = ground_truth_box_coordinates
        g_x_min = g_box_coordinates[0]
        g_y_min = g_box_coordinates[1]
        g_x_max = g_box_coordinates[2]
        g_y_max = g_box_coordinates[3]
        g_width =  g_x_max - g_x_min
        g_height = g_y_max - g_y_min
        g_center_x = 0.5 * (g_x_min + g_x_max)
        g_center_y = 0.5 * (g_y_min + g_y_max)

        g_hat_center_x = (g_center_x - d_center_x) / d_width
        g_hat_center_y = (g_center_y - d_center_y) / d_height
        g_hat_width  = np.log(g_width  / d_width)
        g_hat_height = np.log(g_height / d_height)
        encoded_coordinates = np.array([g_hat_center_x, g_hat_center_y,
                                            g_hat_width, g_hat_height])
        return encoded_coordinates

    def assign_boxes(self, ground_truth):
        # ground_truth include num_boxes, 4 + num_classes (including background)
        #assigned_boxes = np.zeros_like(self.prior_boxes)
        # si estas haciendo la mascar no necesitas hacer los crops, implicitamente
        # se estan haciendo un chingo de crops
        assigned_boxes = np.zeros((self.num_priors, 4 + self.num_classes))
        assigned_boxes[:, 3 + self.background_index] = 1.0
        num_objects_in_image = ground_truth.shape[0]
        for object_arg in range(num_objects_in_image):
            ground_truth_box_coordinates = ground_truth[object_arg, 0:4]
            ious = self._calculate_intersection_over_unions(
                                            ground_truth_box_coordinates)
            ious_mask = ious > self.overlap_threshold
            if not ious_mask.any():
                ious_mask[ious.argmax()] = True
            assigned_box[ious_mask, 0:4] = self.prior_boxes[ious_mask]
            #tengo que meter las clases, y hacerle encode a las pinshis cajas en la mascara

            best_iou_arg = np.argmax(ious)
            assigned_box_coordinates = self.prior_boxes[best_iou_arg, 0:4]
            encoded_box = self._encode_box(assigned_box_coordinates,
                                        ground_truth_box_coordinates)


            assigned_boxes[best_iou_arg] = best_prior_box_coordinates



