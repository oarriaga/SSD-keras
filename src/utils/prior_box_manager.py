import numpy as np

class PriorBoxManager(object):
    """docstring for PriorBoxManager"""
    def __init__(self, prior_boxes, overlap_threshold=.5, background_id=0,
                 num_classes=21):
        super(PriorBoxManager, self).__init__()
        self.prior_boxes = self._flatten_prior_boxes(prior_boxes)
        self.num_priors = self.prior_boxes.shape[0]
        self.num_classes = num_classes
        self.overlap_threshold = overlap_threshold
        self.background_id = background_id

    def _flatten_prior_boxes(self, prior_boxes):
        prior_boxes = [layer_boxes.reshape(-1, 4)
                       for layer_boxes in prior_boxes]
        prior_boxes = np.concatenate(prior_boxes, axis=0)
        return prior_boxes

    def _calculate_intersection_over_unions(self, ground_truth_data):
        ground_truth_x_min = ground_truth_data[0]
        ground_truth_y_min = ground_truth_data[1]
        ground_truth_x_max = ground_truth_data[2]
        ground_truth_y_max = ground_truth_data[3]
        prior_boxes_x_min = self.prior_boxes[:, 0]
        prior_boxes_y_min = self.prior_boxes[:, 1]
        prior_boxes_x_max = self.prior_boxes[:, 2]
        prior_boxes_y_max = self.prior_boxes[:, 3]
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

    def _assign_boxes_to_object(self, ground_truth_box, return_iou=True):
        ious = self._calculate_intersection_over_unions(ground_truth_box)
        print(np.max(ious))
        encoded_boxes = np.zeros((self.num_priors, 4 + return_iou))
        assign_mask = ious > self.overlap_threshold
        print(np.sum(assign_mask))
        if not assign_mask.any():
            assign_mask[ious.argmax()] = True
        if return_iou:
            encoded_boxes[:, -1][assign_mask] = ious[assign_mask]
        assigned_priors = self.prior_boxes[assign_mask]
        assigned_encoded_priors = self._encode_box(assigned_priors,
                                                   ground_truth_box)
        encoded_boxes[assign_mask, 0:4] = assigned_encoded_priors
        return encoded_boxes.ravel()

    def assign_boxes(self, ground_truth_data):
        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignment[:, 4 + self.background_id] = 1.0
        num_objects_in_image = len(ground_truth_data)
        if num_objects_in_image == 0:
            return assignment
        encoded_boxes = np.apply_along_axis(self._assign_boxes_to_object,
                                            1, ground_truth_data[:, :4])
        print(encoded_boxes.shape)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        print(encoded_boxes.shape)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        print(best_iou) #aquí te quedaste 
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,
                                                         np.arange(assign_num),
                                                         :4]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = ground_truth_data[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1
        return assignment

    def _encode_box(self, assigned_prior_boxes, ground_truth_box):
        d_box_values = assigned_prior_boxes
        d_box_coordinates = d_box_values[:, 0:4]
        d_x_min = d_box_coordinates[:, 0]
        d_y_min = d_box_coordinates[:, 1]
        d_x_max = d_box_coordinates[:, 2]
        d_y_max = d_box_coordinates[:, 3]
        d_center_x = 0.5 * (d_x_min + d_x_max)
        d_center_y = 0.5 * (d_y_min + d_y_max)
        d_width =  d_x_max - d_x_min
        d_height = d_y_max - d_y_min

        g_box_coordinates = ground_truth_box
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
        encoded_boxes = np.concatenate([g_hat_center_x.reshape(-1, 1),
                                        g_hat_center_y.reshape(-1, 1),
                                        g_hat_width.reshape(-1, 1),
                                        g_hat_height.reshape(-1, 1)],
                                        axis=1)
        return encoded_boxes
