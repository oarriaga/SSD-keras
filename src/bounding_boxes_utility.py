import numpy as np

class BoundingBoxUtility(object):
    """Bounding box utility class for managing prior boxes"""
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,
                non_max_supression_threshold=0.45, top_k=400):
        self.num_classes = num_classes
        self.priors = priors
        if priors is None:
            self.num_priors = 0
        else:
            self.num_priors = len(priors)
        self.overlap_threshold = overlap_threshold
        self._non_max_supression_threshold = non_max_supression_threshold
        self._top_k = top_k
        self.boxes = np.empty(shape=(0, 4), dtype='float32')
        self.scores = np.empty(shape=(0), dtype='float32')
        self.non_max_supression = self.make_non_max_supression(
                                            self.boxes,
                                            self.scores,
                                            self._top_k,
                                            self._non_max_supression_threshold)

    @property
    def non_max_supression_threshold(self):
        return self._non_max_supression_threshold

    @non_max_supression_threshold.setter
    def non_max_supression_threshold(self, value):
        self._non_max_supression_threshold = value
        self.non_max_supression_threshold = self.make_non_max_supression(
                                            self.boxes,
                                            self.scores,
                                            self._top_k,
                                            self._non_max_supression_threshold)

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, value):
        self._top_k = value
        self.non_max_supression = self.make_non_max_supression(
                                            self.boxes,
                                            self.scores,
                                            self._top_k,
                                            self._non_max_supression_threshold)

    def make_non_max_supression(self, boxes, scores, top_k,
                            non_max_supression_threshold):
        sorted_arguments = np.argsort(scores)[::-1]
        boxes = boxes[sorted_arguments]
        return boxes[:top_k]

    def calculate_intersection_over_union(self, box):
        """Compute intersection over union for the given box with all priors.

        # Arguments
            box: Box, numpy tensor of shape (4,).

        # Return
            intersection_over_union: Intersection over union,
                numpy tensor of shape (num_priors).
        """
        # compute intersection
        bottom_left_intersected_corners = np.maximum(self.priors[:, :2],
                                                                box[:2])
        upper_right_intersected_corners = np.minimum(self.priors[:,2:4],
                                                                box[2:])
        intersected_widths_and_heights = (upper_right_intersected_corners -
                                            bottom_left_intersected_corners)
        intersected_widths_and_heights = np.maximum(
                                            intersected_widths_and_heights, 0)
        intersected_boxes_widths  = intersected_widths_and_heights[:, 0]
        intersected_boxes_heights = intersected_widths_and_heights[:, 1]
        intersection = intersected_boxes_widths * intersected_boxes_heights

        # compute union
        box_width  = box[2] - box[0] # x_max - x_min
        box_height = box[3] - box[1] # y_max - y_min
        predicted_area = box_width * box_height
        ground_truth_widths  = self.priors[:, 2] - self.priors[:, 0]
        ground_truth_heights = self.priors[:, 3] - self.priors[:, 1]
        ground_truth_areas = ground_truth_widths * ground_truth_heights
        union = predicted_area + ground_truth_areas - intersection
        # compute intersection over union
        intersection_over_union = intersection / union
        return intersection_over_union

    def encode_box(self, box, return_iou=True):
        """Encode box for training, do it only for assigned priors.

        # Arguments
            box: Box, numpy tensor of shape (4,).
            return_iou: Whether to concat iou to encoded values.

        # Return
            encoded_box: Tensor with encoded box
                numpy tensor of shape (num_priors, 4 + int(return_iou)).
        """
        iou = self.calculate_intersection_over_union(box)
        if return_iou:
            encoded_box = np.zeros((self.num_priors, 4 + 1))
        else:
            encoded_box = np.zeros((self.num_priors, 4))
        assign_mask = iou > self.overlap_threshold
        # if no mask is above the threshold we take the max iou.
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        assigned_priors = self.priors[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_width_and_height = box[2:] - box[:2]
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_width_and_height = (assigned_priors[:, 2:4] -
                                            assigned_priors[:, :2])

        # encoding the boxes equation 2 from [1]
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_width_and_height
        # the line below is probably a misinterpretation of equation 2 in [1]
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_width_and_height /
                                            assigned_priors_width_and_height)
        # the line below is probably a misinterpretation of equation 2 in  [1]
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]
        # flattens the (num_priors,4 + 1) into (num_priors * (4 + 1))
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        """Assign boxes to priors for training.

        # Arguments
            boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes),
                num_classes without background.

        # Return
            assignment: Tensor with assigned boxes,
                numpy tensor of shape (num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                assignment[:, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                assignment[:, -7:] are all 0. See loss for more details.
        """
        assignment = np.zeros(shape=(self.num_priors,  4 + self.num_classes + 8))
        # this might be a flag to count the number of negative examples
        # 4 means index 5 which could probably be the background class
        assignment[:, 4] = 1.0 # is this the background class ? 

        if len(boxes) == 0:
            return assignment
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # you pass from 2 to 3 dim using the first as a dummy dim?
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        best_iou = encoded_boxes[:, :, -1].max(axis=0) # reconsider this???
        best_iou_indices = encoded_boxes[:, :, -1].argmax(axis=0) # and also this???
        best_iou_mask = best_iou > 0
        best_iou_indices = best_iou_indices[best_iou_mask]
        num_assigned_boxes = len(best_iou_indices) # ?
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_indices,
                                                np.arange(num_assigned_boxes),
                                                :4]
        # this probably unsets the best boxes with the background class
        assignment[:, 4][best_iou_mask] = 0
        # this are the classes
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_indices, 4:]
        # this is a mystery of the universe maybe is the background class probability.
        # it is probably the count of positive examples.
        assignment[:, -8][best_iou_mask] = 1
        return assignment
