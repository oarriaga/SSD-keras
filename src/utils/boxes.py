from itertools import product
import numpy as np


def calculate_intersection_over_union(box_data, prior_boxes):
    """Calculate intersection over union of box_data with respect to
    prior_boxes.

    Arguments:
        ground_truth_data: numpy array with shape (4) indicating x_min, y_min,
        x_max and y_max coordinates of the bounding box.
        prior_boxes: numpy array with shape (num_boxes, 4).

    Returns:
        intersections_over_unions: numpy array with shape (num_boxes) which
        corresponds to the intersection over unions of box_data with respect
        to all prior_boxes.
    """
    x_min = box_data[0]
    y_min = box_data[1]
    x_max = box_data[2]
    y_max = box_data[3]
    prior_boxes_x_min = prior_boxes[:, 0]
    prior_boxes_y_min = prior_boxes[:, 1]
    prior_boxes_x_max = prior_boxes[:, 2]
    prior_boxes_y_max = prior_boxes[:, 3]
    # calculating the intersection
    intersections_x_min = np.maximum(prior_boxes_x_min, x_min)
    intersections_y_min = np.maximum(prior_boxes_y_min, y_min)
    intersections_x_max = np.minimum(prior_boxes_x_max, x_max)
    intersections_y_max = np.minimum(prior_boxes_y_max, y_max)
    intersected_widths = intersections_x_max - intersections_x_min
    intersected_heights = intersections_y_max - intersections_y_min
    intersected_widths = np.maximum(intersected_widths, 0)
    intersected_heights = np.maximum(intersected_heights, 0)
    intersections = intersected_widths * intersected_heights
    # calculating the union
    prior_box_widths = prior_boxes_x_max - prior_boxes_x_min
    prior_box_heights = prior_boxes_y_max - prior_boxes_y_min
    prior_box_areas = prior_box_widths * prior_box_heights
    box_width = x_max - x_min
    box_height = y_max - y_min
    ground_truth_area = box_width * box_height
    unions = prior_box_areas + ground_truth_area - intersections
    intersection_over_union = intersections / unions
    return intersection_over_union


def regress_boxes_old(assigned_prior_boxes, ground_truth_box,
                      box_scale_factors):
    """Regress assigned_prior_boxes to ground_truth_box as mentioned in
    Faster-RCNN and Single-shot Multi-box Detector papers.

    Arguments:
        assigned_prior_boxes: numpy array with shape (num_assigned_priors, 4)
        indicating x_min, y_min, x_max and y_max for every prior box.
        ground_truth_box: numpy array with shape (4) indicating
        x_min, y_min, x_max and y_max of the ground truth box.
        box_scale_factors: numpy array with shape (4) containing
        the values for scaling the localization gradient.
        (https://github.com/weiliu89/caffe/issues/155)

    Returns:
        regressed_boxes: numpy array with shape (num_assigned_boxes)
        which correspond to the regressed values of all
        assigned_prior_boxes to the ground_truth_box
    """
    d_box_values = assigned_prior_boxes
    d_box_coordinates = d_box_values[:, 0:4]
    d_x_min = d_box_coordinates[:, 0]
    d_y_min = d_box_coordinates[:, 1]
    d_x_max = d_box_coordinates[:, 2]
    d_y_max = d_box_coordinates[:, 3]
    d_center_x = 0.5 * (d_x_min + d_x_max)
    d_center_y = 0.5 * (d_y_min + d_y_max)
    d_width = d_x_max - d_x_min
    d_height = d_y_max - d_y_min

    g_box_coordinates = ground_truth_box
    g_x_min = g_box_coordinates[0]
    g_y_min = g_box_coordinates[1]
    g_x_max = g_box_coordinates[2]
    g_y_max = g_box_coordinates[3]
    g_width = g_x_max - g_x_min
    g_height = g_y_max - g_y_min
    g_center_x = 0.5 * (g_x_min + g_x_max)
    g_center_y = 0.5 * (g_y_min + g_y_max)

    scale_center_x = box_scale_factors[0]
    scale_center_y = box_scale_factors[1]
    scale_width = box_scale_factors[2]
    scale_height = box_scale_factors[3]

    g_hat_center_x = (g_center_x - d_center_x) / (d_width * scale_center_x)
    g_hat_center_y = (g_center_y - d_center_y) / (d_height * scale_center_y)
    g_hat_width = np.log(g_width / d_width) / scale_width
    g_hat_height = np.log(g_height / d_height) / scale_height
    regressed_boxes = np.concatenate([g_hat_center_x.reshape(-1, 1),
                                     g_hat_center_y.reshape(-1, 1),
                                     g_hat_width.reshape(-1, 1),
                                     g_hat_height.reshape(-1, 1)],
                                     axis=1)
    return regressed_boxes


"""
def regress_boxes(asigned_prior_boxes, ground_truth_box, box_scale_factors):
    x_min_scale, y_min_scale, x_max_scale, y_max_scale = box_scale_factors
    # distance between match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]
"""


def regress_boxes():
    pass


def unregress_boxes(predicted_box_data, prior_boxes,
                    box_scale_factors=[.1, .1, .2, .2]):

    x_min_scale, y_min_scale, x_max_scale, y_max_scale = box_scale_factors
    x_min_predicted = predicted_box_data[:, 0]
    y_min_predicted = predicted_box_data[:, 1]
    x_max_predicted = predicted_box_data[:, 2]
    y_max_predicted = predicted_box_data[:, 3]

    x_min = x_min_predicted * x_min_scale * prior_boxes[:, 2]
    x_min = x_min + prior_boxes[:, 0]

    y_min = y_min_predicted * y_min_scale * prior_boxes[:, 3]
    y_min = y_min + prior_boxes[:, 1]

    x_max = prior_boxes[:, 2] * np.exp(x_max_predicted * x_max_scale)
    y_max = prior_boxes[:, 3] * np.exp(y_max_predicted * y_max_scale)

    unregressed_boxes = np.concatenate([x_min[:, None], y_min[:, None],
                                        x_max[:, None], y_max[:, None]],
                                       axis=1)

    unregressed_boxes[:, :2] -= unregressed_boxes[:, 2:] / 2
    unregressed_boxes[:, 2:] += unregressed_boxes[:, :2]
    unregressed_boxes = np.clip(unregressed_boxes, 0.0, 1.0)
    if predicted_box_data.shape[1] > 4:
        unregressed_boxes = np.concatenate([unregressed_boxes,
                                           predicted_box_data[:, 4:]], axis=-1)
    return unregressed_boxes


def assign_prior_boxes_to_ground_truth(ground_truth_box, prior_boxes,
                                       box_scale_factors, regress=True,
                                       overlap_threshold=.5, return_iou=True):
    """ Assigns and regresses prior boxes to a single ground_truth_box
    data sample.
    TODO: Change this function so that it does not regress the boxes
    automatically. It should only assign them but not regress them!
    Arguments:
        prior_boxes: numpy array with shape (num_prior_boxes, 4)
        indicating x_min, y_min, x_max and y_max for every prior box.
        ground_truth_box: numpy array with shape (4) indicating
        x_min, y_min, x_max and y_max of the ground truth box.
        box_scale_factors: numpy array with shape (num_boxes, 4)
        Which represents a scaling of the localization gradient.
        (https://github.com/weiliu89/caffe/issues/155)

    Returns:
        regressed_boxes: numpy array with shape (num_assigned_boxes)
        which correspond to the regressed values of all
        assigned_prior_boxes to the ground_truth_box
    """
    ious = calculate_intersection_over_union(ground_truth_box, prior_boxes)
    regressed_boxes = np.zeros((len(prior_boxes), 4 + return_iou))
    assign_mask = ious > overlap_threshold
    if not assign_mask.any():
        assign_mask[ious.argmax()] = True
    if return_iou:
        regressed_boxes[:, -1][assign_mask] = ious[assign_mask]
    assigned_prior_boxes = prior_boxes[assign_mask]
    if regress:
        assigned_regressed_priors = regress_boxes(assigned_prior_boxes,
                                                  ground_truth_box,
                                                  box_scale_factors)
        regressed_boxes[assign_mask, 0:4] = assigned_regressed_priors
        return regressed_boxes.ravel()
    else:
        regressed_boxes[assign_mask, 0:4] = assigned_prior_boxes[:, 0:4]
        return regressed_boxes.ravel()


def assign_prior_boxes(prior_boxes, ground_truth_data, num_classes,
                       box_scale_factors, regress=True,
                       overlap_threshold=.5, background_id=0):
    """ Assign and regress prior boxes to all ground truth samples.
    Arguments:
        prior_boxes: numpy array with shape (num_prior_boxes, 4)
        indicating x_min, y_min, x_max and y_max for every prior box.
        ground_truth_data: numpy array with shape (num_samples, 4)
        indicating x_min, y_min, x_max and y_max of the ground truth box.
        box_scale_factors: numpy array with shape (num_boxes, 4)
        Which represents a scaling of the localization gradient.
        (https://github.com/weiliu89/caffe/issues/155)

    Returns:
        assignments: numpy array with shape
        (num_samples, 4 + num_classes + 8)
        which correspond to the regressed values of all
        assigned_prior_boxes to the ground_truth_box
    """
    assignments = np.zeros((len(prior_boxes), 4 + num_classes))
    assignments[:, 4 + background_id] = 1.0
    num_objects_in_image = len(ground_truth_data)
    if num_objects_in_image == 0:
        return assignments
    encoded_boxes = np.apply_along_axis(assign_prior_boxes_to_ground_truth, 1,
                                        ground_truth_data[:, :4], prior_boxes,
                                        box_scale_factors, regress,
                                        overlap_threshold)
    encoded_boxes = encoded_boxes.reshape(-1, len(prior_boxes), 5)
    best_iou = encoded_boxes[:, :, -1].max(axis=0)
    best_iou_indices = encoded_boxes[:, :, -1].argmax(axis=0)
    best_iou_mask = best_iou > 0
    best_iou_indices = best_iou_indices[best_iou_mask]
    num_assigned_boxes = len(best_iou_indices)
    encoded_boxes = encoded_boxes[:, best_iou_mask, :]
    box_sequence = np.arange(num_assigned_boxes)
    assignments[best_iou_mask, :4] = encoded_boxes[best_iou_indices,
                                                   box_sequence, :4]

    assignments[:, 4][best_iou_mask] = 0
    assignments[:, 5:][best_iou_mask] = ground_truth_data[best_iou_indices,
                                                          5:]
    return assignments


def create_prior_boxes(configuration=None):
    if configuration is None:
        configuration = get_configuration_file()
    image_size = configuration['image_size']
    feature_map_sizes = configuration['feature_map_sizes']
    min_sizes = configuration['min_sizes']
    max_sizes = configuration['max_sizes']
    steps = configuration['steps']
    model_aspect_ratios = configuration['aspect_ratios']
    mean = []
    for feature_map_arg, feature_map_size in enumerate(feature_map_sizes):
        step = steps[feature_map_arg]
        min_size = min_sizes[feature_map_arg]
        max_size = max_sizes[feature_map_arg]
        aspect_ratios = model_aspect_ratios[feature_map_arg]
        for y, x in product(range(feature_map_size), repeat=2):
            f_k = image_size / step
            center_x = (x + 0.5) / f_k
            center_y = (y + 0.5) / f_k
            s_k = min_size / image_size
            mean = mean + [center_x, center_y, s_k, s_k]
            s_k_prime = np.sqrt(s_k * (max_size / image_size))
            mean = mean + [center_x, center_y, s_k_prime, s_k_prime]
            for aspect_ratio in aspect_ratios:
                mean = mean + [center_x, center_y, s_k * np.sqrt(aspect_ratio),
                               s_k / np.sqrt(aspect_ratio)]
                mean = mean + [center_x, center_y, s_k / np.sqrt(aspect_ratio),
                               s_k * np.sqrt(aspect_ratio)]

    output = np.asarray(mean).reshape((-1, 4))
    output = np.clip(output, 0, 1)
    return output


def get_configuration_file():
    configuration = {'feature_map_sizes': [38, 19, 10, 5, 3, 1],
                     'image_size': 300,
                     'steps': [8, 16, 32, 64, 100, 300],
                     'min_sizes': [30, 60, 111, 162, 213, 264],
                     'max_sizes': [60, 111, 162, 213, 264, 315],
                     'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                     'variance': [0.1, 0.2]}
    return configuration


def denormalize_boxes(box_data, original_image_shape):
    """
    Arguments:
        box_data: numpy array with shape (num_samples, 4) or
        (num_samples, 4 + num_classes)
        original_image_shape: The original image_shape (width, height)
        of the loaded image shape.

    Returns:
        denormalize_box_data: A numpy array of shape (num_samples, 4)
        or (num_samples, 4 + num_classes) containing the original box
        coordinates.
    """
    # TODO: check of the copy is necessary
    x_min = box_data[:, 0].copy()
    y_min = box_data[:, 1].copy()
    x_max = box_data[:, 2].copy()
    y_max = box_data[:, 3].copy()
    original_image_height, original_image_width = original_image_shape
    x_min = x_min * original_image_width
    y_min = y_min * original_image_height
    x_max = x_max * original_image_width
    y_max = y_max * original_image_height
    denormalized_box_data = np.concatenate([x_min[:, None], y_min[:, None],
                                           x_max[:, None], y_max[:, None]],
                                           axis=1)

    if box_data.shape[1] > 4:
        denormalized_box_data = np.concatenate([denormalized_box_data,
                                               box_data[:, 4:]], axis=-1)
    return denormalized_box_data


def apply_non_max_suppression(boxes, scores, iou_thresh=.45, top_k=200):
    """ non maximum suppression in numpy

    Arguments:
        boxes : array of boox coordinates of shape (num_samples, 4)
            where each columns corresponds to x_min, y_min, x_max, y_max
        scores : array of scores given for each box in 'boxes'
        iou_thresh : float intersection over union threshold for removing boxes
        top_k : int Number of maximum objects per class

    Returns:
        selected_indices : array of integers Selected indices of kept boxes
        num_selected_boxes : int Number of selected boxes
    """

    selected_indices = np.zeros(shape=len(scores))
    if boxes is None or len(boxes) == 0:
            return selected_indices
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]
    areas = (x_max - x_min) * (y_max - y_min)
    remaining_sorted_box_indices = np.argsort(scores)
    remaining_sorted_box_indices = remaining_sorted_box_indices[-top_k:]

    num_selected_boxes = 0
    while len(remaining_sorted_box_indices) > 0:
            best_score_index = remaining_sorted_box_indices[-1]
            selected_indices[num_selected_boxes] = best_score_index
            num_selected_boxes = num_selected_boxes + 1
            if len(remaining_sorted_box_indices) == 1:
                break

            remaining_sorted_box_indices = remaining_sorted_box_indices[:-1]

            best_x_min = x_min[best_score_index]
            best_y_min = y_min[best_score_index]
            best_x_max = x_max[best_score_index]
            best_y_max = y_max[best_score_index]

            remaining_x_min = x_min[remaining_sorted_box_indices]
            remaining_y_min = y_min[remaining_sorted_box_indices]
            remaining_x_max = x_max[remaining_sorted_box_indices]
            remaining_y_max = y_max[remaining_sorted_box_indices]

            inner_x_min = np.maximum(remaining_x_min, best_x_min)
            inner_y_min = np.maximum(remaining_y_min, best_y_min)
            inner_x_max = np.minimum(remaining_x_max, best_x_max)
            inner_y_max = np.minimum(remaining_y_max, best_y_max)

            inner_box_widths = inner_x_max - inner_x_min
            inner_box_heights = inner_y_max - inner_y_min

            inner_box_widths = np.maximum(inner_box_widths, 0.0)
            inner_box_heights = np.maximum(inner_box_heights, 0.0)

            intersections = inner_box_widths * inner_box_heights
            remaining_box_areas = areas[remaining_sorted_box_indices]
            best_area = areas[best_score_index]
            unions = remaining_box_areas + best_area - intersections

            intersec_over_union = intersections / unions
            intersec_over_union_mask = intersec_over_union <= iou_thresh
            remaining_sorted_box_indices = remaining_sorted_box_indices[
                                                intersec_over_union_mask]

    return selected_indices.astype(int), num_selected_boxes
