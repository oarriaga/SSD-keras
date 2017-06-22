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

def regress_boxes(assigned_prior_boxes, ground_truth_box, box_scale_factors):
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

    scale_center_x = box_scale_factors[0]
    scale_center_y = box_scale_factors[1]
    scale_width = box_scale_factors[2]
    scale_height = box_scale_factors[3]

    g_hat_center_x = (g_center_x - d_center_x) / (d_width * scale_center_x)
    g_hat_center_y = (g_center_y - d_center_y) / (d_height * scale_center_y)
    g_hat_width  = np.log(g_width  / d_width) / scale_width
    g_hat_height = np.log(g_height / d_height) / scale_height
    regressed_boxes = np.concatenate([g_hat_center_x.reshape(-1, 1),
                                    g_hat_center_y.reshape(-1, 1),
                                    g_hat_width.reshape(-1, 1),
                                    g_hat_height.reshape(-1, 1)],
                                    axis=1)
    return regressed_boxes

def decode_boxes(predicted_boxes, prior_boxes, box_scale_factors):
    """Decode from regressed coordinates in predicted_boxes
    to box_coordinates in prior_boxes by applying the inverse function of the
    regress_boxes function.

    Arguments:
        predicted_boxes: numpy array with shape (num_assigned_priors, 4)
        indicating x_min, y_min, x_max and y_max for every prior box.
        prior_boxes: numpy array with shape (4) indicating
        x_min, y_min, x_max and y_max of the ground truth box.
        box_scale_factors: numpy array with shape (num_boxes, 4)
        Which represents a scaling of the localization gradient.
        (https://github.com/weiliu89/caffe/issues/155)

    Raises:
        ValueError: if the number of predicted_boxes is not the same as
        the number of prior boxes.

    Returns:
        decoded_boxes: numpy array with shape (num_predicted_boxes, 4) or
        (num_prior_boxes, 4 + num_clases) which correspond
        to the decoded box coordinates of all prior_boxes.
    """

    if len(predicted_boxes) != len(prior_boxes):
        raise ValueError(
                'Mismatch between predicted_boxes and prior_boxes length')

    prior_x_min = prior_boxes[:, 0]
    prior_y_min = prior_boxes[:, 1]
    prior_x_max = prior_boxes[:, 2]
    prior_y_max = prior_boxes[:, 3]

    prior_width = prior_x_max - prior_x_min
    prior_height = prior_y_max - prior_y_min
    prior_center_x = 0.5 * (prior_x_max + prior_x_min)
    prior_center_y = 0.5 * (prior_y_max + prior_y_min)

    pred_center_x = predicted_boxes[:, 0]
    pred_center_y = predicted_boxes[:, 1]
    pred_width = predicted_boxes[:, 2]
    pred_height = predicted_boxes[:, 3]

    scale_center_x = box_scale_factors[0]
    scale_center_y = box_scale_factors[1]
    scale_width = box_scale_factors[2]
    scale_height = box_scale_factors[3]

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
                                ground_truth_box, box_scale_factors)
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
    #assignments = np.zeros((len(prior_boxes), 4 + num_classes + 8))
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
    assignments[best_iou_mask, :4] = encoded_boxes[best_iou_indices,
                                            np.arange(num_assigned_boxes),
                                            :4]

    assignments[:, 4][best_iou_mask] = 0
    #assignments[:, 5:-8][best_iou_mask] = ground_truth_data[best_iou_indices, 5:]
    assignments[:, 5:][best_iou_mask] = ground_truth_data[best_iou_indices, 5:]
    #assignments[:, -8][best_iou_mask] = 1
    return assignments

def load_model_configurations(model):
    """
    Arguments:
        model: A SSD model with PriorBox layers that indicate the
        parameters of the prior boxes to be created.

    Returns:
        model_configurations: A dictionary of the model parameters.
    """
    model_configurations = []
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == 'PriorBox':
            layer_data = {}
            layer_data['layer_width'] = layer.input_shape[1]
            layer_data['layer_height'] = layer.input_shape[2]
            layer_data['min_size'] = layer.min_size
            layer_data['max_size'] = layer.max_size
            layer_data['aspect_ratios'] = layer.aspect_ratios
            layer_data['num_prior'] = len(layer.aspect_ratios)
            model_configurations.append(layer_data)
    return model_configurations

def create_prior_boxes(model):
    """
    Arguments:
        image_shape: The image shape (width, height) to the
        input model.
        model_configurations: The model configurations created by
        load_model_configurations that indicate the parameters
        inside the PriorBox layers.

    Returns:
        prior_boxes: A numpy array containing all prior boxes
    """
    image_width, image_height = model.input_shape[1:3]
    model_configurations = load_model_configurations(model)

    boxes_parameters = []
    for layer_config in model_configurations:
        layer_width = layer_config["layer_width"]
        layer_height = layer_config["layer_height"]
        # RENAME: to num_aspect_ratios
        num_priors = layer_config["num_prior"]
        aspect_ratios = layer_config["aspect_ratios"]
        min_size = layer_config["min_size"]
        max_size = layer_config["max_size"]

        # .5 is to locate every step in the center of the bounding box
        step_x = 0.5 * (float(image_width) / float(layer_width))
        step_y = 0.5 * (float(image_height) / float(layer_height))

        linspace_x = np.linspace(step_x, image_width - step_x, layer_width)
        linspace_y = np.linspace(step_y, image_height - step_y, layer_height)

        centers_x, centers_y = np.meshgrid(linspace_x, linspace_y)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        assert(num_priors == len(aspect_ratios))
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))

        box_widths = []
        box_heights = []
        for aspect_ratio in aspect_ratios:
            if aspect_ratio == 1 and len(box_widths) == 0:
                box_widths.append(min_size)
                box_heights.append(min_size)
            elif aspect_ratio == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(min_size * max_size))
                box_heights.append(np.sqrt(min_size * max_size))
            elif aspect_ratio != 1:
                box_widths.append(min_size * np.sqrt(aspect_ratio))
                box_heights.append(min_size / np.sqrt(aspect_ratio))
        # we take half of the widths and heights since we are at the center
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        # Normalize to 0-1
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= image_width
        prior_boxes[:, 1::2] /= image_height
        prior_boxes = prior_boxes.reshape(-1, 4)

        # clip to 0-1
        layer_prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        boxes_parameters.append(layer_prior_boxes)

    return np.concatenate(boxes_parameters, axis=0)

def denormalize_box(box_data, original_image_shape):
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
    x_min = box_data[:, 0]
    y_min = box_data[:, 1]
    x_max = box_data[:, 2]
    y_max = box_data[:, 3]
    original_image_height, original_image_width = original_image_shape
    x_min = x_min * original_image_width
    y_min = y_min * original_image_height
    x_max = x_max * original_image_width
    y_max = y_max * original_image_height
    denormalized_box_data = np.concatenate([x_min[:, None], y_min[:, None],
                                    x_max[:, None], y_max[:, None]], axis=1)

    if box_data.shape[1] > 4:
        denormalized_box_data = np.concatenate([denormalized_box_data,
                                            box_data[:, 4:]], axis=-1)
    return denormalized_box_data

def apply_non_max_suppression(boxes, iou_threshold=.2):
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
            iou = calculate_intersection_over_union(box, test_boxes)
            current_class = np.argmax(classes[i])
            box_classes = np.argmax(classes[sorted_box_indices[:last]], axis=-1)
            class_mask = current_class == box_classes
            overlap_mask = iou > iou_threshold
            delete_mask = np.logical_and(overlap_mask, class_mask)
            sorted_box_indices = np.delete(sorted_box_indices, np.concatenate(([last],
                    np.where(delete_mask)[0])))
    return boxes[selected_indices]

def filter_boxes(predictions, num_classes, background_index=0,
                                    lower_probability_threshold=.4):
    predictions = np.squeeze(predictions)
    box_classes = predictions[:, 4:(4 + num_classes)]
    best_classes = np.argmax(box_classes, axis=-1)
    best_probabilities = np.max(box_classes, axis=-1)
    background_mask = best_classes != background_index
    lower_bound_mask = lower_probability_threshold < best_probabilities
    mask = np.logical_and(background_mask, lower_bound_mask)
    selected_boxes = predictions[mask, :(4 + num_classes)]
    return selected_boxes
