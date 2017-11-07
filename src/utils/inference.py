import numpy as np
from .boxes import unregress_boxes
from .boxes import denormalize_boxes
from .boxes import apply_non_max_suppression
from .boxes import denormalize_box
from .preprocessing import load_image
from .preprocessing import substract_mean
import cv2
import matplotlib.pyplot as plt


def _infer(image_array, model, original_image_shape, prior_boxes):
    box_data_size = model.output_shape[1]
    image_array = substract_mean(image_array)
    image_array = np.expand_dims(image_array, 0)
    predictions = model.predict(image_array)
    detections = detect(predictions, prior_boxes)
    if detections is None:
        return np.zeros(shape=(1, box_data_size))
    detections = denormalize_boxes(detections, original_image_shape)
    return detections


def infer_from_path(image_path, model, prior_boxes):
    target_size = model.input_shape[1:3]
    image, original_image_shape = load_image(image_path, target_size, True)
    detections = _infer(image, model, original_image_shape, prior_boxes)
    return detections


def infer_from_array(image_array, model, original_image_shape, prior_boxes):
    detections = _infer(image_array, model, original_image_shape, prior_boxes)
    return detections


def detect(box_data, prior_boxes, conf_thresh=0.01, nms_thresh=.45,
           top_k=200, variances=[.1, .1, .2, .2]):

    box_data = np.squeeze(box_data)
    regressed_boxes = box_data[:, :4]
    class_predictions = box_data[:, 4:]
    unregressed_boxes = unregress_boxes(regressed_boxes,
                                        prior_boxes, variances)

    num_classes = class_predictions.shape[1]
    output = np.zeros((1, num_classes, top_k, 5))
    for class_arg in range(1, num_classes):
        conf_mask = class_predictions[:, class_arg] >= (conf_thresh)
        scores = class_predictions[:, class_arg][conf_mask]
        if len(scores) == 0:
            continue
        boxes = unregressed_boxes[conf_mask]
        indices, count = apply_non_max_suppression(boxes, scores,
                                                   nms_thresh, top_k)
        scores = np.expand_dims(scores, -1)
        selections = np.concatenate((scores[indices[:count]],
                                    boxes[indices[:count]]), axis=1)

        output[0, class_arg, :count, :] = selections
    return output


# plotting functions ----------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_colors(num_colors=21):
    return plt.cm.hsv(np.linspace(0, 1, num_colors)) * 255


def plot_detections(detections, original_image_array, conf_thresh=0.01,
                    arg_to_class=None, colors=None, font=FONT):
    detections = np.squeeze(detections)
    num_classes = detections.shape[0]
    for class_arg in range(1, num_classes):
        class_detections = detections[class_arg, :]
        confidence_mask = np.squeeze(class_detections[:, 0] > conf_thresh)
        selected_class_detections = class_detections[confidence_mask]
        if len(selected_class_detections) == 0:
            continue
        plot_box_data(selected_class_detections, original_image_array,
                      arg_to_class, class_arg, colors, font)


def plot_box_data(box_data, original_image_array,
                  arg_to_class=None, class_arg=None, colors=None, font=FONT):

    np.apply_along_axis(plot_single_box_data, 1, box_data,
                        original_image_array, arg_to_class, class_arg,
                        colors, font)


def plot_single_box_data(box_data, original_image_array, arg_to_class=None,
                         class_arg=None, colors=None, font=FONT):

    image_shape = original_image_array.shape[0:2]
    box_data_length = len(box_data)

    # case after model predict
    if ((box_data_length > 4) and (class_arg is not None)):
        class_score = box_data[0]
        coordinates = box_data[1:]
        color = colors[class_arg]
        class_name = arg_to_class[class_arg]
        display_text = '{:0.2f}, {}'.format(class_score, class_name)
        x_min, y_min, x_max, y_max = denormalize_box(coordinates, image_shape)
        cv2.putText(original_image_array, display_text, (x_min, y_min - 10),
                    font, .7, color, 1, cv2.LINE_AA)
        thickness = 2

    # case for ground truth
    elif ((box_data_length > 4) and (class_arg is None)):
        class_scores = box_data[4:]
        class_arg = np.argmax(class_scores)
        class_score = class_scores[class_arg]
        coordinates = box_data[:4]
        color = colors[class_arg]
        class_name = arg_to_class[class_arg]
        display_text = '{:0.2f}, {}'.format(class_score, class_name)
        x_min, y_min, x_max, y_max = denormalize_box(coordinates, image_shape)
        cv2.putText(original_image_array, display_text, (x_min, y_min - 10),
                    font, .4, color, 1, cv2.LINE_AA)
        thickness = 1

    # case for only boxes coordinates
    elif box_data_length == 4:
        x_min, y_min, x_max, y_max = denormalize_box(box_data, image_shape)
        color = (255, 0, 0)
        thickness = 1

    cv2.rectangle(original_image_array, (x_min, y_min),
                  (x_max, y_max), color, thickness)
