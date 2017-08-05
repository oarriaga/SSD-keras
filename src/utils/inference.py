import numpy as np
from .boxes import decode_boxes
from .boxes import filter_boxes
from .boxes import denormalize_box
# from .boxes import apply_non_max_suppression
from .tf_boxes import apply_non_max_suppression


def predict(model, image_array, prior_boxes, original_image_shape,
            num_classes=21, class_threshold=.1,
            iou_nms_threshold=.45, background_index=0,
            box_scale_factors=[.1, .1, .2, .2], input_size=(300, 300)):

    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predictions = np.squeeze(predictions)
    decoded_predictions = decode_boxes(predictions, prior_boxes)
    selected_boxes = filter_boxes(decoded_predictions,
                                  num_classes, background_index,
                                  class_threshold)
    if len(selected_boxes) == 0:
        return None
    selected_boxes = denormalize_box(selected_boxes, original_image_shape)
    supressed_boxes = []
    for class_arg in range(1, num_classes):
        best_classes = np.argmax(selected_boxes[:, 4:], axis=1)
        class_mask = best_classes == class_arg
        class_boxes = selected_boxes[class_mask]
        if len(class_boxes) == 0:
            continue
        class_boxes = apply_non_max_suppression(class_boxes, iou_nms_threshold)
        supressed_boxes.append(class_boxes)
    if len(supressed_boxes) == 0:
        return None
    supressed_boxes = np.concatenate(supressed_boxes, axis=0)
    return supressed_boxes
