import numpy as np
from .preprocessing import preprocess_images
from .boxes import decode_boxes
from .boxes import filter_boxes
from .preprocessing import resize_image_array
from .boxes import denormalize_box
#from .boxes import apply_non_max_suppression
from .tf_boxes import apply_non_max_suppression

def predict(model, image_array, prior_boxes, original_image_shape,
            num_classes=21, lower_probability_threshold=.1,
            iou_threshold=.5, background_index=0,
            box_scale_factors=[.1, .1, .2, .2]):

    image_array = image_array.astype('float32')
    input_size = model.input_shape[1:3]
    image_array = resize_image_array(image_array, input_size)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_images(image_array)
    predictions = model.predict(image_array)
    predictions = np.squeeze(predictions)
    decoded_predictions = decode_boxes(predictions, prior_boxes,
                                              box_scale_factors)
    selected_boxes = filter_boxes(decoded_predictions,
                num_classes, background_index,
                lower_probability_threshold)
    if len(selected_boxes) == 0:
        return None
    selected_boxes = denormalize_box(selected_boxes, original_image_shape)
    selected_boxes = apply_non_max_suppression(selected_boxes, iou_threshold)
    return selected_boxes

