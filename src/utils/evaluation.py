from .datasets import DataManager
from .datasets import get_class_names
from .preprocessing import load_image
from .boxes import create_prior_boxes
from .boxes import denormalize_box
from .inference import predict
import numpy as np
from tqdm import tqdm


def record_detections(model, dataset_name, data_prefix, image_prefix,
                      class_names=None, iou_threshold=.5,
                      iou_nms_threshold=.45):
    target_size = model.input_shape[1:3]
    if class_names is None:
        class_names = get_class_names(dataset_name)
    data_manager = DataManager(dataset_name, class_names,
                               data_prefix, image_prefix)
    ground_truth_data = data_manager.load_data()
    image_names = sorted(list(ground_truth_data.keys()))
    num_classes = len(class_names)
    prior_boxes = create_prior_boxes()
    difficult_data_flags = data_manager.parser.difficult_objects
    image_ids = []
    predicted_boxes = []
    predicted_scores = []
    difficult_objects = []
    predicted_classes = []
    ground_truth_boxes = []
    ground_truth_classes = []
    for image_name in tqdm(image_names):
        image_path = image_prefix + image_name
        image_array, original_image_size = load_image(image_path, target_size)
        predicted_data = predict(model, image_array, prior_boxes,
                                 original_image_size, num_classes,
                                 iou_threshold, iou_nms_threshold)
        if predicted_data is None:
            predicted_data = np.zeros(shape=(1, 4 + num_classes))
        image_ids.append(image_name)
        predicted_boxes.append(predicted_data[:, :4])
        predicted_classes.append(np.argmax(predicted_data[:, 4:], axis=1))
        predicted_scores.append(np.max(predicted_data[:, 4:], axis=1))
        ground_truth_sample = ground_truth_data[image_name]
        ground_truth_sample = denormalize_box(ground_truth_sample,
                                              original_image_size)
        ground_truth_boxes.append(ground_truth_sample[:, :4])
        ground_truth_classes.append(np.argmax(ground_truth_sample[:, 4:],
                                    axis=1))
        difficult_boxes = difficult_data_flags[image_name]
        difficult_boxes = np.asarray(difficult_boxes, dtype=bool)
        difficult_objects.append(difficult_boxes)

    data_records = {}
    data_records['image_id'] = image_ids
    data_records['detection_boxes'] = predicted_boxes
    data_records['detection_scores'] = predicted_scores
    data_records['detection_classes'] = predicted_classes
    data_records['groundtruth_boxes'] = ground_truth_boxes
    data_records['groundtruth_classes'] = ground_truth_classes
    data_records['difficult'] = difficult_objects
    return data_records
