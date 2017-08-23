import numpy as np
from tensorflow_test import np_methods
from models import SSD300
from preprocessing import load_image
from datasets import get_class_names
from datasets import get_arg_to_class
from preprocessing import substract_mean
from utils.boxes import create_prior_boxes
from utils.boxes import decode_boxes
from tensorflow_test import visualization


# loading model
weights_path = '../trained_models/SSD300_weights.hdf5'
model = SSD300(weights_path=weights_path)

# loading images
image_path = '../images/voc_cars.jpg'
original_image_array = load_image(image_path)[0]
target_size = model.input_shape[1:3]
image_array, original_image_shape = load_image(image_path, target_size)

# variables for visualization
dataset_name = 'VOC2007'
class_names = get_class_names(dataset_name)
arg_to_class = get_arg_to_class(class_names)

# forward pass
image_array = substract_mean(image_array)
image_array = np.expand_dims(image_array, 0)
predictions = model.predict(image_array)

# creating prior boxes
prior_boxes = create_prior_boxes()

# starting tests
predictions = np.squeeze(predictions)
decoded_boxes = decode_boxes(predictions, prior_boxes)
rbboxes = decoded_boxes[:, :4]
rscores = np.max(decoded_boxes[:, 4:], axis=1)
rclasses = np.argmax(decoded_boxes[:, 4:], axis=1)
rbbox_img = [0., 0., 1., 1.]

"""
rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                    rpredictions,
                    rlocalisations,
                    ssd_anchors,
                    decode=True)
"""

rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores,
                                                    rbboxes, top_k=400)
rclasses, rscores, rbboxes = np_methods.bboxes_nms(
            rclasses, rscores, rbboxes, nms_threshold=0.45)
rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
visualization.plt_bboxes(original_image_array, rclasses, rscores, rbboxes)
