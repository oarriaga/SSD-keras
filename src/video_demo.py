import matplotlib.pyplot as plt
import numpy as np
import cv2

from datasets import get_class_names
from utils.preprocessing import substract_mean
from utils.inference import detect
from utils.inference import plot_detections


class VideoDemo(object):
    def __init__(self, prior_boxes, dataset_name='VOC2007',
                 box_scale_factors=[.1, .1, .2, .2],
                 background_index=0, lower_probability_threshold=.1,
                 iou_threshold=.2, class_names=None):

        self.prior_boxes = prior_boxes
        self.box_scale_factors = box_scale_factors
        self.background_index = background_index
        self.iou_threshold = iou_threshold
        self.lower_probability_threshold = lower_probability_threshold
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = get_class_names(dataset_name)
        self.num_classes = len(self.class_names)
        self.colors = plt.cm.hsv(np.linspace(0, 1, self.num_classes)).tolist()
        self.colors = np.asarray(self.colors) * 255
        self.arg_to_class = dict(zip(list(range(self.num_classes)),
                                 self.class_names))
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def start_video(self, model):
        camera = cv2.VideoCapture(0)
        while True:
            frame = camera.read()[1]
            if frame is None:
                continue
            image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_array = cv2.resize(image_array, (300, 300))
            image_array = substract_mean(image_array)
            image_array = np.expand_dims(image_array, 0)
            predictions = model.predict(image_array)
            detections = detect(predictions, self.prior_boxes)
            plot_detections(detections, frame, 0.6,
                            self.arg_to_class, self.colors)
            cv2.imshow('webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    from models.ssd import SSD300
    from utils.boxes import create_prior_boxes
    from utils.boxes import to_point_form

    dataset_name = 'VOC2007'
    # weights_path = '../trained_models/SSD300_weights.hdf5'
    # weights_path = '../trained_models/weights.07-3.59.hdf5'
    # weights_path = '../trained_models/weights.03-3.37.hdf5'
    # weights_path = '../trained_models/weights.150-3.57.hdf5'
    # weights_path = '../trained_models/weights.12-4.20.hdf5'
    # weights_path = '../trained_models/weights.02-3.44.hdf5'
    # weights_path = '../trained_models/weights.22-5.01.hdf5'
    # weights_path = '../trained_models/weights.79-6.66.hdf5'
    # weights_path = '../trained_models/weights.64-6.52.hdf5'
    # weights_path = '../trained_models/weights.22-3.85.hdf5'
    # weights_path = '../trained_models/weights.50-3.92.hdf5'
    weights_path = '../trained_models/weights.04-3.79.hdf5'
    model = SSD300(weights_path=weights_path)
    prior_boxes = to_point_form(create_prior_boxes())
    # prior_boxes = to_point_form(prior_boxes)
    video = VideoDemo(prior_boxes, dataset_name)
    video.start_video(model)
