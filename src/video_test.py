import matplotlib.pyplot as plt
import numpy as np
import cv2

from utils.utils import get_class_names
from utils.utils import preprocess_images
from utils.boxes import decode_boxes
from utils.boxes import filter_boxes
from utils.visualizer import draw_normalized_box

class VideoTest(object):
    def __init__(self, prior_boxes, box_scale_factors=[.1, .1, .2, .2],
            background_index=0, lower_probability_threshold=.4,
            class_names=None, dataset_name='VOC2007'):

        self.prior_boxes = prior_boxes
        self.box_scale_factors = box_scale_factors
        self.background_index = background_index
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
            ret, frame = camera.read()
            image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_array = image_array.astype('float32')
            image_array = cv2.resize(image_array, (300, 300))
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_images(image_array)
            predictions = model.predict(image_array)
            predictions = np.squeeze(predictions)
            decoded_predictions = decode_boxes(predictions,
                                            self.prior_boxes,
                                            self.box_scale_factors)
            selected_boxes = filter_boxes(decoded_predictions,
                        self.num_classes, self.background_index,
                        self.lower_probability_threshold)
            if len(selected_boxes) == 0:
                continue
            draw_normalized_box(selected_boxes, frame,
                        self.arg_to_class, self.colors, self.font)
            cv2.imshow('webcam', frame)
            if cv2.waitKey(1)&0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    from models.ssd import SSD300
    from utils.boxes import create_prior_boxes
    num_classes = 21
    dataset_name = 'VOC2007'
    weights_filename = '../trained_models/weights_SSD300.hdf5'
    model = SSD300(num_classes=num_classes)
    prior_boxes = create_prior_boxes(model)
    model.load_weights(weights_filename)
    video = VideoTest(prior_boxes, dataset_name=dataset_name)
    video.start_video(model)
