import matplotlib.pyplot as plt
import numpy as np
import cv2

from utils.datasets import get_class_names
from utils.inference import predict
from utils.visualizer import draw_video_boxes

class VideoTest(object):
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
            selected_boxes = predict(model, image_array, prior_boxes,
                                    frame.shape[0:2], self.num_classes,
                                    self.lower_probability_threshold,
                                    self.iou_threshold,
                                    self.background_index,
                                    self.box_scale_factors)
            if selected_boxes is None:
                continue
            draw_video_boxes(selected_boxes, frame, self.arg_to_class,
                                        self.colors, self.font)

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
    video = VideoTest(prior_boxes, dataset_name)
    video.start_video(model)

