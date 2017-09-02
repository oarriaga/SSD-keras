# from ..box_utils import nms
from utils.boxes import unregress_boxes
from utils.tf_boxes import apply_non_max_suppression
import numpy as np


class Detect():
    def __init__(self, num_classes=21, bkg_label=0, top_k=200,
                 conf_thresh=0.01, nms_thresh=.45):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = [.1, .1, .2, .2]
        self.output = np.zeros((1, self.num_classes, self.top_k, 5))

    def forward(self, box_data, prior_boxes):
        regressed_boxes = box_data[:, :4]
        class_predictions = box_data[:, 4:]
        decoded_boxes = unregress_boxes(regressed_boxes, prior_boxes,
                                        self.variance)
        for class_arg in range(1, self.num_classes):
            class_mask = class_predictions[class_arg] > (self.conf_thresh)
            scores = class_predictions[class_arg][class_mask]
            if len(scores) == 0:
                continue
            boxes = decoded_boxes[class_mask]
            indices = apply_non_max_suppression(boxes, scores,
                                                self.nms_thresh,
                                                self.top_k)
            count = len(indices)
            selections = np.concatenate((scores[indices, None],
                                         boxes[indices, None]))
            self.output[1, class_arg, :count] = selections
        return self.output
