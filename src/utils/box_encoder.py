import numpy as np

class BoxEncoder(object):
    def __init__(self, assigned_boxes, ground_truth_boxes):
        self.encoded_boxes = assigned_boxes.copy()
        self.ground_truth_boxes = ground_truth_boxes

    def encode_boxes(self):
        for image_key in self.assigned_boxes.keys():
            d_box_coordinates = self.encoded_boxes[image_key][:, 0:4]
            d_x_min = d_box_coordinates[:, 0]
            d_y_min = d_box_coordinates[:, 1]
            d_x_max = d_box_coordinates[:, 2]
            d_y_max = d_box_coordinates[:, 3]
            d_center_x = 0.5 * (d_x_min + d_x_max)
            d_center_y = 0.5 * (d_y_min + d_y_max)
            d_width =  d_x_max - d_x_min
            d_height = d_y_max - d_y_min

            g_box_coordinates = self.ground_truth_boxes[image_key][:, 0:4]
            g_x_min = g_box_coordinates[:, 0]
            g_y_min = g_box_coordinates[:, 1]
            g_x_max = g_box_coordinates[:, 2]
            g_y_max = g_box_coordinates[:, 3]
            g_width =  g_x_max - g_x_min
            g_height = g_y_max - g_y_min
            g_center_x = 0.5 * (g_x_min + g_x_max)
            g_center_y = 0.5 * (g_y_min + g_y_max)

            g_hat_center_x = (g_center_x - d_center_x) / d_width
            g_hat_center_y = (g_center_y - d_center_y) / d_height

            g_hat_width  = np.log(g_width  / d_width)
            g_hat_height = np.log(g_height / d_height)

            self.encoded_boxes[image_key][:, 0] = g_hat_center_x
            self.encoded_boxes[image_key][:, 1] = g_hat_center_y
            self.encoded_boxes[image_key][:, 2] = g_hat_width
            self.encoded_boxes[image_key][:, 3] = g_hat_height

        return self.encoded_boxes

    def decode_boxes(self, predicted_boxes, prior_boxes, variances):
        """Convert bboxes from local predictions to shifted priors.

        # Arguments
            mbox_loc: Numpy array of predicted locations.
            mbox_priorbox: Numpy array of prior boxes.
            variances: Numpy array of variances.

        # Return
            decode_bbox: Shifted priors.
        """

        prior_width = prior_boxes[:, 2] - prior_boxes[:, 0]
        prior_height = prior_boxes[:, 3] - prior_boxes[:, 1]
        prior_center_x = 0.5 * (prior_boxes[:, 2] + prior_boxes[:, 0])
        prior_center_y = 0.5 * (prior_boxes[:, 3] + prior_boxes[:, 1])
        # ???????????????

        decoded_center_x = predicted_boxes[:, 0] * prior_width * variances[:, 0]
        decoded_center_x = decoded_center_x + prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_width * variances[:, 1]
        decode_bbox_center_y += prior_center_y
        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox


