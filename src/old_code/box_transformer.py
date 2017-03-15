import numpy as np

class BoxTransformer(object):
    def __init__(self, assigned_truths, ground_truths):
        self.assigned_truths = assigned_truths
        self.transformed_boxes = {}
        self.ground_truths = ground_truths

    def encode_boxes(self):
        for image_key, assigned_truth_values in self.assigned_truths.items():
            num_objects_in_image = assigned_truth_values.shape[0]
            transformed_truths = assigned_truth_values.copy()
            for object_arg in range(num_objects_in_image):
                d_box_values = assigned_truth_values[object_arg].copy()
                d_box_coordinates = d_box_values[0:4]
                d_x_min = d_box_coordinates[0]
                d_y_min = d_box_coordinates[1]
                d_x_max = d_box_coordinates[2]
                d_y_max = d_box_coordinates[3]
                d_center_x = 0.5 * (d_x_min + d_x_max)
                d_center_y = 0.5 * (d_y_min + d_y_max)
                d_width =  d_x_max - d_x_min
                d_height = d_y_max - d_y_min

                g_box_values = self.ground_truths[image_key].copy()
                g_box_coordinates = g_box_values[object_arg, 0:4]
                g_x_min = g_box_coordinates[0]
                g_y_min = g_box_coordinates[1]
                g_x_max = g_box_coordinates[2]
                g_y_max = g_box_coordinates[3]
                g_width =  g_x_max - g_x_min
                g_height = g_y_max - g_y_min
                g_center_x = 0.5 * (g_x_min + g_x_max)
                g_center_y = 0.5 * (g_y_min + g_y_max)

                g_hat_center_x = (g_center_x - d_center_x) / d_width
                g_hat_center_y = (g_center_y - d_center_y) / d_height
                g_hat_width  = np.log(g_width  / d_width)
                g_hat_height = np.log(g_height / d_height)

                transformed_truths[object_arg, 0] = g_hat_center_x
                transformed_truths[object_arg, 1] = g_hat_center_y
                transformed_truths[object_arg, 2] = g_hat_width
                transformed_truths[object_arg, 2] = g_hat_height

                self.transformed_boxes[image_key] = transformed_truths

        return self.transformed_boxes

    def decode_boxes(self, predicted_boxes, prior_boxes, variances):
        """Convert bboxes from local predictions to shifted priors.

        # Arguments
            mbox_loc: Numpy array of predicted locations.
            mbox_priorbox: Numpy array of prior boxes.
            variances: Numpy array of variances.

        # Return
            decode_bbox: Shifted priors.
        """

        prior_x_min = prior_boxes[:, 0]
        prior_y_min = prior_boxes[:, 1]
        prior_x_max = prior_boxes[:, 2]
        prior_y_max = prior_boxes[:, 3]

        prior_width = prior_x_max - prior_x_min
        prior_height = prior_y_max - prior_y_min
        prior_center_x = 0.5 * (prior_x_max + prior_x_min)
        prior_center_y = 0.5 * (prior_y_max + prior_y_min)

        #rename to g_hat_center_x all the other variables 
        pred_center_x = predicted_boxes[:, 0]
        pred_center_y = predicted_boxes[:, 1]
        pred_width = predicted_boxes[:, 2]
        pred_height = predicted_boxes[:, 3]

        decoded_center_x = pred_center_x * prior_width * variances[:, 0]
        decoded_center_x = decoded_center_x + prior_center_x
        decoded_center_y = pred_center_y * prior_width * variances[:, 1]
        decoded_center_y = decoded_center_y + prior_center_y

        decoded_width = np.exp(pred_width * variances[:, 2])
        decoded_width = decoded_width * prior_width
        decoded_height = np.exp(pred_height * variances[:, 3])
        decoded_height = decoded_height * prior_height

        decoded_x_min = decoded_center_x - (0.5 * decoded_width)
        decoded_y_min = decoded_center_y - (0.5 * decoded_height)
        decoded_x_max = decoded_center_x + (0.5 * decoded_width)
        decoded_y_max = decoded_center_y + (0.5 * decoded_height)

        decoded_box = np.concatenate((decoded_x_min[:, None],
                                      decoded_y_min[:, None],
                                      decoded_x_max[:, None],
                                      decoded_y_max[:, None]), axis=-1)
        decoded_box = np.clip(decoded_box, 0.0, 1.0)
        return decoded_box

