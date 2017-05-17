import numpy as np
import tensorflow as tf

class PriorBoxManager(object):
    def __init__(self, prior_boxes, background_id=0, overlap_threshold=.5,
                 num_classes=21, box_scale_factors=[1, 1, 1, 1]):
        super(PriorBoxManager, self).__init__()
        if type(prior_boxes) == list:
            self.prior_boxes = self._flatten_prior_boxes(prior_boxes)
        else:
            self.prior_boxes = prior_boxes
        self.num_priors = self.prior_boxes.shape[0]
        self.num_classes = num_classes
        self.overlap_threshold = overlap_threshold
        self.background_id = background_id
        self.box_scale_factors = box_scale_factors
        self.assigned_prior_boxes = []
        # to be removed eventually (all this nms with tf) 
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                max_output_size=100,
                                                iou_threshold=0.1)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

    def _flatten_prior_boxes(self, prior_boxes):
        prior_boxes = [layer_boxes.reshape(-1, 4)
                       for layer_boxes in prior_boxes]
        prior_boxes = np.concatenate(prior_boxes, axis=0)
        return prior_boxes

    def _calculate_intersection_over_unions(self, ground_truth_data):
        ground_truth_x_min = ground_truth_data[0]
        ground_truth_y_min = ground_truth_data[1]
        ground_truth_x_max = ground_truth_data[2]
        ground_truth_y_max = ground_truth_data[3]
        prior_boxes_x_min = self.prior_boxes[:, 0]
        prior_boxes_y_min = self.prior_boxes[:, 1]
        prior_boxes_x_max = self.prior_boxes[:, 2]
        prior_boxes_y_max = self.prior_boxes[:, 3]
        # calculating the intersection
        intersections_x_min = np.maximum(prior_boxes_x_min, ground_truth_x_min)
        intersections_y_min = np.maximum(prior_boxes_y_min, ground_truth_y_min)
        intersections_x_max = np.minimum(prior_boxes_x_max, ground_truth_x_max)
        intersections_y_max = np.minimum(prior_boxes_y_max, ground_truth_y_max)
        intersected_widths = intersections_x_max - intersections_x_min
        intersected_heights = intersections_y_max - intersections_y_min
        intersected_widths = np.maximum(intersected_widths, 0)
        intersected_heights = np.maximum(intersected_heights, 0)
        intersections = intersected_widths * intersected_heights
        # calculating the union
        prior_box_widths = prior_boxes_x_max - prior_boxes_x_min
        prior_box_heights = prior_boxes_y_max - prior_boxes_y_min
        prior_box_areas = prior_box_widths * prior_box_heights
        ground_truth_width = ground_truth_x_max - ground_truth_x_min
        ground_truth_height = ground_truth_y_max - ground_truth_y_min
        ground_truth_area = ground_truth_width * ground_truth_height
        unions = prior_box_areas + ground_truth_area - intersections
        intersection_over_unions = intersections / unions
        return intersection_over_unions

    def _encode_box(self, assigned_prior_boxes, ground_truth_box):
        d_box_values = assigned_prior_boxes
        d_box_coordinates = d_box_values[:, 0:4]
        d_x_min = d_box_coordinates[:, 0]
        d_y_min = d_box_coordinates[:, 1]
        d_x_max = d_box_coordinates[:, 2]
        d_y_max = d_box_coordinates[:, 3]
        d_center_x = 0.5 * (d_x_min + d_x_max)
        d_center_y = 0.5 * (d_y_min + d_y_max)
        d_width =  d_x_max - d_x_min
        d_height = d_y_max - d_y_min

        g_box_coordinates = ground_truth_box
        g_x_min = g_box_coordinates[0]
        g_y_min = g_box_coordinates[1]
        g_x_max = g_box_coordinates[2]
        g_y_max = g_box_coordinates[3]
        g_width =  g_x_max - g_x_min
        g_height = g_y_max - g_y_min
        g_center_x = 0.5 * (g_x_min + g_x_max)
        g_center_y = 0.5 * (g_y_min + g_y_max)

        scale_center_x = self.box_scale_factors[0]
        scale_center_y = self.box_scale_factors[1]
        scale_width = self.box_scale_factors[2]
        scale_height = self.box_scale_factors[3]

        g_hat_center_x = (g_center_x - d_center_x) / (d_width * scale_center_x)
        g_hat_center_y = (g_center_y - d_center_y) / (d_height * scale_center_y)
        g_hat_width  = np.log(g_width  / d_width) / scale_width
        g_hat_height = np.log(g_height / d_height) / scale_height
        encoded_boxes = np.concatenate([g_hat_center_x.reshape(-1, 1),
                                        g_hat_center_y.reshape(-1, 1),
                                        g_hat_width.reshape(-1, 1),
                                        g_hat_height.reshape(-1, 1)],
                                        axis=1)
        return encoded_boxes

    def _assign_boxes_to_object(self, ground_truth_box, return_iou=True):
        ious = self._calculate_intersection_over_unions(ground_truth_box)
        #print(np.max(ious))
        encoded_boxes = np.zeros((self.num_priors, 4 + return_iou))
        assign_mask = ious > self.overlap_threshold
        # all the other will contain an iou of zero
        #print(np.sum(assign_mask))
        if not assign_mask.any():
            assign_mask[ious.argmax()] = True
        if return_iou:
            encoded_boxes[:, -1][assign_mask] = ious[assign_mask]
        assigned_prior_boxes = self.prior_boxes[assign_mask]
        self.assigned_prior_boxes.append(assigned_prior_boxes)
        assigned_encoded_priors = self._encode_box(assigned_prior_boxes,
                                                   ground_truth_box)
        encoded_boxes[assign_mask, 0:4] = assigned_encoded_priors
        return encoded_boxes.ravel()

    def assign_boxes(self, ground_truth_data):
        self.assigned_prior_boxes = []
        assignments = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        #assignments = np.zeros((self.num_priors, 4 + self.num_classes))
        assignments[:, 4 + self.background_id] = 1.0
        num_objects_in_image = len(ground_truth_data)
        if num_objects_in_image == 0:
            return assignments
        #make this into a function
        encoded_boxes = np.apply_along_axis(self._assign_boxes_to_object,
                                            1, ground_truth_data[:, :4])
        # (num_objects_in_image, num_priors, encoded_coordinates + iou)
        #self.assigned_prior_boxes = np.asarray(self.assigned_prior_boxes)
        #self.assigned_prior_boxes = np.concatenate(self.assigned_prior_boxes, 0)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        # we will take the best boxes for every object in the image
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_indices = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_indices = best_iou_indices[best_iou_mask]
        num_assigned_boxes = len(best_iou_indices)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        # np.arange is needed since you want to do it for every box
        assignments[best_iou_mask, :4] = encoded_boxes[best_iou_indices,
                                                np.arange(num_assigned_boxes),
                                                :4]

        assignments[:, 4][best_iou_mask] = 0
        #assignments[:, 5:-8][best_iou_mask] = ground_truth_data[best_iou_indices, 4:]
        assignments[:, 5:-8][best_iou_mask] = ground_truth_data[best_iou_indices, 5:]
        assignments[:, -8][best_iou_mask] = 1
        return assignments
        #assignments[best_iou_mask, 4:] = ground_truth_data[best_iou_indices, 4:]
        #return assignments

    def decode_boxes(self, predicted_boxes):
        prior_x_min = self.prior_boxes[:, 0]
        prior_y_min = self.prior_boxes[:, 1]
        prior_x_max = self.prior_boxes[:, 2]
        prior_y_max = self.prior_boxes[:, 3]

        prior_width = prior_x_max - prior_x_min
        prior_height = prior_y_max - prior_y_min
        prior_center_x = 0.5 * (prior_x_max + prior_x_min)
        prior_center_y = 0.5 * (prior_y_max + prior_y_min)

        # TODO rename to g_hat_center_x all the other variables 
        pred_center_x = predicted_boxes[:, 0]
        pred_center_y = predicted_boxes[:, 1]
        pred_width = predicted_boxes[:, 2]
        pred_height = predicted_boxes[:, 3]

        scale_center_x = self.box_scale_factors[0]
        scale_center_y = self.box_scale_factors[1]
        scale_width = self.box_scale_factors[2]
        scale_height = self.box_scale_factors[3]

        decoded_center_x = pred_center_x * prior_width * scale_center_x
        decoded_center_x = decoded_center_x + prior_center_x
        decoded_center_y = pred_center_y * prior_height * scale_center_y
        decoded_center_y = decoded_center_y + prior_center_y

        decoded_width = np.exp(pred_width * scale_width)
        decoded_width = decoded_width * prior_width
        decoded_height = np.exp(pred_height * scale_height)
        decoded_height = decoded_height * prior_height

        decoded_x_min = decoded_center_x - (0.5 * decoded_width)
        decoded_y_min = decoded_center_y - (0.5 * decoded_height)
        decoded_x_max = decoded_center_x + (0.5 * decoded_width)
        decoded_y_max = decoded_center_y + (0.5 * decoded_height)

        decoded_boxes = np.concatenate((decoded_x_min[:, None],
                                      decoded_y_min[:, None],
                                      decoded_x_max[:, None],
                                      decoded_y_max[:, None]), axis=-1)
        decoded_boxes = np.clip(decoded_boxes, 0.0, 1.0)
        if predicted_boxes.shape[1] > 4:
            decoded_boxes = np.concatenate([decoded_boxes,
                            predicted_boxes[:, 4:]], axis=-1)
        return decoded_boxes


    def detection_out2(self, predictions, background_label_id=0, keep_top_k=100,
                      confidence_threshold=0.01):
        """Do non maximum suppression (nms) on prediction results.

        # Arguments
            predictions: Numpy array of predicted values.
            num_classes: Number of classes for prediction.
            background_label_id: Label of background class.
            keep_top_k: Number of total bboxes to be kept per image
                after nms step.
            confidence_threshold: Only consider detections,
                whose confidences are larger than a threshold.

        # Return
            results: List of predictions for every picture. Each prediction is:
                [label, confidence, xmin, ymin, xmax, ymax]
        """
        box_coordinates = predictions[:, :, :4]
        box_classes = predictions[:, :, 4:-8]
        num_samples = len(box_coordinates)
        results = []
        for sample_arg in range(num_samples):
            results.append([])
            decoded_boxes = self.decode_boxes(box_coordinates[sample_arg])
            for class_arg in range(self.num_classes):
                if class_arg == background_label_id:
                    continue
                class_probabilities = box_classes[sample_arg, :, class_arg]
                class_probabilities_mask = class_probabilities > confidence_threshold
                if len(class_probabilities[class_probabilities_mask]) > 0:
                    positive_decoded_boxes = decoded_boxes[class_probabilities_mask]
                    positive_class_probabilities = class_probabilities[class_probabilities_mask]

                    input_dictionary = {self.boxes: positive_decoded_boxes,
                                        self.scores: positive_class_probabilities}
                    indices = self.sess.run(self.nms, feed_dict=input_dictionary)
                    selected_boxes = positive_decoded_boxes[indices]
                    probabilities = positive_class_probabilities[indices][:, None]
                    labels = class_arg * np.ones((len(indices), 1))
                    c_pred = np.concatenate((labels, probabilities, selected_boxes),
                                            axis=1)
                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                results[-1] = results[-1][:keep_top_k]
        return results

    def detection_out(self, predictions, background_label_id=0, keep_top_k=100,
                      confidence_threshold=0.01):
        """Do non maximum suppression (nms) on prediction results.

        # Arguments
            predictions: Numpy array of predicted values.
            num_classes: Number of classes for prediction.
            background_label_id: Label of background class.
            keep_top_k: Number of total bboxes to be kept per image
                after nms step.
            confidence_threshold: Only consider detections,
                whose confidences are larger than a threshold.

        # Return
            results: List of predictions for every picture. Each prediction is:
                [label, confidence, xmin, ymin, xmax, ymax]
        """
        mbox_loc = predictions[:, :, :4]
        variances = predictions[:, :, -4:]
        mbox_priorbox = predictions[:, :, -8:-4]
        mbox_conf = predictions[:, :, 4:-8]
        results = []
        for i in range(len(mbox_loc)):
            results.append([])
            #decode_bbox = self.decode_boxes(mbox_loc[i],
                                            #mbox_priorbox[i], variances[i])

            decode_bbox = self.decode_boxes(mbox_loc[i])
            for c in range(self.num_classes):
                if c == background_label_id:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    feed_dict = {self.boxes: boxes_to_process,
                                 self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes),
                                            axis=1)
                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                results[-1] = results[-1][:keep_top_k]
        return results

