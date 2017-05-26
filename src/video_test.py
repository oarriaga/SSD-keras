import matplotlib.pyplot as plt
import numpy as np
from utils.utils import get_class_names
from utils.utils import preprocess_images
import cv2
import tensorflow as tf

class VideoTest(object):
    def __init__(self, prior_boxes, box_scale_factors=[.1, .1, .2, .2],
            background_index=0, lower_probability_bound=.5, class_names=None,
            dataset_name='VOC2007'):

        self.prior_boxes = prior_boxes
        self.box_scale_factors = box_scale_factors
        self.background_index = 0
        self.lower_probability_bound = lower_probability_bound
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = get_class_names(dataset_name)
        self.num_classes = len(self.class_names)
        self.colors = plt.cm.hsv(np.linspace(0, 1, self.num_classes)).tolist()
        self.colors = np.asarray(self.colors) * 255
        self.arg_to_class = dict(zip(list(range(self.num_classes)),
                                                self.class_names))
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        #num_prior_boxes = len(prior_boxes)
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None))
        self.top_k = 100
        self.iou_threshold = .1
        self.non_maximum_supression = tf.image.non_max_suppression(self.boxes,
                                                    self.scores, 100,
                                            iou_threshold=.2)
        self.session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))



    def _filter_boxes(self, predictions):
        predictions = np.squeeze(predictions)
        predictions = self._decode_boxes(predictions)
        box_classes = predictions[:, 4:(4 + self.num_classes)]
        best_classes = np.argmax(box_classes, axis=-1)
        best_probabilities = np.max(box_classes, axis=-1)
        background_mask = best_classes != self.background_index
        lower_bound_mask = self.lower_probability_bound < best_probabilities
        mask = np.logical_and(background_mask, lower_bound_mask)
        selected_boxes = predictions[mask, :(4 + self.num_classes)]
        return selected_boxes

    def _filter_boxes_2(self, predictions):
        predictions = np.squeeze(predictions)
        predictions = self._decode_boxes(predictions)
        box_classes = predictions[:, 4:(4 + self.num_classes)]
        best_classes = np.argmax(box_classes, axis=-1)
        best_probabilities = np.max(box_classes, axis=-1)
        background_mask = best_classes != self.background_index
        lower_bound_mask = self.lower_probability_bound < best_probabilities
        mask = np.logical_and(background_mask, lower_bound_mask)
        selected_boxes = predictions[mask, :(4 + self.num_classes)]
        return selected_boxes


    def apply_non_max_suppression_tf(self, boxes, class_probabilities):
        #print('box_shape:', boxes.shape)
        #print('class_prob_shape:', class_probabilities.shape)
        #self.boxes = tf.placeholder(dtype='float32', shape=boxes.shape)
        #self.scores = tf.placeholder(dtype='float32', shape=class_probabilities.shape)
        num_boxes = len(boxes)
        self.boxes = tf.placeholder(dtype='float32', shape=(num_boxes, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(num_boxes))
        feed_dict = {self.boxes: boxes, self.scores: class_probabilities}
        self.non_maximum_supression = tf.image.non_max_suppression(self.boxes,
                                                            self.scores, 100,
                                                            iou_threshold=.1)
        #print(feed_dict)
        #print('Hola aqui estoy')
        #self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        #self.scores = tf.placeholder(dtype='float32', shape=(None,))
        indices = self.session.run(self.non_maximum_supression,
                                            feed_dict=feed_dict)
        return indices

    def _calculate_intersection_over_unions(self, ground_truth_data, prior_boxes):
        ground_truth_x_min = ground_truth_data[0]
        ground_truth_y_min = ground_truth_data[1]
        ground_truth_x_max = ground_truth_data[2]
        ground_truth_y_max = ground_truth_data[3]
        prior_boxes_x_min = prior_boxes[:, 0]
        prior_boxes_y_min = prior_boxes[:, 1]
        prior_boxes_x_max = prior_boxes[:, 2]
        prior_boxes_y_max = prior_boxes[:, 3]
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



    def apply_non_max_suppression_fast(self, boxes, iou_threshold=.2):
        """ This function should be modified to include class comparison.
        I believe that the current implementation might filter smaller
        boxes within a big box even tough they are from different classes
        """
        if len(boxes) == 0:
                return []
        selected_indices = []
        x_min = boxes[:, 0]
        y_min = boxes[:, 1]
        x_max = boxes[:, 2]
        y_max = boxes[:, 3]
        classes = boxes[:, 4:]
        sorted_box_indices = np.argsort(y_max)
        while len(sorted_box_indices) > 0:
                last = len(sorted_box_indices) - 1
                i = sorted_box_indices[last]
                selected_indices.append(i)
                box = [x_min[i], y_min[i], x_max[i], y_max[i]]
                box = np.asarray(box)
                print(box.shape)
                print(x_min[sorted_box_indices[:last]].shape)
                print(y_max[sorted_box_indices[:last]].shape)
                print(x_min[sorted_box_indices[:last]].shape)
                print(y_max[sorted_box_indices[:last]].shape)
                test_boxes = [x_min[sorted_box_indices[:last], None],
                         y_min[sorted_box_indices[:last], None],
                         x_max[sorted_box_indices[:last], None],
                         y_max[sorted_box_indices[:last], None]]
                #boxes = np.asarray(boxes)
                test_boxes = np.concatenate(test_boxes, axis=-1)
                print(boxes.shape)
                iou = self._calculate_intersection_over_unions(box, test_boxes)
                #xx1 = np.maximum(x_min[i], x_min[idxs[:last]])
                #yy1 = np.maximum(y_min[i], y_min[idxs[:last]])
                #xx2 = np.minimum(x_max[i], x_max[idxs[:last]])
                #yy2 = np.minimum(y_max[i], y_max[idxs[:last]])
                #width = np.maximum(0, xx2 - xx1)
                #height = np.maximum(0, yy2 - yy1)
                #overlap = (width * height) / area[idxs[:last]]
                """ Here I can include another condition in the np.where
                in order to delete if and only if the boxes are of the
                same class.
                """
                current_class = np.argmax(classes[i])
                box_classes = np.argmax(classes[sorted_box_indices[:last]], axis=-1)
                class_mask = current_class == box_classes
                print(class_mask)
                #print(overlap)
                overlap_mask = iou > iou_threshold
                #print(overlap_mask)
                delete_mask = np.logical_and(overlap_mask, class_mask)
                sorted_box_indices = np.delete(sorted_box_indices, np.concatenate(([last],
                        np.where(delete_mask)[0])))
        return boxes[selected_indices]


    def _decode_boxes(self, predicted_boxes):
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

    def _denormalize_box(self, box_coordinates, image_size):
        x_min = box_coordinates[:, 0]
        y_min = box_coordinates[:, 1]
        x_max = box_coordinates[:, 2]
        y_max = box_coordinates[:, 3]
        original_image_width, original_image_height = image_size
        x_min = x_min * original_image_width
        y_min = y_min * original_image_height
        x_max = x_max * original_image_width
        y_max = y_max * original_image_height
        return np.concatenate([x_min[:, None], y_min[:, None],
                               x_max[:, None], y_max[:, None]], axis=1)

    def _draw_normalized_box(self, box_data, original_image_array, tf_nms=False):
        image_array = np.squeeze(original_image_array)
        image_array = image_array.astype('uint8')
        image_size = image_array.shape[0:2]
        image_size = (image_size[1], image_size[0])
        box_classes = box_data[:, 4:]
        box_coordinates = box_data[:, 0:4]
        original_coordinates = self._denormalize_box(box_coordinates,
                                                            image_size)
        box_data = np.concatenate([original_coordinates, box_classes], axis=-1)
        if tf_nms:
            #print(box_data.shape)
            selected_indices = self.apply_non_max_suppression_tf(box_data[:, 0:4],
                                                             np.max(box_data[:, 4:],axis=-1))
            box_data = box_data[selected_indices]
        else:
            box_data = self.apply_non_max_suppression_fast(box_data)

        if len(box_data) == 0:
            return
        figure, axis = plt.subplots(1)
        axis.imshow(image_array)
        x_min = box_data[:, 0]
        y_min = box_data[:, 1]
        x_max = box_data[:, 2]
        y_max = box_data[:, 3]
        width = x_max - x_min
        height = y_max - y_min
        classes = box_data[:, 4:]
        num_boxes = len(box_data)
        for box_arg in range(num_boxes):
            x_min_box = int(x_min[box_arg])
            y_min_box = int(y_min[box_arg])
            box_width = int(width[box_arg])
            box_height = int(height[box_arg])
            box_class = classes[box_arg]
            label_arg = np.argmax(box_class)
            score = box_class[label_arg]
            class_name = self.arg_to_class[label_arg]
            color = self.colors[label_arg]
            display_text = '{:0.2f}, {}'.format(score, class_name)
            cv2.rectangle(original_image_array, (x_min_box, y_min_box),
                        (x_min_box + box_width, y_min_box + box_height),
                                                                color, 2)
            cv2.putText(original_image_array, display_text,
                        (x_min_box, y_min_box - 30), self.font,
                        .7, color, 1, cv2.LINE_AA)

    def draw_boxes_in_video(self, predictions, original_image_array):
        decoded_predictions = self._decode_boxes(predictions)
        selected_boxes = self._filter_boxes(decoded_predictions)
        if len(selected_boxes) == 0:
            return
        #if len(decoded_predictions) == 0:
            #return
        self._draw_normalized_box(selected_boxes, original_image_array)

    def start_video(self, model):
        camera = cv2.VideoCapture(0)
        while True:
            ret, frame = camera.read()
            #print(frame.shape)
            image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_array = image_array.astype('float32')
            image_array = cv2.resize(image_array, (300, 300))
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_images(image_array)
            predictions = model.predict(image_array)
            predictions = np.squeeze(predictions)
            self.draw_boxes_in_video(predictions, frame)
            cv2.imshow('webcam', frame)
            if cv2.waitKey(1)&0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    from models import SSD300
    from utils.prior_box_creator import PriorBoxCreator
    num_classes = 21
    dataset_name = 'VOC2007'
    model = SSD300(num_classes=num_classes)
    box_creator = PriorBoxCreator(model)
    prior_boxes = box_creator.create_boxes()
    weights_filename = '../trained_models/weights_SSD300.hdf5'
    model.load_weights(weights_filename)
    video = VideoTest(prior_boxes, dataset_name=dataset_name)
    video.start_video(model)
