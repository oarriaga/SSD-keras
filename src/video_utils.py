import matplotlib.pyplot as plt
import numpy as np
from utils.utils import get_class_names

class VideoTest(object):
    def __init__(self, prior_boxes, box_scale_factors=[.1, .1, .2, .2],
            background_index=0, lower_probability_bound=.9, class_names=None,
            dataset_name='VOC2007'):

        self.prior_boxes = prior_boxes
        self.box_scale_factors = box_scale_factors
        self.background_index = 0
        self.lower_probability_bound = .9
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = get_class_names(dataset_name)
        self.num_classes = len(self.class_names)
        self.colors = plt.cm.hsv(np.linspace(0, 1, self.num_classes)).tolist()
        self.arg_to_class = dict(zip(list(range(self.num_classes)),
                                                self.class_names))

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

    def _draw_normalized_box(self, box_coordinates, original_image_array):
        image_array = np.squeeze(original_image_array)
        image_array = image_array.astype('uint8')
        image_size = image_array.shape[0:2]
        image_size = (image_size[1], image_size[0])
        figure, axis = plt.subplots(1)
        axis.imshow(image_array)
        original_coordinates = self._denormalize_box(box_coordinates, image_size)
        x_min = original_coordinates[:, 0]
        y_min = original_coordinates[:, 1]
        x_max = original_coordinates[:, 2]
        y_max = original_coordinates[:, 3]
        width = x_max - x_min
        height = y_max - y_min
        classes = box_coordinates[:, 4:]
        num_boxes = len(box_coordinates)
        for box_arg in range(num_boxes):
            x_min_box = x_min[box_arg]
            y_min_box = y_min[box_arg]
            box_width = width[box_arg]
            box_height = height[box_arg]
            box_class = classes[box_arg]
            label_arg = np.argmax(box_class)
            score = box_class[label_arg]
            class_name = self.arg_to_class[label_arg]
            color = self.colors[label_arg]
            rectangle = plt.Rectangle((x_min_box, y_min_box),
                            box_width, box_height, fill=False,
                            linewidth=2, edgecolor=color)
            axis.add_patch(rectangle)
            display_text = '{:0.2f}, {}'.format(score, class_name)
            axis.text(x_min_box, y_min_box, display_text, style='italic',
                      bbox={'facecolor':color, 'alpha':0.5, 'pad':10})
        plt.show()

    def draw_boxes(self, predictions, original_image_array):
        decoded_predictions = self._decode_boxes(predictions)
        selected_boxes = self._filter_boxes(decoded_predictions)
        self._draw_normalized_box(selected_boxes, original_image_array)


if __name__ == "__main__":
    from models import SSD300
    from utils.utils import load_image
    from utils.prior_box_creator import PriorBoxCreator
    from scipy.misc import imread

    # loading model
    model = SSD300()
    box_creator = PriorBoxCreator(model)
    prior_boxes = box_creator.create_boxes()
    weights_filename = '../trained_models/ssd300_weights.34-1.54.hdf5'
    model.load_weights(weights_filename)

    #loading image
    image_filename = 'test_resources/fish-bike.jpg'
    original_image_array = imread(image_filename)
    image_array = load_image(image_filename, target_size=(300, 300))
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predictions = np.squeeze(predictions)

    # testing class
    video = VideoTest(prior_boxes)
    video.draw_boxes(predictions, original_image_array)

"""
if __name__ == "__main__":
    from models import SSD300
    from utils.utils import load_image
    from utils.visualizer import Visualizer
    from utils.utils import get_class_names
    from utils.prior_box_creator import PriorBoxCreator
    from utils.prior_box_manager import PriorBoxManager
    from scipy.misc import imread
    model = SSD300()
    box_creator = PriorBoxCreator(model)
    prior_boxes = box_creator.create_boxes()
    box_manager = PriorBoxManager(prior_boxes, box_scale_factors=[.1, .1, .2, .2])
    weights_filename = '../trained_models/ssd300_weights.34-1.54.hdf5'
    model.load_weights(weights_filename)
    image_filename = 'test_resources/fish-bike.jpg'

    image = imread(image_filename)
    image_size = image.shape[0:2]
    image_array = load_image(image_filename, target_size=(300, 300))
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    #selected_boxes = detect_image(predictions)
    class_names = get_class_names('VOC2007')
    num_classes = len(class_names)
    background_index = 0
    lower_bound = 0.9
    predictions = np.squeeze(predictions)
    predictions = box_manager.decode_boxes(predictions)

    encoded_coordinates = predictions[:, :4]
    box_classes = predictions[:, 4:(4 + num_classes)]
    best_classes = np.argmax(box_classes, axis=-1)
    best_probabilities = np.max(box_classes, axis=-1)
    background_mask = best_classes != background_index
    lower_bound_mask = lower_bound < best_probabilities
    mask = np.logical_and(background_mask, lower_bound_mask)
    selected_boxes = predictions[mask, :(4 + num_classes)]

    arg_to_class = dict(zip(list(range(num_classes)), class_names))
    box_visualizer = Visualizer(arg_to_class=arg_to_class)
    box_visualizer.draw_normalized_box(selected_boxes, image)
"""

