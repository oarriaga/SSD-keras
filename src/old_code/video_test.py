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

