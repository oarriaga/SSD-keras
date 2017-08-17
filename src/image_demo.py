from models import SSD300
from preprocessing import load_image
from datasets import get_class_names
from datasets import get_arg_to_class
from utils.inference import infer
from visualizer import draw_image_boxes

# parameters
dataset_name = 'VOC2007'
image_path = '../images/boys.jpg'
weights_path = '../trained_models/SSD300_weights.hdf5'
model = SSD300(weights_path=weights_path)
class_names = get_class_names(dataset_name)
arg_to_class = get_arg_to_class(class_names)
original_image_array = load_image(image_path)[0]
original_image_shape = original_image_array.shape[1:3]

detections = infer(image_path, model, original_image_shape)
draw_image_boxes(detections, original_image_array, arg_to_class)
