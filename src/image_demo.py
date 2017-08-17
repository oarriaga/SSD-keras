from models import SSD300
from preprocessing import load_image
from datasets import get_class_names
from datasets import get_arg_to_class
# from utils.inference import infer_from_path
from utils.inference import infer_from_array
from visualizer import draw_image_boxes

# parameters
dataset_name = 'VOC2007'
image_path = '../images/boys.jpg'
weights_path = '../trained_models/SSD300_weights.hdf5'
model = SSD300(weights_path=weights_path)
class_names = get_class_names(dataset_name)
arg_to_class = get_arg_to_class(class_names)
original_image_array = load_image(image_path)[0]

# detections = infer_from_path(image_path, model)
target_size = model.input_shape[1:3]
image_array, original_image_shape = load_image(image_path, target_size)
detections = infer_from_array(image_array, model, original_image_shape)

draw_image_boxes(detections, original_image_array, arg_to_class)
