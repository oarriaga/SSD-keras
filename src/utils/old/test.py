from utils.datasets import DataManager
from utils.datasets import get_class_names
from utils.inference import predict
from utils.preprocessing import load_image
from utils.boxes import create_prior_boxes
from utils.preprocessing import image_to_array
from utils.preprocessing import load_pil_image
from utils.preprocessing import get_image_size
from utils.boxes import denormalize_box
from models.ssd import SSD300
from utils.visualizer import draw_image_boxes
from utils.datasets import get_arg_to_class


dataset_name = 'VOC2007'
data_prefix = '../datasets/VOCtest/VOCdevkit/VOC2007/Annotations/'
image_prefix = '../datasets/VOCtest/VOCdevkit/VOC2007/JPEGImages/'
weights_path = '../trained_models/weights_SSD300.hdf5'
model = SSD300(weights_path=weights_path)
prior_boxes = create_prior_boxes(model)
input_shape = model.input_shape[1:3]
class_threshold = .01
iou_threshold = .5

labels = []
scores = []

class_names = get_class_names(dataset_name)
class_names = [class_names[0]] + [class_names[10]]
class_decoder = get_arg_to_class(class_names)
num_classes = len(class_names)
data_manager = DataManager(dataset_name, class_names,
                            data_prefix, image_prefix)
ground_truth_data = data_manager.load_data()
image_names = sorted(list(ground_truth_data.keys()))
print('Number of images found:', len(image_names))
for image_name in image_names:
    ground_truth_sample = ground_truth_data[image_name]
    image_prefix = data_manager.image_prefix
    image_path = image_prefix + image_name
    image_array = load_image(image_path, input_shape)
    original_image_array = image_to_array(load_pil_image(image_path))
    original_image_size = get_image_size(image_path)
    predicted_data = predict(model, image_array, prior_boxes, original_image_size,
                                    num_classes, class_threshold, iou_threshold)
    ground_truth_sample = denormalize_box(ground_truth_sample, original_image_size)
    ground_truth_boxes_in_image = len(ground_truth_sample)
    #if predicted_data is None:
        #print('Skipped image:', image_name)
        #continue
    draw_image_boxes(predicted_data, original_image_array, class_decoder, normalized=False)



