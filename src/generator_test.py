from datasets.data_manager import DataManager
from utils.boxes import create_prior_boxes
from utils.inference import plot_box_data
from utils.inference import get_colors
from utils.boxes import unregress_boxes
from utils.boxes import to_point_form
from utils.generator import ImageGenerator
import matplotlib.pyplot as plt

# parameters
dataset_name = 'VOC2012'
batch_size = 20
colors = get_colors(25)
prior_boxes = to_point_form(create_prior_boxes())

# loading training data
data_manager = DataManager(dataset_name, 'train')
train_data = data_manager.load_data()
arg_to_class = data_manager.arg_to_class
# loading validation data
val_data = DataManager(dataset_name, 'val').load_data()

# generating output
generator = ImageGenerator(train_data, val_data, prior_boxes, batch_size)
generated_data = next(generator.flow('train'))
transformed_image_batch = generated_data[0]['input_1']
generated_output = generated_data[1]['predictions']

for batch_arg, transformed_image in enumerate(transformed_image_batch):
    positive_mask = generated_output[batch_arg, :, 4] != 1
    regressed_boxes = generated_output[batch_arg]
    unregressed_boxes = unregress_boxes(regressed_boxes, prior_boxes)
    unregressed_positive_boxes = unregressed_boxes[positive_mask]
    plot_box_data(unregressed_positive_boxes, transformed_image,
                  arg_to_class, colors=colors)
    plt.imshow(transformed_image.astype('uint8'))
    plt.show()
