import random
from datasets.data_manager import DataManager
from utils.boxes import create_prior_boxes
from utils.preprocessing import load_image
from utils.inference import plot_box_data
from utils.inference import get_colors
from utils.boxes import assign_prior_boxes
from utils.boxes import to_point_form
from utils.boxes import unregress_boxes
from utils.generator import ImageGenerator
from utils.sequencer_manager import SequenceManager
import matplotlib.pyplot as plt


# data manager
# ------------------------------------------------------------------
split = 'train'
dataset_name = 'VOC2012'
dataset_manager = DataManager(dataset_name, split)
ground_truth_data = dataset_manager.load_data()
class_names = dataset_manager.class_names
print('Found:', len(ground_truth_data), 'images')
print('Class names: \n', class_names)


class_names = ['background', 'diningtable', 'chair']
dataset_manager = DataManager(dataset_name, split, class_names)
ground_truth_data = dataset_manager.load_data()
class_names = dataset_manager.class_names
print('Found:', len(ground_truth_data), 'images')
print('Class names: \n', class_names)


# prior boxes
# ------------------------------------------------------------------
# model = SSD300()
prior_boxes = create_prior_boxes()
prior_boxes = to_point_form(prior_boxes)
print('Prior boxes shape:', prior_boxes.shape)
print('Prior box example:', prior_boxes[777])


image_path = '../images/fish-bike.jpg'
# input_shape = model.input_shape[1:3]
input_shape = (300, 300)
image_array = load_image(image_path, input_shape)
box_coordinates = prior_boxes[7010:7015, :]
plot_box_data(box_coordinates, image_array)
plt.imshow(image_array)
plt.show()


# ground truth
# ------------------------------------------------------------------
image_name, box_data = random.sample(ground_truth_data.items(), 1)[0]
print('Data sample: \n', box_data)
# image_path = dataset_manager.images_path + image_name
image_path = image_name
arg_to_class = dataset_manager.arg_to_class
colors = get_colors(len(class_names))
image_array = load_image(image_path, input_shape)
plot_box_data(box_data, image_array, arg_to_class, colors=colors)
plt.imshow(image_array)
plt.show()

# assigned boxes
assigned_boxes = assign_prior_boxes(prior_boxes, box_data, len(class_names),
                                    regress=False, overlap_threshold=.5)
positive_mask = assigned_boxes[:, 4] != 1
positive_boxes = assigned_boxes[positive_mask]
image_array = load_image(image_path, input_shape)
plot_box_data(positive_boxes, image_array, arg_to_class, colors=colors)
plt.imshow(image_array)
plt.show()

# regressed boxes
assigned_regressed_boxes = assign_prior_boxes(
        prior_boxes, box_data, len(class_names),
        regress=True, overlap_threshold=.5)
positive_mask = assigned_regressed_boxes[:, 4] != 1
regressed_positive_boxes = assigned_regressed_boxes[positive_mask]
image_array = load_image(image_path, input_shape)
plot_box_data(regressed_positive_boxes, image_array,
              arg_to_class, colors=colors)
plt.imshow(image_array)
plt.show()


# un-regressed boxes
assigned_unregressed_boxes = unregress_boxes(assigned_regressed_boxes,
                                             prior_boxes)
unregressed_positive_boxes = assigned_unregressed_boxes[positive_mask]
image_array = load_image(image_path, input_shape)
plot_box_data(unregressed_positive_boxes, image_array,
              arg_to_class, colors=colors)
plt.imshow(image_array)
plt.show()


# data augmentations
# ------------------------------------------------------------------
data_manager = DataManager(dataset_name, 'train')
class_names = data_manager.class_names
train_data = data_manager.load_data()
arg_to_class = data_manager.arg_to_class
colors = get_colors(25)
val_data = DataManager(dataset_name, 'val').load_data()
"""
generator = ImageGenerator(train_data, val_data, prior_boxes,
                           batch_size=21)
generated_data = next(generator.flow('train'))
"""
generator = SequenceManager(train_data, 'train', prior_boxes)
generated_data = generator.__getitem__(10)

transformed_image_batch = generated_data[0]['input_1']
generated_output = generated_data[1]['predictions']
for batch_arg, transformed_image in enumerate(transformed_image_batch):
    positive_mask = generated_output[batch_arg, :, 4] != 1
    regressed_boxes = generated_output[batch_arg]
    unregressed_boxes = unregress_boxes(regressed_boxes, prior_boxes)
    unregressed_positive_boxes = unregressed_boxes[positive_mask]
    print('Regressed boxes shape:', regressed_boxes[positive_mask].shape)
    print('Unregressed boxes shape:', unregressed_boxes[positive_mask].shape)
    plot_box_data(unregressed_positive_boxes, transformed_image,
                  arg_to_class, colors=colors)
    plt.imshow(transformed_image.astype('uint8'))
    plt.show()
    """
    unregressed_assigned_boxes = assign_prior_boxes(
            prior_boxes, box_data, len(class_names),
            regress=False, overlap_threshold=.5)

    plot_box_data(unregressed_assigned_boxes, transformed_image,
                  arg_to_class, colors=colors)
    plt.imshow(transformed_image.astype('uint8'))
    plt.show()
    """
