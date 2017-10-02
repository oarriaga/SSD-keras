import matplotlib.pyplot as plt
from datasets import DataManager
from models.ssd import SSD300
from utils.boxes import create_prior_boxes
from utils.inference import plot_box_data
from utils.preprocessing import load_image


split = 'train'
dataset_name = 'VOC2007'

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


model = SSD300()
prior_boxes = create_prior_boxes()
print('Prior boxes shape:', prior_boxes.shape)
print('Prior box example:', prior_boxes[777])


image_path = '../images/fish-bike.jpg'
input_shape = model.input_shape[1:3]
image_array = load_image(image_path, input_shape)[0]
box_coordinates = prior_boxes[0:10, :]
plot_box_data(box_coordinates, image_array)

"""
# In[ ]:

#box_coordinates = prior_boxes[5400:6900, :]
#draw_image_boxes(box_coordinates, image_array)


# ## Ground truths

# In[17]:

image_name = sorted(list(ground_truth_data.keys()))[1019]
box_data = ground_truth_data[image_name]
print('Data sample: \n', box_data)
image_prefix = dataset_manager.image_prefix
image_path = image_prefix + image_name
class_decoder = dataset_manager.arg_to_class
image_array = load_image(image_path, input_shape)
draw_image_boxes(box_data, image_array, class_decoder)


# ## Assigned prior boxes without regression

# In[18]:

from utils.boxes import assign_prior_boxes


# In[19]:

#prior_box_manager = PriorBoxManager(prior_boxes, num_classes=num_classes)
num_classes = len(class_names)
box_scale_factors = [.1, .1, .2, .2]
assigned_boxes = assign_prior_boxes(prior_boxes, box_data,
                                    num_classes, box_scale_factors,
                                    regress=False,
                                    overlap_threshold=.6)
positive_mask = assigned_boxes[:, 4] != 1
positive_boxes = assigned_boxes[positive_mask]
draw_image_boxes(positive_boxes, image_array, class_decoder)


# ## Assigned prior boxes with regression

# In[ ]:

"""
assigned_regressed_boxes = assign_prior_boxes(prior_boxes, box_data,
                                              num_classes, box_scale_factors,
                                              regress=True)
positive_mask = assigned_regressed_boxes[:, 4] != 1
encoded_positive_boxes = assigned_regressed_boxes[positive_mask]
draw_image_boxes(assigned_regressed_boxes, image_array, class_decoder)
"""


# ## Assigned decoded boxes

# In[20]:

from utils.boxes import decode_boxes


# In[ ]:

"""
assigned_decoded_boxes = decode_boxes(assigned_regressed_boxes,
                                      prior_boxes, box_scale_factors)
decoded_positive_boxes = assigned_decoded_boxes[positive_mask]
draw_image_boxes(decoded_positive_boxes, image_array, class_decoder)
"""


# ## Data augmentation

# In[21]:

from utils.data_augmentation import ImageGenerator
from utils.preprocessing import load_image
from utils.visualizer import plot_images
from utils.train import split_data
import numpy as np


# In[22]:

batch_size = 1
train_keys, validation_keys = split_data(ground_truth_data, training_ratio=.8)
image_generator = ImageGenerator(ground_truth_data,
                                 prior_boxes,
                                 num_classes,
                                 box_scale_factors,
                                 batch_size,
                                 input_shape,
                                 train_keys, validation_keys,
                                 image_prefix,
                                 vertical_flip_probability=0,
                                 horizontal_flip_probability=1)

generated_data = next(image_generator.flow(mode='demo'))
generated_input = generated_data[0]['input_1']
generated_output = generated_data[1]['predictions']
generated_image = np.squeeze(generated_input[0]).astype('uint8')
validation_image_name = image_prefix + validation_keys[0]
original_image = load_image(validation_image_name, input_shape)
plot_images(original_image, generated_image)


# ## Modified ground-truth

# In[23]:

generated_encoded_boxes = np.squeeze(generated_output)
generated_boxes = decode_boxes(generated_encoded_boxes, prior_boxes,
                                                  box_scale_factors)
positive_mask = generated_boxes[:, 4] != 1
generated_positive_boxes = generated_boxes[positive_mask]
draw_image_boxes(generated_positive_boxes, generated_image, class_decoder)


# ## Detecting objects

# In[24]:

import glob
from utils.inference import predict
from utils.preprocessing import get_image_size
from utils.preprocessing import load_pil_image
from scipy.misc import imread
from utils.preprocessing import image_to_array
from utils.datasets import get_class_names
from utils.datasets import get_arg_to_class


# In[25]:

weights_path = '../trained_models/SSD300_weights.hdf5'
image_paths = glob.glob('../images/*.jpg')
class_names = get_class_names(dataset_name)
class_decoder = get_arg_to_class(class_names)
num_classes = len(class_names)
probability_threshold = .6
model = SSD300(num_classes=21, weights_path=weights_path)
for image_path in image_paths:
    image_array = load_image(image_path, input_shape)
    original_image_array = image_to_array(load_pil_image(image_path)) 
    original_image_size = get_image_size(image_path)
    selected_boxes = predict(model, image_array, prior_boxes, original_image_size,
                             num_classes, probability_threshold, .5)
    draw_image_boxes(selected_boxes, original_image_array, class_decoder,
                                                         normalized=False)

"""
