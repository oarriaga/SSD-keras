from image_generator import ImageGenerator
from models import SSD300
from utils.prior_box_creator import PriorBoxCreator
from utils.prior_box_assigner import PriorBoxAssigner
from utils.XML_parser import XMLParser
from utils.utils import split_data
from utils.utils import read_image, resize_image
import numpy as np
import matplotlib.pyplot as plt

image_shape = (300, 300, 3)
model = SSD300(image_shape)
box_creator = PriorBoxCreator(model)
prior_boxes = box_creator.create_boxes()

layer_scale, box_arg = 0, 780
box_coordinates = prior_boxes[layer_scale][box_arg,:,:]
image_path = '../images/'
image_key = '007040.jpg'
box_creator.draw_boxes(image_path + image_key, box_coordinates)

data_path = '../datasets/VOCdevkit/VOC2007/'
ground_truths = XMLParser(data_path+'Annotations/').get_data()
prior_box_manager = PriorBoxAssigner(prior_boxes, ground_truths)
assigned_boxes = prior_box_manager.assign_boxes()
prior_box_manager.draw_assigned_boxes(image_path, image_shape[0:2], image_key)

batch_size = 7
train_keys, validation_keys = split_data(assigned_boxes, training_ratio=.8)

image_generator = ImageGenerator(assigned_boxes, batch_size,
                                image_shape[0:2],
                                train_keys, validation_keys,
                                data_path+'JPEGImages/')

transformed_image = next(image_generator.flow(mode='demo'))[0]
transformed_image = np.squeeze(transformed_image[0]).astype('uint8')
original_image = read_image(data_path+'JPEGImages/'+validation_keys[0])
original_image = resize_image(original_image, image_shape[0:2])

plt.figure(1)
plt.subplot(121)
plt.title('Original image')
plt.imshow(original_image)
plt.subplot(122)
plt.title('Transformed image')
plt.imshow(transformed_image)
plt.show()



