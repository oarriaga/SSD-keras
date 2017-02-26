# In[1]:

from image_generator import ImageGenerator
from utils import create_prior_box
from XML_preprocessor import XML_preprocessor
from models import SSD300
from multibox_loss import MultiboxLoss
from bounding_boxes_utility import BoundingBoxUtility
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
#import numpy as np


# In[6]:

# constants
image_shape = (300, 300, 3)
num_classes = 21
batch_size = 16
variances = [0.1, 0.1, 0.2, 0.2]
training_data_ratio = .8
data_path = '../datasets/VOCdevkit/VOC2007/'

box_configs = [
    {'layer_width': 38, 'layer_height': 38, 'num_prior': 3, 'min_size': 30.0,
     'max_size': None, 'aspect_ratios': [1.0, 2.0, 1/2.0]},
    {'layer_width': 19, 'layer_height': 19, 'num_prior': 6, 'min_size': 60.0,
     'max_size': 114.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
    {'layer_width': 10, 'layer_height': 10, 'num_prior': 6, 'min_size': 114.0,
     'max_size': 168.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
    {'layer_width':  5, 'layer_height':  5, 'num_prior': 6, 'min_size': 168.0,
     'max_size': 222.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
    {'layer_width':  3, 'layer_height':  3, 'num_prior': 6, 'min_size': 222.0,
     'max_size': 276.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
    {'layer_width':  1, 'layer_height':  1, 'num_prior': 6, 'min_size': 276.0,
     'max_size': 330.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
]


# In[7]:

priors = create_prior_box(image_shape[0:2], box_configs, variances)
bounding_box_utils = BoundingBoxUtility(num_classes, priors)
ground_truth_data = XML_preprocessor(data_path+'Annotations/').data


# In[11]:

keys = sorted(ground_truth_data.keys())
num_train = int(round(training_data_ratio * len(keys)))
train_keys = keys[:num_train]
validation_keys = keys[num_train:]
num_val = len(validation_keys)


# In[12]:

image_generator = ImageGenerator(ground_truth_data, bounding_box_utils, batch_size,
                    data_path+'JPEGImages/',train_keys, validation_keys, image_shape[:2])


# In[13]:

model = SSD300(image_shape, num_classes=num_classes)
model.load_weights('../weights/weights_SSD300.hdf5', by_name=True)


# In[14]:

freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']

for layer in model.layers:
    if layer.name in freeze:
        layer.trainable = False


# In[15]:

models_path = '../weights/model_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(models_path, verbose=1, save_weights_only=True)
base_learning_rate = 3e-4
def schedule(epoch, decay=0.9):
    return base_learning_rate * decay**(epoch)
learning_rate_scheduler =  LearningRateScheduler(schedule)
callbacks = [model_checkpoint, learning_rate_scheduler]


# In[16]:

multibox_loss = MultiboxLoss(num_classes, neg_pos_ratio=2.0).compute_loss


# In[17]:

optimizer = Adam(lr=base_learning_rate)
model.compile(optimizer=optimizer,
              loss=multibox_loss)


# In[18]:

nb_epoch = 30
history = model.fit_generator(image_generator.flow(True), num_train,
                              nb_epoch, verbose=1,
                              nb_worker=1)
                              #validation_data=image_generator.flow(False),
                              #nb_val_samples=num_val,
                              #nb_worker=1)

