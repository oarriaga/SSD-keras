# In[1]:

from image_generator import ImageGenerator
from utils import create_prior_box, get_prior_parameters
from XML_parser import XMLParser
from models import SSD300
from multibox_loss import MultiboxLoss
from bounding_boxes_manager import BoundingBoxManager
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
#import numpy as np


# In[6]:

# constants
image_shape = (300, 300, 3)
num_classes = 21
batch_size = 5
variances = [0.1, 0.1, 0.2, 0.2]
training_data_ratio = .8
data_path = '../datasets/VOCdevkit/VOC2007/'

model = SSD300(image_shape, num_classes=num_classes)
model.load_weights('../weights/weights_SSD300.hdf5', by_name=True)
freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']

box_configurations = get_prior_parameters(model)

priors = create_prior_box(image_shape[0:2], box_configurations, variances)
bounding_box_manager = BoundingBoxManager(num_classes, priors)
ground_truth_data = XMLParser(data_path+'Annotations/').data


# In[11]:

keys = sorted(ground_truth_data.keys())
num_train = int(round(training_data_ratio * len(keys)))
train_keys = keys[:num_train]
validation_keys = keys[num_train:]
num_val = len(validation_keys)


# In[12]:

image_generator = ImageGenerator(ground_truth_data, bounding_box_manager, batch_size,
                    data_path+'JPEGImages/',train_keys, validation_keys, image_shape[:2])


# In[13]:

# In[14]:

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
                              callbacks=callbacks,
                              validation_data=image_generator.flow(False),
                              nb_val_samples=num_val,
                              nb_worker=1)

