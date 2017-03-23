
# coding: utf-8

# In[1]:

#import cv2
import keras
import pickle

from ssd import SSD300
from my_multibox_loss_2 import MultiboxLoss
#from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))


# In[2]:

# some constants
NUM_CLASSES = 21
input_shape = (300, 300, 3)


# In[3]:

priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)


# In[4]:

#gt = pickle.load(open('gt_pascal.pkl', 'rb'))
from PASCAL_VOC.get_data_from_XML import XML_preprocessor
data_path = '../datasets/VOCdevkit/VOC2007/Annotations/'
gt = XML_preprocessor(data_path).data
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)


# In[5]:
from image_generator import ImageGenerator
path_prefix = '../datasets/VOCdevkit/VOC2007/JPEGImages/'
gen = ImageGenerator(gt, bbox_util, 7, (input_shape[0], input_shape[1]), train_keys, val_keys, path_prefix, vertical_flip_probability=0)

# In[6]:

# In[7]:

model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('../trained_models/weights_SSD300.hdf5', by_name=True)


# In[8]:

freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']#,
#           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

for L in model.layers:
    if L.name in freeze:
        L.trainable = False


# In[9]:

def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

callbacks = [keras.callbacks.ModelCheckpoint('../trained_models/model_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
             keras.callbacks.LearningRateScheduler(schedule)]


# In[10]:

base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)
# optim = keras.optimizers.RMSprop(lr=base_lr)
# optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)


# In[11]:

nb_epoch = 30
history = model.fit_generator(gen.flow(mode='train'), len(train_keys),
                              nb_epoch, verbose=1,
                              callbacks=callbacks,
                              validation_data=gen.flow(mode='val'),
                              nb_val_samples=len(val_keys),
                              nb_worker=1)

