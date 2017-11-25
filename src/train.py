import os
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger

from datasets import DataManager
from models.experimental_loss import MultiboxLoss
from models import SSD300
from keras.optimizers import Adam
# from keras.optimizers import SGD
from utils.generator import ImageGenerator
from utils.boxes import create_prior_boxes
from utils.boxes import to_point_form
# from utils.training_utils import scheduler
from utils.training_utils import LearningRateManager

# hyper-parameters
batch_size = 5
num_epochs = 150
image_shape = (300, 300, 3)
box_scale_factors = [.1, .1, .2, .2]
negative_positive_ratio = 3
learning_rate = 3e-3
weight_decay = 5e-4
optimizer = Adam(learning_rate, decay=weight_decay)
gamma_decay = 0.98
randomize_top = True
weights_path = '../trained_models/SSD300_weights.hdf5'
datasets = ['VOC2007', 'VOC2012']
splits = ['trainval', 'trainval']
class_names = 'all'
difficult_boxes = True
model_path = '../trained_models/SSD_scratch_all/'
save_path = model_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
frozen_layers = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
                 'conv2_1', 'conv2_2', 'pool2',
                 'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
                 'conv4_1', 'conv4_2', 'conv4_3', 'pool4',
                 'conv5_1', 'conv5_2', 'conv5_3', 'pool5',
                 'fc6', 'fc7']


dataset_manager = DataManager(datasets, splits, class_names, difficult_boxes)
train_data = dataset_manager.load_data()
val_data = test_data = DataManager('VOC2007', 'test').load_data()
class_names = dataset_manager.class_names
num_classes = len(class_names)

# generator
prior_boxes = to_point_form(create_prior_boxes())
generator = ImageGenerator(train_data, val_data, prior_boxes, batch_size,
                           box_scale_factors, num_classes)

# model
multibox_loss = MultiboxLoss(num_classes, negative_positive_ratio, batch_size)
model = SSD300(image_shape, num_classes, weights_path,
               frozen_layers, randomize_top)
model.compile(optimizer, loss=multibox_loss.compute_loss)

# callbacks
if not os.path.exists(model_path):
    os.makedirs(model_path)

checkpoint = ModelCheckpoint(save_path, verbose=1, period=1)
log = CSVLogger(model_path + 'SSD_scratch.log')
learning_rate_manager = LearningRateManager(gamma_decay, learning_rate)
learning_rate_schedule = LearningRateScheduler(learning_rate_manager.schedule)
plateau = ReduceLROnPlateau('val_loss', factor=0.9, patience=1, verbose=1)
early_stop = EarlyStopping('val_loss', min_delta=1e-4, patience=14, verbose=1)
callbacks = [checkpoint, log, learning_rate_schedule]

# model fit
model.fit_generator(generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_data) / batch_size),
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=generator.flow(mode='val'),
                    validation_steps=int(len(val_data) / batch_size),
                    # use_multiprocessing=True,
                    max_queue_size=7,
                    workers=5)
