import os
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

from datasets import DataManager
from models.experimental_loss import MultiboxLoss
from models import SSD300
from keras.optimizers import SGD
# from utils.generator import ImageGenerator
from utils.sequencer_manager import SequenceManager
from utils.boxes import create_prior_boxes
from utils.boxes import to_point_form
from utils.training_utils import LearningRateManager

# hyper-parameters
batch_size = 3
num_epochs = 233
image_shape = (300, 300, 3)
box_scale_factors = [.1, .1, .2, .2]
negative_positive_ratio = 3
learning_rate = 1e-4
weight_decay = 5e-4
momentum = .9
optimizer = SGD(learning_rate, momentum, decay=weight_decay)
# optimizer = 'adam'
decay = 0.1
step_epochs = [154, 193, 232]
randomize_top = True
weights_path = '../trained_models/VGG16_weights.hdf5'
train_datasets = ['VOC2007', 'VOC2012']
train_splits = ['trainval', 'trainval']
val_dataset = 'VOC2007'
val_split = 'test'
class_names = 'all'
difficult_boxes = True
model_path = '../trained_models/SSD_SGD_scratch_all2/'
save_path = model_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'

train_data_manager = DataManager(train_datasets, train_splits,
                                 class_names, difficult_boxes)
train_data = train_data_manager.load_data()
class_names = train_data_manager.class_names
num_classes = len(class_names)
val_data_manager = DataManager(val_dataset, val_split, class_names, False)
val_data = val_data_manager.load_data()

# generator
prior_boxes = to_point_form(create_prior_boxes())
train_sequencer = SequenceManager(train_data, 'train', prior_boxes,
                                  batch_size, box_scale_factors, num_classes)

val_sequencer = SequenceManager(val_data, 'val', prior_boxes,
                                batch_size, box_scale_factors, num_classes)

# model
multibox_loss = MultiboxLoss(num_classes, negative_positive_ratio, batch_size)
model = SSD300(image_shape, num_classes, weights_path)
model.compile(optimizer, loss=multibox_loss.compute_loss)

# callbacks
if not os.path.exists(model_path):
    os.makedirs(model_path)

checkpoint = ModelCheckpoint(save_path, verbose=1, period=1)
log = CSVLogger(model_path + 'SSD_scratch.log')
learning_rate_manager = LearningRateManager(learning_rate, decay, step_epochs)
learning_rate_schedule = LearningRateScheduler(learning_rate_manager.schedule)
callbacks = [checkpoint, log, learning_rate_schedule]

# model fit
model.fit_generator(train_sequencer,
                    steps_per_epoch=int(len(train_data) / batch_size),
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=val_sequencer,
                    validation_steps=int(len(val_data) / batch_size),
                    use_multiprocessing=False,
                    max_queue_size=7,
                    workers=5)
