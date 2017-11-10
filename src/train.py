import os
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger

from datasets import DataManager
from models import MultiboxLoss
from models import SSD300
# from keras.optimizers import Adam
from keras.optimizers import SGD
from utils.generator import ImageGenerator
from utils.boxes import create_prior_boxes
from utils.boxes import to_point_form
from utils.training_utils import Scheduler

batch_size = 5
num_epochs = 10000
image_shape = (300, 300, 3)
box_scale_factors = [.1, .1, .2, .2]
negative_positive_ratio = 3
scheduled_epochs = [80, 100, 120]

dataset_manager = DataManager(['VOC2007', 'VOC2012'], ['trainval', 'trainval'])
train_data = dataset_manager.load_data()
val_data = test_data = DataManager('VOC2007', 'test').load_data()
class_names = dataset_manager.class_names
num_classes = len(class_names)


prior_boxes = to_point_form(create_prior_boxes())
generator = ImageGenerator(train_data, val_data, prior_boxes, batch_size,
                           box_scale_factors, num_classes)

weights_path = '../trained_models/SSD300_weights.hdf5'
frozen_layers = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
                 'conv2_1', 'conv2_2', 'pool2',
                 'conv3_1', 'conv3_2', 'conv3_3', 'pool3']

model = SSD300(image_shape, num_classes, weights_path, frozen_layers, True)

base_lr = 3e-3
sgd = SGD(base_lr, momentum=.9, decay=5e-4)

multibox_loss = MultiboxLoss(
        num_classes, neg_pos_ratio=negative_positive_ratio).compute_loss

model.compile(sgd, loss=multibox_loss)


# callbacks
model_path = '../trained_models/SSD_scratch/'

if not os.path.exists(model_path):
    os.makedirs(model_path)

save_path = model_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'

checkpoint = ModelCheckpoint(save_path, verbose=1,
                             save_best_only=False, period=5)

log = CSVLogger(model_path + 'SSD_scratch.log')
scheduler = Scheduler(scheduled_epochs, decay=0.1, base_learning_rate=base_lr)
learning_rate_schedule = LearningRateScheduler(scheduler.schedule)

# plateau = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
# early_stop = EarlyStopping(min_delta=1e-3, patience=25, verbose=1)
# callbacks = [checkpoint, plateau, early_stop, log]
callbacks = [checkpoint, learning_rate_schedule, log]

model.fit_generator(generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_data) / batch_size),
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=generator.flow(mode='val'),
                    validation_steps=int(len(val_data) / batch_size),
                    use_multiprocessing=True,
                    max_queue_size=5,
                    workers=3)
