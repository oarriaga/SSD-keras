import os

from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD

from models import SSD300
from datasets import DataManager
from datasets import get_class_names
from models.multibox_loss import MultiboxLoss
from utils.data_generator import DataGenerator
from utils.training_utils import LearningRateManager
from utils.boxes import create_prior_boxes, to_point_form

model_name = 'SSD300_VOC2007'

# hyper-parameters
batch_size = 5
num_epochs = 250
alpha_loss = 1.0
learning_rate = 1e-3
momentum = .9
weight_decay = 5e-4
gamma_decay = 0.1
scheduled_epochs = [155, 195, 235]
negative_positive_ratio = 3
base_weights_path = '../trained_models/VGG16_weights.hdf5'

# data management
class_names = get_class_names('VOC2007')
val_dataset, val_split = 'VOC2007', 'test'
train_datasets, train_splits = ['VOC2007', 'VOC2012'], ['trainval', 'trainval']
train_data_manager = DataManager(train_datasets, train_splits, class_names)
train_data = train_data_manager.load_data()
num_classes = len(class_names)
val_data_manager = DataManager(val_dataset, val_split, class_names)
val_data = val_data_manager.load_data()

# model
model = SSD300(num_classes=num_classes)
model.load_weights(base_weights_path, by_name=True)
prior_boxes = to_point_form(create_prior_boxes())
multibox_loss = MultiboxLoss(num_classes, negative_positive_ratio, alpha_loss)
optimizer = SGD(learning_rate, momentum, weight_decay)
model.compile(optimizer, loss=multibox_loss.compute_loss)
data_generator = DataGenerator(train_data, prior_boxes, batch_size,
                               num_classes, val_data)

# callbacks
model_path = '../trained_models/' + model_name + '/'
save_path = model_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
if not os.path.exists(model_path):
    os.makedirs(model_path)
log = CSVLogger(model_path + model_name + '.log')
checkpoint = ModelCheckpoint(save_path, verbose=1, period=1)
scheduler = LearningRateManager(learning_rate, gamma_decay, scheduled_epochs)
schedule_learning_rate = LearningRateScheduler(scheduler.schedule, verbose=1)
callbacks = [checkpoint, log, schedule_learning_rate]

model.summary()
model.fit_generator(data_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_data) / batch_size),
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=data_generator.flow(mode='val'),
                    validation_steps=int(len(val_data) / batch_size),
                    use_multiprocessing=False,
                    workers=1)
