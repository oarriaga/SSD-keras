import os

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from datasets import DataManager
from models import SSD300
from utils.boxes import create_prior_boxes, to_point_form
from models.multibox_loss import MultiboxLoss
from datasets import get_class_names
from utils.data_generator import DataGenerator

model_name = 'SSD300_VOC2007'
weights_path = '../trained_models/SSD300_weights.hdf5'

# hyper-parameters
batch_size = 5
num_epochs = 250
base_learning_rate = 1e-3
negative_positive_ratio = 3

# data
class_names = get_class_names('VOC2007')
val_dataset, val_split = 'VOC2007', 'test'
train_datasets, train_splits = ['VOC2007', 'VOC2012'], ['trainval', 'trainval']
train_data_manager = DataManager(train_datasets, train_splits, class_names)
train_data = train_data_manager.load_data()
num_classes = len(class_names)
val_data_manager = DataManager(val_dataset, val_split, class_names, False)
val_data = val_data_manager.load_data()

# model
model = SSD300(num_classes=num_classes, weights_path=weights_path)
prior_boxes = to_point_form(create_prior_boxes())
multibox_loss = MultiboxLoss(num_classes, negative_positive_ratio)
optimizer = Adam(base_learning_rate)
model.compile(optimizer, loss=multibox_loss.compute_loss)
data_generator = DataGenerator(train_data, prior_boxes, batch_size,
                               num_classes, val_data)

# callbacks
model_path = '../trained_models/' + model_name + '/'
save_path = model_path + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
if not os.path.exists(model_path):
    os.makedirs(model_path)
early_stop = EarlyStopping(patience=7)
log = CSVLogger(model_path + model_name + '.log')
checkpoint = ModelCheckpoint(save_path, verbose=1, period=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=2, verbose=1)
callbacks = [checkpoint, log, reduce_lr, early_stop]

# model fit
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
