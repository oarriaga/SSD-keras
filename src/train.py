from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

from utils.data_augmentation import ImageGenerator
from utils.train import MultiboxLoss
from utils.train import scheduler
from utils.train import split_data
from models.ssd import SSD300
from utils.boxes import create_prior_boxes
from utils.datasets import DataManager

# parameters
batch_size = 5
num_epochs = 15
num_classes = 21
optimizer = Adam(lr=3e-4)
root_prefix = '../datasets/VOCdevkit/VOC2007/'
ground_data_prefix = root_prefix + 'Annotations/'
image_prefix = root_prefix + 'JPEGImages/'
image_shape = (300, 300 ,3)
dataset_name = 'VOC2007'
weights_path = '../trained_models/weights_SSD300.hdf5'
trained_models_path = '../trained_models/model_checkpoints/'
trained_models_filename = (trained_models_path +
                        'ssd300_weights.{epoch:03d}-{val_loss:.3f}.hdf5')
frozen_layers = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
                'conv2_1', 'conv2_2', 'pool2',
                'conv3_1', 'conv3_2', 'conv3_3', 'pool3']
box_scale_factors = [.1, .1, .2, .2]

# loading and splitting data
data_manager = DataManager(dataset_name)
ground_truth_data = data_manager.get_data()
train_keys, validation_keys = split_data(ground_truth_data, training_ratio=.8)

# instantiating model
model = SSD300(image_shape, num_classes, weights_path, frozen_layers)
multibox_loss = MultiboxLoss(num_classes, neg_pos_ratio=2.0).compute_loss
model.compile(optimizer, loss=multibox_loss, metrics=['acc'])

# setting parameters for data augmentation generator
prior_boxes = create_prior_boxes(model)
image_generator = ImageGenerator(ground_truth_data,
                                 prior_boxes,
                                 num_classes,
                                 box_scale_factors,
                                 batch_size,
                                 image_shape[0:2],
                                 train_keys, validation_keys,
                                 image_prefix,
                                 vertical_flip_probability=0.5,
                                 horizontal_flip_probability=0.5)

# instantiating callbacks
learning_rate_schedule = LearningRateScheduler(scheduler)
model_names = (trained_models_filename)
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False)

# training model with real-time data augmentation
model.fit_generator(image_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_keys) / batch_size),
                    epochs = num_epochs, verbose = 1,
                    callbacks=[model_checkpoint, learning_rate_schedule],
                    validation_data=image_generator.flow(mode='val'),
                    validation_steps=int(len(validation_keys) / batch_size))
