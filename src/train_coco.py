from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

from image_generator import ImageGenerator
from multibox_loss import MultiboxLoss
from models import SSD300
from data_loader import DataLoader
from utils.prior_box_creator import PriorBoxCreator
from utils.prior_box_manager import PriorBoxManager
from utils.utils import split_data
from utils.utils import scheduler

# constants
dataset_name = 'COCO'
batch_size = 1
num_epochs = 1000
training_ratio = .8
image_path_prefix = '../datasets/COCO/images/train2014/'
image_shape = (300, 300 ,3)

# loading dataset
ground_truth_manager = DataLoader(dataset_name)
num_classes = ground_truth_manager.parser.num_classes
ground_truth_data = ground_truth_manager.get_data()
train_keys, validation_keys = split_data(ground_truth_data,
                                        training_ratio)

# model parameters
model = SSD300(image_shape, num_classes)
model.load_weights('../trained_models/weights_SSD300.hdf5', by_name=True)
freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']
for layer in model.layers:
    if layer.name in freeze:
        layer.trainable = False
multibox_loss = MultiboxLoss(num_classes, neg_pos_ratio=2.0).compute_loss
model.compile(optimizer=Adam(lr=3e-4), loss=multibox_loss)

# creating prior/default boxes 
box_creator = PriorBoxCreator(model)
prior_boxes = box_creator.create_boxes()

# prior box assigner
prior_box_manager = PriorBoxManager(prior_boxes, num_classes=num_classes,
                                    box_scale_factors=[.1, .1, .2, .2])

image_generator = ImageGenerator(ground_truth_data,
                                 prior_box_manager,
                                 batch_size,
                                 image_shape[0:2],
                                 train_keys, validation_keys,
                                 image_path_prefix,
                                 vertical_flip_probability=0.5,
                                 horizontal_flip_probability=0.5)


model_names = ('../trained_models/model_checkpoints/COCO/' +
               'SSD300_COCO_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False)
learning_rate_schedule = LearningRateScheduler(scheduler)

model.fit_generator(image_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_keys) / batch_size),
                    epochs = num_epochs, verbose = 1,
                    callbacks=[model_checkpoint, learning_rate_schedule],
                    validation_data=image_generator.flow(mode='val'),
                    validation_steps=int(len(validation_keys) / batch_size))
