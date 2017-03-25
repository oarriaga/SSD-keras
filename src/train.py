import pickle

from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

from image_generator import ImageGenerator
from multibox_loss import MultiboxLoss
from models import SSD300
#from utils.prior_box_creator import PriorBoxCreator
from utils.prior_box_manager import PriorBoxManager
from utils.XML_parser import XMLParser
from utils.utils import split_data
from utils.utils import scheduler

# constants
batch_size = 7
num_epochs = 15
num_classes = 21
root_prefix = '../datasets/VOCdevkit/VOC2007/'
ground_data_prefix = root_prefix + 'Annotations/'
image_prefix = root_prefix + 'JPEGImages/'
image_shape = (300, 300 ,3)
model = SSD300(image_shape, num_classes)

model.load_weights('../trained_models/weights_SSD300.hdf5', by_name=True)
freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']

for layer in model.layers:
    if layer.name in freeze:
        layer.trainable = False

multibox_loss = MultiboxLoss(num_classes, neg_pos_ratio=2.0).compute_loss
model.compile(optimizer=Adam(lr=3e-4), loss=multibox_loss, metrics=['acc'])

#box_creator = PriorBoxCreator(model)
#prior_boxes = box_creator.create_boxes()
prior_boxes = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))

ground_truth_manager = XMLParser(ground_data_prefix, background_id=None)
ground_truth_data = ground_truth_manager.get_data()

train_keys, validation_keys = split_data(ground_truth_data, training_ratio=.8)

prior_box_manager = PriorBoxManager(prior_boxes,
                                    box_scale_factors=[.1, .1, .2, .2])

image_generator = ImageGenerator(ground_truth_data,
                                 prior_box_manager,
                                 batch_size,
                                 image_shape[0:2],
                                 train_keys, validation_keys,
                                 image_prefix,
                                 vertical_flip_probability=0,
                                 horizontal_flip_probability=0.5)


model_names = ('../trained_models/model_checkpoints/' +
               'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=True)
learning_rate_schedule = LearningRateScheduler(scheduler)

model.fit_generator(image_generator.flow(mode='train'),
                    len(train_keys),
                    num_epochs,
                    callbacks=[model_checkpoint, learning_rate_schedule],
                    validation_data=image_generator.flow(mode='val'),
                    nb_val_samples = len(validation_keys))
