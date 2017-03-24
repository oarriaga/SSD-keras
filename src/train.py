from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

from image_generator import ImageGenerator
#from training_beta import MultiboxLoss
from multibox_loss import MultiboxLoss
#from ssd_training import MultiboxLoss
from ssd import SSD300
from utils.prior_box_creator import PriorBoxCreator
from utils.prior_box_manager import PriorBoxManager
from utils.XML_parser import XMLParser
from utils.utils import split_data
from utils.utils import scheduler
from utils.utils import flatten_prior_boxes
from utils.utils import add_variances

from utils.ssd_utils import BBoxUtility

batch_size = 7
num_epochs = 15
num_classes = 21
model = SSD300()
model.load_weights('../trained_models/weights_SSD300.hdf5', by_name=True)
freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']

for layer in model.layers:
    if layer.name in freeze:
        layer.trainable = False

image_shape = model.input_shape[1:]
box_creator = PriorBoxCreator(model)
prior_boxes = box_creator.create_boxes()

root_prefix = '../datasets/VOCdevkit/VOC2007/'
ground_data_prefix = root_prefix + 'Annotations/'
image_prefix = root_prefix + 'JPEGImages/'


ground_truth_manager = XMLParser(ground_data_prefix, background_id=None)
ground_truth_data = ground_truth_manager.get_data()

train_keys, validation_keys = split_data(ground_truth_data, training_ratio=.8)

prior_box_manager = PriorBoxManager(prior_boxes,
                                    box_scale_factors=[.1, .1, .2, .2])

prior_boxes = flatten_prior_boxes(prior_boxes)
prior_boxes = add_variances(prior_boxes)
bbox_util = BBoxUtility(num_classes, prior_boxes)


image_generator = ImageGenerator(ground_truth_data,
                                 #prior_box_manager,
                                 bbox_util,
                                 batch_size,
                                 image_shape[0:2],
                                 train_keys, validation_keys,
                                 image_prefix,
                                 vertical_flip_probability=0,
                                 horizontal_flip_probability=0.5)
multibox_loss = MultiboxLoss(num_classes, neg_pos_ratio=3.0).compute_loss
model.compile(optimizer=Adam(lr=3e-4), loss=multibox_loss, metrics=['acc'])
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
                    callbacks=[model_checkpoint],
                    validation_data=image_generator.flow(mode='val'),
                    nb_val_samples = len(validation_keys))
