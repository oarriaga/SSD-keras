from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

from image_generator import ImageGenerator
from multibox_loss import MultiboxLoss
from models import SSD300
from utils.prior_box_creator import PriorBoxCreator
#from utils.prior_box_creator_paper import PriorBoxCreator
from utils.prior_box_manager import PriorBoxManager
from utils.XML_parser import XMLParser
from utils.utils import split_data
from utils.utils import scheduler

# constants
batch_size = 5
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

box_creator = PriorBoxCreator(model)
model_configurations = box_creator.model_configurations
prior_boxes = box_creator.create_boxes()

