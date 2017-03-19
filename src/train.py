from keras.callbacks import ModelCheckpoint

from image_generator import ImageGenerator
from models import my_SSD
from multibox_loss import MultiboxLoss
from utils.prior_box_creator import PriorBoxCreator
from utils.prior_box_manager import PriorBoxManager
from utils.box_visualizer import BoxVisualizer
from utils.XML_parser import XMLParser
from utils.utils import split_data

batch_size = 10
num_epochs = 15
num_classes = 21
model = my_SSD(num_classes)
image_shape = model.input_shape[1:]
box_creator = PriorBoxCreator(model)
prior_boxes = box_creator.create_boxes()

root_prefix = '../datasets/VOCdevkit/VOC2007/'
image_prefix = root_prefix + 'JPEGImages/'
box_visualizer = BoxVisualizer(image_prefix, image_shape[0:2])

ground_data_prefix = root_prefix + 'Annotations/'
ground_truth_manager = XMLParser(ground_data_prefix, background_id=0)
ground_truth_data = ground_truth_manager.get_data()

train_keys, validation_keys = split_data(ground_truth_data, training_ratio=.8)

prior_box_manager = PriorBoxManager(prior_boxes)
image_generator = ImageGenerator(ground_truth_data,
                                 prior_box_manager,
                                 batch_size,
                                 image_shape[0:2],
                                 train_keys, validation_keys,
                                 image_prefix,
                                 vertical_flip_probability=0,
                                 horizontal_flip_probability=0.5)

multibox_loss = MultiboxLoss(num_classes, neg_pos_ratio=2.0).compute_loss
model.compile(optimizer='adam', loss=multibox_loss, metrics=['acc'])

model_names = ('../trained_models/model_checkpoints/' +
               'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=True)

model.fit_generator(image_generator.flow(mode='train'),
                    len(train_keys),
                    num_epochs,
                    callbacks=[model_checkpoint],
                    validation_data=image_generator.flow(mode='val'),
                    nb_val_samples = len(validation_keys))
