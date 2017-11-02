from datasets import DataManager
from models import MultiboxLoss
from models import SSD300
from keras.optimizers import Adam
from utils.data_augmentation import ImageGenerator
from utils.boxes import create_prior_boxes

batch_size = 3
num_epochs = 2

dataset_manager = DataManager(['VOC2007', 'VOC2012'], ['trainval', 'trainval'])
train_data = dataset_manager.load_data()
val_data = test_data = DataManager('VOC2007', 'test').load_data()
class_names = dataset_manager.class_names
num_classes = len(class_names)

image_shape = (300, 300, 3)
weights_path = '../trained_models/SSD300_weights.hdf5'
frozen_layers = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
                 'conv2_1', 'conv2_2', 'pool2',
                 'conv3_1', 'conv3_2', 'conv3_3', 'pool3']

model = SSD300(image_shape, num_classes, weights_path, frozen_layers)
multibox_loss = MultiboxLoss(num_classes, neg_pos_ratio=2.0).compute_loss
model.compile(Adam(lr=3e-4), loss=multibox_loss, metrics=['acc'])


prior_boxes = create_prior_boxes()
generator = ImageGenerator(train_data, val_data, prior_boxes, batch_size)

model.fit_generator(generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_data) / batch_size),
                    epochs=num_epochs, verbose=1,
                    validation_data=generator.flow(mode='val'),
                    validation_steps=int(len(val_data) / batch_size))
