from datasets.data_manager import DataManager
from utils.boxes import create_prior_boxes
from utils.boxes import to_point_form
from utils.generator import ImageGenerator
from models.experimental_loss import MultiboxLoss
import keras.backend as K

# parameters
dataset_name = 'VOC2012'
batch_size = 5
num_classes = 21
negative_positive_ratio = 3
prior_boxes = to_point_form(create_prior_boxes())

# loading training data
data_manager = DataManager(dataset_name, 'train')
train_data = data_manager.load_data()
arg_to_class = data_manager.arg_to_class
# loading validation data
val_data = DataManager(dataset_name, 'val').load_data()

# generating output
data = ImageGenerator(train_data, val_data, prior_boxes, batch_size)
generator = data.flow('train')
output_1 = next(generator)[1]['predictions']
output_2 = next(generator)[1]['predictions']

multibox_loss = MultiboxLoss(num_classes, negative_positive_ratio, batch_size)

output_1 = K.variable(output_1)
output_2 = K.variable(output_2)
loss = multibox_loss.compute_loss(output_1, output_2)
session = K.get_session()
loss = loss.eval(session=session)
print(loss)
