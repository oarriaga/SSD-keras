import keras.backend as K
from utils.data_generator import DataGenerator
from datasets.data_manager import DataManager
from models.multibox_loss import MultiboxLoss
from utils.pytorch_multibox_loss import MultiBoxLoss as MultiboxLossTorch
from utils.boxes import create_prior_boxes, to_point_form
import torch
from torch.autograd import Variable
import numpy as np
from utils.pytorch_layers import PriorBox
from utils.pytorch_parameters import v2

# parameters
batch_size = 5
num_classes = 21
negative_positive_ratio = 3
prior_boxes = to_point_form(create_prior_boxes())

# loading training data
data_manager = DataManager('VOC2012', 'train')
train_data = data_manager.load_data()
arg_to_class = data_manager.arg_to_class

# generating output
generator = DataGenerator(train_data, prior_boxes, batch_size)
data = generator.flow('train')
output_1 = next(data)[1]['predictions']
output_2 = next(data)[1]['predictions']

multibox_loss = MultiboxLoss(num_classes, negative_positive_ratio, batch_size)

output_1 = K.variable(output_1)
output_2 = K.variable(output_2)
loss = multibox_loss.compute_loss(output_1, output_2)
session = K.get_session()
keras_loss = loss.eval(session=session)
print(keras_loss)

# probably the prior boxes and the ground truths don't fit properly and this
# will cause a problem
# prior_boxes_torch = PriorBox(v2).forward().type(torch.DoubleTensor).contiguous()
prior_boxes_torch = PriorBox(v2).forward().type(torch.DoubleTensor)
# prior_boxes_torch = prior_boxes_torch.numpy()
# prior_boxes_keras = create_prior_boxes()
prior_boxes_torch = Variable(prior_boxes_torch, volatile=True)
multibox_loss_torch = MultiboxLossTorch(num_classes, 0.5, True, 0,
                                        True, 3, 0.5, False, False)

loc_preds_1 = output_1.eval(session=session)[:, :, :4]
loc_preds_1 = Variable(torch.from_numpy(loc_preds_1).contiguous())
con_preds_1 = output_1.eval(session=session)[:, :, 4:]
con_preds_1 = Variable(torch.from_numpy(con_preds_1).contiguous())
torch_input_1 = [loc_preds_1, con_preds_1, prior_boxes_torch]

loc_preds_2 = output_2.eval(session=session)[:, :, :4]
con_preds_2 = output_2.eval(session=session)[:, :, 4:]
con_preds_2 = np.expand_dims(np.argmax(con_preds_2, axis=-1), -1)
torch_input_2 = np.concatenate((loc_preds_2, con_preds_2), axis=-1)
torch_input_2 = torch.from_numpy(torch_input_2).contiguous()
torch_input_2 = Variable(torch_input_2)
torch_loss = multibox_loss_torch(torch_input_1, torch_input_2)

print('keras_loss', np.mean(keras_loss))
print('torch_loss', torch_loss)
