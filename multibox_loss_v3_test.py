from multibox_loss_v3 import MultiboxLoss
import numpy as np
import tensorflow as tf

session = tf.InteractiveSession()

num_classes = 4
batch_size = 16
num_boxes = 3750
batch_size = (batch_size, num_boxes, 4+num_classes+8)

y_true = np.random.rand(*batch_size)
y_pred = np.random.rand(*batch_size)
y_true[:, :, -1] = (y_true[:, :, -1] > .5).astype(int)
y_pred[:, :, -1] = (y_pred[:, :, -1] > .5).astype(int)

multibox_loss = MultiboxLoss(num_classes)
multibox_loss.compute_loss(y_true, y_pred)







