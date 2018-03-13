from utils.visualizations import plot_kernels
import matplotlib.pyplot as plt
# from models.ssd_utils import construct_SSD
from models import SSD300


"""
weights_path = '../trained_models/SSD300_weights.hdf5'
model = SSD300(weights_path=weights_path)
model.summary()
kernel_weights = model.get_layer('conv2d_1').get_weights()[0]
plot_kernels(kernel_weights[:, :, 0:1, :])
plot_kernels(kernel_weights[:, :, 1:2, :])
plot_kernels(kernel_weights[:, :, 2:3, :])
plt.show()
"""

weights_path = '../trained_models/VGG16_weights.hdf5'
model = SSD300()
model.load_weights(weights_path, by_name=True)
kernel_weights = model.get_layer('conv2d_1').get_weights()[0]
plot_kernels(kernel_weights[:, :, 0:1, :])
plot_kernels(kernel_weights[:, :, 1:2, :])
plot_kernels(kernel_weights[:, :, 2:3, :])
plt.show()
