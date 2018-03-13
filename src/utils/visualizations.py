import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import keras.backend as K
import cv2


def plot_kernels(kernel_weights, title='Kernel weights'):
    plt.figure(figsize=(20, 20))
    plt.title(title)
    mosaic = make_mosaic(kernel_weights)
    pretty_imshow(plt.gca(), mosaic)


def make_mosaic(images, border=1):
    print(images.shape)
    h, w, num_channels, num_images = images.shape
    side_box_size = int(np.ceil(np.sqrt(num_images)))
    border_lengths = (side_box_size - 1) * border
    image_shape = (side_box_size * w + border_lengths,
                   side_box_size * h + border_lengths,
                   num_channels)
    if num_channels == 1:
        image_shape = image_shape[:2]
        images = np.squeeze(images)
    mosaic = np.ma.masked_all(image_shape, dtype=np.float32)

    padded_h = h + border
    padded_w = w + border
    for image_arg in range(num_images):
        row_arg = int(np.floor(image_arg / side_box_size))
        col_arg = image_arg % side_box_size
        image = images[..., image_arg]
        mosaic[row_arg * padded_h:row_arg * padded_h + h,
               col_arg * padded_w:col_arg * padded_w + w] = image
    return mosaic


def pretty_imshow(axis, data, value_ranges=None,
                  with_colorbar=True, cmap=None):

    if value_ranges is None:
        value_ranges = (data.min(), data.max())

    if with_colorbar:
        image = axis.imshow(data, vmin=value_ranges[0], vmax=value_ranges[1],
                            interpolation='nearest', cmap=cmap)
        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=cax)
    else:
        image = axis.imshow(data, vmin=value_ranges[0], vmax=value_ranges[1],
                            interpolation='nearest')


def get_feature_map(model, layer_name, image, title='feature_map'):

    _get_feature_map = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.get_layer(layer_name).output])

    feature_map = _get_feature_map((image, False))[0]
    feature_map = np.squeeze(feature_map)
    feature_map = np.expand_dims(feature_map, 2)
    return feature_map


def preprocess_image(image_path, BGR_mean=(84.07, 89.3, 94.1)):
    image_array = cv2.imread(image_path)
    image_array = image_array.astype('float32')
    image_array[..., 0] -= BGR_mean[0]
    image_array[..., 1] -= BGR_mean[1]
    image_array[..., 2] -= BGR_mean[2]
    image_array = np.expand_dims(image_array, 0)
    return image_array


"""
if __name__ == "__main__":
    # import pickle
    # weights = pickle.load(open('conv2d_1_weights.pkl', 'rb'))
    # plot_kernels(weights)
    # plt.show()
    from keras.models import load_model
    import glob
    model = load_model('../trained_models/xception_128x128_plain.hdf5')
    model.summary()
    layer_name = 'conv2d_10'
    path = '/home/octavio/clara/data/invariance_tests/zoom_blue/'
    image_paths = sorted(glob.glob(path + '*.png'))
    for image_path in image_paths:
        image_array = preprocess_image(image_path)
        f, axis_array = plt.subplots(1, 2)
        axis_array[0].imshow(np.squeeze(image_array))
        feature_map = get_feature_map(model, layer_name, image_array)
        mosaic = make_mosaic(feature_map)
        # plt.figure(figsize=(15, 15))
        # pretty_imshow(plt.gca(), np.squeeze(mosaic))
        axis_array[1].imshow(np.squeeze(mosaic))
        plt.show()
"""
