from itertools import product
import numpy as np

from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import concatenate
from keras.layers import Concatenate
import keras.backend as K


def add_ssd_modules(output_tensors, num_classes, num_priors,
                    with_batch_norm=True):

    classification_layers, regression_layers = [], []
    for layer_arg, base_layer in enumerate(output_tensors):

        str_arg = str(layer_arg)
        # classification leaf
        class_name = 'classification_leaf_' + str(layer_arg)
        class_leaf = Conv2D(num_priors[layer_arg] * num_classes, (3, 3),
                            padding='same', name=class_name)(base_layer)
        if with_batch_norm:
            class_leaf = BatchNormalization(
                    name='batch_norm_ssd_3_' + str_arg)(class_leaf)

        class_leaf = Flatten(name='flatten_ssd_1_' + str_arg)(class_leaf)
        classification_layers.append(class_leaf)

        # regression leaf
        regress_name = 'regression_leaf_' + str(layer_arg)
        regress_leaf = Conv2D(num_priors[layer_arg] * 4, (3, 3),
                              padding='same', name=regress_name)(base_layer)
        if with_batch_norm:
            regress_leaf = BatchNormalization(
                    name='batch_norm_ssd_4_' + str_arg)(regress_leaf)

        regress_leaf = Flatten(
                name='batch_norm_ssd_5_' + str_arg)(regress_leaf)
        regression_layers.append(regress_leaf)

    classifications = concatenate(classification_layers, axis=1)
    regressions = concatenate(regression_layers, axis=1)

    if hasattr(regressions, '_keras_shape'):
        num_boxes = regressions._keras_shape[-1] // 4
    elif hasattr(regressions, 'int_shape'):
        num_boxes = K.int_shape(regressions)[-1] // 4

    classifications = Reshape(
            (num_boxes, num_classes),
            name='reshape_new_ssd_1_' + str_arg)(classifications)

    classifications = Activation(
            'softmax', name='activation_ssd_1_' + str_arg)(classifications)

    regressions = Reshape((num_boxes, 4),
                          name='reshape_ssd_2_' + str_arg)(regressions)

    boxes_output = concatenate([regressions, classifications],
                               axis=2, name='predictions')
    return boxes_output


def modify_SSD(ssd_model, num_classes,
               num_priors=[4, 6, 6, 6, 4, 4],
               branch_name_prefix='prediction_feature_map_'):

    classification_layers, regression_layers = [], []
    layer_arg = 0
    for layer in ssd_model.layers:
        if branch_name_prefix in layer.name:
            branch = layer.output
            class_leaf_name = 'classification_leaf_new_' + str(layer_arg)
            classification = Conv2D(num_priors[layer_arg] * num_classes,
                                    (3, 3), padding='same',
                                    name=class_leaf_name)(branch)
            classification = BatchNormalization(
                    name=class_leaf_name + '_bn')(classification)
            classification = Flatten(
                    name=class_leaf_name + '_flatten')(classification)
            classification_layers.append(classification)

            # regression leaf
            regression_leaf_name = 'regression_leaf_new_' + str(layer_arg)
            regression = Conv2D(num_priors[layer_arg] * 4, (3, 3),
                                padding='same',
                                name=regression_leaf_name)(branch)
            regression = BatchNormalization(
                    name=regression_leaf_name + '_bn')(regression)
            regression = Flatten(
                    name=regression_leaf_name + '_flatten')(regression)
            regression_layers.append(regression)
            layer_arg = layer_arg + 1

    classifications = Concatenate(
            axis=1, name='concatenate_last_new')(classification_layers)
    regressions = concatenate(regression_layers, axis=1)

    if hasattr(regressions, '_keras_shape'):
        num_boxes = regressions._keras_shape[-1] // 4
    elif hasattr(regressions, 'int_shape'):
        num_boxes = K.int_shape(regressions)[-1] // 4

    classifications = Reshape((num_boxes, num_classes),
                              name='reshape_classes_new')(classifications)
    classifications = Activation('softmax',
                                 name='activation_last_new')(classifications)
    regressions = Reshape((num_boxes, 4),
                          name='reshape_regressions_new')(regressions)
    boxes_output = concatenate([regressions, classifications],
                               axis=2, name='predictions')
    return Model(ssd_model.input, boxes_output)


def construct_SSD(base_model, num_classes=21,
                  num_priors=[4, 6, 6, 6, 4, 4],
                  branch_name_prefix='branch_'):

    branch_tensors = []
    for layer in base_model.layers:
        if branch_name_prefix in layer.name:
            branch_tensors.append(layer.output)
    print(branch_tensors)
    boxes_output = add_ssd_modules(branch_tensors, num_classes, num_priors)
    return Model(base_model.input, boxes_output)


def make_prior_boxes(
            model, feature_map_prefix='branch_',
            layer_aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            min_sizes=[30, 60, 111, 162, 213, 264],
            max_sizes=[60, 111, 162, 213, 264, 315]):

    image_size = model.input_shape[1]
    branch_layer_names = []
    for layer in model.layers:
        if feature_map_prefix in layer.name:
            branch_layer_names.append(layer.name)
            # feature_map_sizes.append(layer.output_shape[1])

    # this sort will not work when the layers are bigger than 9
    feature_map_sizes = []
    for layer_name in sorted(branch_layer_names):
        layer = model.get_layer(layer_name)
        feature_map_sizes.append(layer.output_shape[1])

    prior_boxes = []
    for feature_map_arg, feature_map_size in enumerate(feature_map_sizes):
        min_size = min_sizes[feature_map_arg]
        max_size = max_sizes[feature_map_arg]
        aspect_ratios = layer_aspect_ratios[feature_map_arg]
        for y, x in product(range(feature_map_size), repeat=2):
            f_k = feature_map_size
            center_x = (x + 0.5) / f_k
            center_y = (y + 0.5) / f_k
            s_k = min_size / image_size
            prior_boxes = prior_boxes + [center_x, center_y, s_k, s_k]
            s_k_prime = np.sqrt(s_k * (max_size / image_size))
            prior_boxes = (prior_boxes +
                           [center_x, center_y, s_k_prime, s_k_prime])
            for aspect_ratio in aspect_ratios:

                prior_boxes = (prior_boxes +
                               [center_x, center_y,
                                s_k * np.sqrt(aspect_ratio),
                                s_k / np.sqrt(aspect_ratio)])

                prior_boxes = (prior_boxes +
                               [center_x, center_y,
                                s_k / np.sqrt(aspect_ratio),
                                s_k * np.sqrt(aspect_ratio)])

    prior_boxes = np.asarray(prior_boxes).reshape((-1, 4))
    prior_boxes = np.clip(prior_boxes, 0, 1)
    prior_boxes = to_point_form(prior_boxes)
    return prior_boxes


def to_point_form(boxes):

    center_x = boxes[:, 0]
    center_y = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    x_min = center_x - (width / 2.)
    x_max = center_x + (width / 2.)
    y_min = center_y - (height / 2.)
    y_max = center_y + (height / 2.)

    return np.concatenate([x_min[:, None], y_min[:, None],
                           x_max[:, None], y_max[:, None]], axis=1)
