import tensorflow as tf
import numpy as np


session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))


def apply_non_max_suppression(box_data, iou_treshold=.45,
                              max_output_size=200):
    num_boxes = len(box_data)
    coordinates = tf.placeholder(dtype='float32', shape=(num_boxes, 4))
    scores = tf.placeholder(dtype='float32', shape=(num_boxes))
    class_prob = np.max(box_data[:, 4:], axis=-1)
    boxes = box_data[:, (1, 0, 3, 2)]
    feed_dict = {coordinates: boxes, scores: class_prob}
    non_maximum_supression = tf.image.non_max_suppression(coordinates,
                                                          scores,
                                                          max_output_size,
                                                          iou_treshold)
    indices = session.run(non_maximum_supression, feed_dict=feed_dict)
    return box_data[indices]
