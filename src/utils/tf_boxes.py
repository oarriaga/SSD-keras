import tensorflow as tf


session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))


def apply_non_max_suppression(boxes, scores, nms_treshold=.45,
                              max_output_size=200):
    num_boxes = len(boxes)
    coordinates = tf.placeholder(dtype='float32', shape=(num_boxes, 4))
    tf_scores = tf.placeholder(dtype='float32', shape=(num_boxes))
    boxes = boxes[:, (1, 0, 3, 2)]
    feed_dict = {coordinates: boxes, tf_scores: scores}
    non_maximum_supression = tf.image.non_max_suppression(coordinates,
                                                          scores,
                                                          max_output_size,
                                                          nms_treshold)
    indices = session.run(non_maximum_supression, feed_dict=feed_dict)
    return indices
