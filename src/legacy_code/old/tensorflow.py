import tensorflow as tf

non_maximum_supression = tf.image.non_max_suppression(boxes, scores, 100,
                                                    iou_threshold=.2)

self.session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

