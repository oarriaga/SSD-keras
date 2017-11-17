import tensorflow as tf
import keras.backend as K


class MultiboxLoss(object):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.background_id = background_id
        self.negatives_for_hard = negatives_for_hard

    def smooth_l1(self, y_true, y_pred):
        absolute_value_loss = tf.abs(y_true - y_pred) - 0.5
        square_loss = 0.5 * (y_true - y_pred)**2
        absolute_value_condition = K.less(absolute_value_loss, 1.0)
        l1_smooth_loss = tf.where(absolute_value_condition, square_loss,
                                  absolute_value_loss)
        return K.sum(l1_smooth_loss, axis=-1)

    def cross_entropy(self, y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred)

    def compute_loss(self, y_true, y_pred):
        batch_size = K.shape(y_true)[0]
        num_prior_boxes = K.cast(K.shape(y_true)[1], 'float')

        class_loss = self.cross_entropy(y_true[:, :, 4:], y_pred[:, :, 4:])
        localization_loss = self.smooth_l1(y_true[:, :, :4], y_pred[:, :, :4])

        positive_mask = 1 - y_true[:, :, 4 + self.background_id]
        num_positives = tf.reduce_sum(positive_mask, axis=-1)
        positive_localization_losses = (localization_loss * positive_mask)
        positive_class_losses = (class_loss * positive_mask)
        positive_class_loss = K.sum(positive_class_losses, 1)
        positive_localization_loss = K.sum(positive_localization_losses, 1)

        num_negatives_1 = self.neg_pos_ratio * num_positives
        num_negatives_2 = num_prior_boxes - num_positives
        num_negatives = tf.minimum(num_negatives_1, num_negatives_2)

        num_positive_mask = tf.greater(num_negatives, 0)
        has_a_positive = tf.to_float(tf.reduce_any(num_positive_mask))
        num_negatives = tf.concat([num_negatives,
                                  [(1 - has_a_positive) *
                                      self.negatives_for_hard]], 0)
        num_positive_mask = tf.greater(num_negatives, 0)
        num_neg_batch = tf.reduce_min(tf.boolean_mask(num_negatives,
                                      num_positive_mask))
        num_neg_batch = tf.to_int32(num_neg_batch)

        pred_class_values = K.max(y_pred[:, :, 5:], axis=2)
        int_negatives_mask = y_true[:, :, 4 + self.background_id]
        pred_negative_class_values = pred_class_values * int_negatives_mask
        top_k_negative_indices = tf.nn.top_k(pred_negative_class_values,
                                             k=num_neg_batch)[1]

        batch_indices = K.expand_dims(K.arange(0, batch_size), 1)
        batch_indices = K.tile(batch_indices, (1, num_neg_batch))
        batch_indices = K.flatten(batch_indices) * K.cast(num_prior_boxes,
                                                          'int32')
        full_indices = batch_indices + K.flatten(top_k_negative_indices)

        negative_class_loss = K.gather(K.flatten(class_loss), full_indices)
        negative_class_loss = K.reshape(negative_class_loss,
                                        [batch_size, num_neg_batch])
        negative_class_loss = K.sum(negative_class_loss, 1)

        # loss is sum of positives and negatives
        total_loss = (positive_class_loss +
                      negative_class_loss)
        num_prior_boxes_per_batch = (num_positives +
                                     K.cast(num_neg_batch, 'float'))
        total_loss = total_loss / num_prior_boxes_per_batch
        num_positives = tf.where(K.not_equal(num_positives, 0), num_positives,
                                 K.ones_like(num_positives))
        positive_localization_loss = self.alpha * positive_class_loss
        positive_localization_loss = positive_localization_loss / num_positives
        total_loss = total_loss + positive_localization_loss
        return total_loss
