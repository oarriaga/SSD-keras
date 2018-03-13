import keras.backend as K
import tensorflow as tf


class MultiboxLoss(object):
    def __init__(self, num_classes, neg_pos_ratio=3,
                 alpha=1.0, background_id=0, max_num_negatives=300):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.background_id = background_id
        self.max_num_negatives = max_num_negatives

    def smooth_l1(self, y_true, y_pred):
        absolute_value_loss = K.abs(y_true - y_pred) - 0.5
        square_loss = 0.5 * (y_true - y_pred)**2
        absolute_value_condition = K.less(absolute_value_loss, 1.0)
        l1_smooth_loss = tf.where(absolute_value_condition, square_loss,
                                  absolute_value_loss)
        return K.sum(l1_smooth_loss, axis=-1)

    def cross_entropy(self, y_true, y_pred):
        y_pred = K.maximum(K.minimum(y_pred, 1 - 1e-15), 1e-15)
        cross_entropy_loss = - K.sum(y_true * K.log(y_pred), axis=-1)
        return cross_entropy_loss

    def compute_loss(self, y_true, y_pred):
        class_loss = self.cross_entropy(y_true[:, :, 4:], y_pred[:, :, 4:])
        local_loss = self.smooth_l1(y_true[:, :, :4], y_pred[:, :, :4])
        negative_mask = y_true[:, :, 4 + self.background_id]
        positive_mask = 1 - negative_mask

        # calculating the positive loss per sample
        positive_local_losses = local_loss * positive_mask
        positive_class_losses = class_loss * positive_mask
        positive_class_loss = K.sum(positive_class_losses, axis=-1)
        positive_local_loss = K.sum(positive_local_losses, axis=-1)

        # obtaining the number of negatives in the batch per sample
        num_positives_per_sample = K.cast(K.sum(positive_mask, -1), 'int32')
        num_positives = K.sum(num_positives_per_sample)
        num_hard_negatives = self.neg_pos_ratio * num_positives_per_sample
        num_negatives_per_sample = K.minimum(num_hard_negatives,
                                             self.max_num_negatives)
        num_negatives = K.sum(num_negatives_per_sample)
        negative_class_losses = class_loss * negative_mask

        # from here on is freestyle
        # return negative_class_losses
        # negative_class_losses = K.batch_flatten(negative_class_losses)
        negative_class_losses = K.flatten(negative_class_losses)
        # return negative_class_losses
        negative_class_loss = tf.nn.top_k(
                negative_class_losses, num_negatives)[0]
        # return negative_class_loss
        return positive_class_loss 
        """
        elements = (negative_class_losses, num_negatives_per_sample)
        negative_class_loss = tf.map_fn(
                lambda x: K.sum(tf.nn.top_k(x[0], x[1])[0]),
                elements, dtype=tf.float32)

        class_loss = positive_class_loss + negative_class_loss
        """
        total_loss = class_loss + (self.alpha * positive_local_loss)

        # when the number of positives is zero set the total loss to zero
        batch_mask = K.not_equal(num_positives_per_sample, 0)
        total_loss = tf.where(batch_mask, total_loss, K.zeros_like(total_loss))

        num_positives_per_sample = tf.where(
                batch_mask, num_positives_per_sample,
                K.ones_like(num_positives_per_sample))

        num_positives_per_sample = K.cast(num_positives_per_sample, 'float32')
        total_loss = total_loss / num_positives_per_sample
        return total_loss
