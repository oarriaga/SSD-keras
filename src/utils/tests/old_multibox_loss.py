import keras.backend as K
import tensorflow as tf


class MultiboxLoss(object):
    def __init__(self, num_classes, neg_pos_ratio=3, batch_size=32,
                 alpha=1.0, background_id=0, max_num_negatives=300):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.background_id = background_id
        self.batch_size = batch_size
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

        # return K.concatenate([class_loss, class_loss_old], axis=0)
        local_loss = self.smooth_l1(y_true[:, :, :4], y_pred[:, :, :4])
        negative_mask = y_true[:, :, 4 + self.background_id]
        positive_mask = 1 - negative_mask

        # calculating the positive loss
        positive_local_losses = local_loss * positive_mask
        positive_class_losses = class_loss * positive_mask
        positive_class_loss = K.sum(positive_class_losses, axis=-1)
        positive_local_loss = K.sum(positive_local_losses, axis=-1)

        # obtaining the number of negatives in the batch
        num_positives_per_sample = K.cast(K.sum(positive_mask, -1), 'int32')
        num_hard_negatives = self.neg_pos_ratio * num_positives_per_sample
        num_negatives_per_sample = K.minimum(num_hard_negatives,
                                             self.max_num_negatives)
        negative_class_losses = class_loss * negative_mask

        negative_class_loss = []
        for sample_arg in range(self.batch_size):
            num_negatives_in_sample = num_negatives_per_sample[sample_arg]
            negative_sample_loss = negative_class_losses[sample_arg]
            selected_negative_sample_losses = tf.nn.top_k(
                                    negative_sample_loss,
                                    k=num_negatives_in_sample,
                                    sorted=True)[0]
            negative_sample_loss = K.sum(selected_negative_sample_losses)
            negative_sample_loss = K.expand_dims(negative_sample_loss, -1)
            negative_class_loss.append(negative_sample_loss)
        negative_class_loss = K.concatenate(negative_class_loss)

        class_loss = positive_class_loss + negative_class_loss
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
