import tensorflow as tf
import keras.backend as K

class MultiboxLoss(object):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=300.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            print('Only 0 as background label id is supported')
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard

    def _l1_smooth_loss(self, y_true, y_pred):
        absolute_value_loss = K.abs(y_true - y_pred) - 0.5
        square_loss = 0.5 * (y_true - y_pred)**2
        absolute_value_condition = K.lesser(absolute_value_loss, 1.0)
        l1_smooth_loss = tf.select(absolute_value_condition,
                                        square_loss,
                                        absolute_value_loss)
        return K.sum(l1_smooth_loss, axis=-1)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = K.maximum(K.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = - K.sum(y_true * K.log(y_pred), axis=-1)
        return softmax_loss

    def compute_loss(self, y_true, y_pred):
        batch_size = K.shape(y_true)[0]
        num_boxes = K.cast(K.shape(y_true)[1], 'float')

        y_pred_loc = y_pred[0]
        y_pred_class = y_pred[1]
        # loss for all priors
        classification_loss = self._softmax_loss(y_true[:, :, 4:], y_pred_class)
        localization_loss = self._l1_smooth_loss(y_true[:, :, :4], y_pred_loc)

        # get positives loss
        # num_positives is matrix of dimensions batch_size, num_priors
        # BUG esto esta mal1!!!!!!! es al revez

        num_positives = K.sum(1 - y_true[:, :, 4 + self.background_id + 1], axis=-1)
        positive_localization_losses = (localization_loss * (1 - y_true[:, :, 4 + self.background_label_id + 1]))
        positive_localization_loss = K.sum(positive_localization_losses, 1)
        positive_classification_losses = (classification_loss * (1 - y_true[:, :, 4 + self.background_label_id + 1]))
        positive_classification_loss = K.sum(positive_classification_losses, 1)

        # TODO: Refactor -------------------------------------------------------
        # every batch contains all priors: here we take the least amount of
        # negatives which depends on the amount of positives at every batch
        # at every set of priors.
        num_negatives = self.neg_pos_ratio * num_positives
        positive_num_negatives_mask = K.greater(num_negatives, 0)
        has_positive = K.cast(K.any(positive_num_negatives_mask), 'float')
        num_negatives = tf.concat(0, [num_negatives,
                            [(1 - has_positive) * self.negatives_for_hard]])
        positive_num_negatives = tf.boolean_mask(num_negatives,
                                    positive_num_negatives_mask)
        num_neg_batch = K.min(positive_num_negatives)
        num_neg_batch = K.cast(num_neg_batch, 'int32')
        #num_neg_batch = K.min(K.cast(num_negatives, 'int32'))
        # ----------------------------------------------------------------------

        #class_start = 4 + self.background_label_id + 1
        #class_end = class_start + self.num_classes - 1
        # each prior box can only have one class then we take the max at axis 2
        #best_class_scores = K.max(y_pred[:, :, class_start:], 2)
        best_class_scores = K.max(y_pred_loc, 2)
        y_true_negatives_mask = y_true[:, :, 4 + self.background_label_id]
        best_negative_class_scores = best_class_scores * y_true_negatives_mask
        top_k_negative_indices = tf.nn.top_k(best_negative_class_scores,
                                                    k=num_neg_batch)[1]
        batch_indices = K.expand_dims(K.arange(0, batch_size), 1)
        batch_indices = K.tile(batch_indices, (1, num_neg_batch))
        batch_indices = K.flatten(batch_indices) * K.cast(num_boxes, 'int32')
        full_indices = batch_indices + K.flatten(top_k_negative_indices)

        negative_classification_loss = K.gather(K.flatten(classification_loss),
                                                                full_indices)
        negative_classification_loss = K.reshape(negative_classification_loss,
                                                [batch_size, num_neg_batch])
        negative_classification_loss = K.sum(negative_classification_loss, 1)

        # loss is sum of positives and negatives
        total_loss = positive_classification_loss +negative_classification_loss
        num_boxes_per_batch = num_positives + K.cast(num_neg_batch, 'float')
        total_loss = total_loss / num_boxes_per_batch
        num_positives = tf.select(K.not_equal(num_positives, 0), num_positives,
                                                   K.ones_like(num_positives))
        positive_localization_loss = self.alpha * positive_classification_loss
        positive_localization_loss = positive_localization_loss / num_positives
        total_loss = total_loss + positive_localization_loss
        return total_loss
