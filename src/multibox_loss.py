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

    def _l1_smooth_loss(self, y_true, y_pred):
        absolute_value_loss = tf.abs(y_true - y_pred) - 0.5
        square_loss = 0.5 * (y_true - y_pred)**2
        absolute_value_condition = K.lesser(absolute_value_loss, 1.0)
        l1_smooth_loss = tf.select(absolute_value_condition, square_loss,
                                                    absolute_value_loss)
        return K.sum(l1_smooth_loss, axis=-1)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = K.maximum(K.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = - K.sum(y_true * K.log(y_pred), axis=-1)
        return softmax_loss

    def compute_loss(self, y_true, y_pred):
        batch_size = K.shape(y_true)[0]
        num_prior_boxes = K.cast(K.shape(y_true)[1], 'float')

        y_pred_localization = y_pred[:, :, :4]
        y_true_localization = y_true[:, :, :4]
        y_pred_classification = y_pred[:, :, 4:(4 + self.num_classes)]
        y_true_classification = y_true[:, :, 4:(4 + self.num_classes)]
        # loss for all priors boxes
        localization_loss = self._l1_smooth_loss(y_true_localization, y_pred_localization)
        classification_loss = self._softmax_loss(y_true_classification, y_pred_classification)

        int_positive_mask = 1 - y_true[:, :, 4 + self.background_id]
        num_positives = tf.reduce_sum(int_positive_mask, axis=-1)
        positive_localization_losses = (localization_loss * int_positive_mask) #scalar times vector
        positive_classification_losses = (classification_loss * int_positive_mask)
        positive_classification_loss = K.sum(positive_classification_losses, 1)
        positive_localization_loss = K.sum(positive_localization_losses, 1)

        # TODO: Refactor/understand ----------------------------------------------
        # every batch contains all priors: here we take the least amount of
        # negatives which depends on the amount of positives at every batch
        # at every set of priors. num_negatives/positives = (?, num_positives)
        # in the second num_positive_mask the values the concatenated value does
        # not get counted since you are doing greater than zero.
        # the most probable value that num_neg_batch will have is:
        # neg_pos_ratio * num_positives where num_positives is the batch element
        # with less positive boxes.
        num_negatives_1 = self.neg_pos_ratio * num_positives
        num_negatives_2 = num_prior_boxes - num_positives
        num_negatives = tf.minimum(num_negatives_1, num_negatives_2)

        #positive_num_negatives_mask = tf.greater(num_negatives, 0)
        num_positive_mask = tf.greater(num_negatives, 0)
        has_a_positive = tf.to_float(tf.reduce_any(num_positive_mask))
        num_negatives = tf.concat(0, [num_negatives,
                                     [(1 - has_a_positive) * self.negatives_for_hard]])
        num_positive_mask = tf.greater(num_negatives, 0)
        num_neg_batch = tf.reduce_min(tf.boolean_mask(num_negatives, num_positive_mask))
        num_neg_batch = tf.to_int32(num_neg_batch)
        # ----------------------------------------------------------------------

        #class_start = 4 + self.background_id + 1
        #class_end = class_start + self.num_classes - 1
        # each prior box can only have one class then we take the max at axis 2
        #best_class_scores = K.max(y_pred[:, :, class_start:], 2)

        # picking up the negative examples with the highest probability (highest loss)
        ### ?????? THIS IS WEIRD, the original implementation starts from 5: therefore it
        #### does not take into consideration the background boxes
        pred_class_values = K.max(y_pred_classification[:, :, 1:], axis=2)
        int_negatives_mask = y_true[:, :, 4 + self.background_id]
        pred_negative_class_values = pred_class_values * int_negatives_mask
        top_k_negative_indices = tf.nn.top_k(pred_negative_class_values, k=num_neg_batch)[1]

        batch_indices = K.expand_dims(K.arange(0, batch_size), 1)
        batch_indices = K.tile(batch_indices, (1, num_neg_batch))
        batch_indices = K.flatten(batch_indices) * K.cast(num_prior_boxes, 'int32')
        full_indices = batch_indices + K.flatten(top_k_negative_indices)

        negative_classification_loss = K.gather(K.flatten(classification_loss), full_indices)
        negative_classification_loss = K.reshape(negative_classification_loss,
                                                    [batch_size, num_neg_batch])
        negative_classification_loss = K.sum(negative_classification_loss, 1)

        # loss is sum of positives and negatives
        total_loss = positive_classification_loss + negative_classification_loss
        num_prior_boxes_per_batch = num_positives + K.cast(num_neg_batch, 'float')
        total_loss = total_loss / num_prior_boxes_per_batch
        num_positives = tf.select(K.not_equal(num_positives, 0), num_positives,
                                                   K.ones_like(num_positives))
        positive_localization_loss = self.alpha * positive_classification_loss
        positive_localization_loss = positive_localization_loss / num_positives
        total_loss = total_loss + positive_localization_loss
        return total_loss
