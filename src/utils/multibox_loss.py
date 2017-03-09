import tensorflow as tf
import keras.backend as K

class MultiboxLoss(object):
    """Multibox loss with some helper functions.

    # Arguments
        num_classes: Number of classes including background.
        alpha: Weight of L1-smooth loss.
        neg_pos_ratio: Max ratio of negative to positive boxes in loss.
        background_label_id: Id of background label.
        negatives_for_hard: Number of negative boxes to consider
            it there is no positive boxes in batch.

    # References
        https://arxiv.org/abs/1512.02325

    # TODO
        Add possibility for background label id be not zero
    """
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported')
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard

    def _l1_smooth_loss(self, y_true, y_pred):
        """Compute L1-smooth loss.

        # Arguments
            y_true: Ground truth bounding boxes,
                tensor of shape (?, num_boxes, 4).
            y_pred: Predicted bounding boxes,
                tensor of shape (?, num_boxes, 4).

        # Returns
            l1_loss: L1-smooth loss, tensor of shape (?, num_boxes).

        # References
            https://arxiv.org/abs/1504.08083
        """
        absolute_value_loss = K.abs(y_true - y_pred) - 0.5
        square_loss = 0.5 * (y_true - y_pred)**2
        absolute_value_condition = K.lesser(absolute_value_loss, 1.0)
        l1_smooth_loss = tf.select(absolute_value_condition,
                                        square_loss,
                                        absolute_value_loss)
        return K.sum(l1_smooth_loss, axis=-1)

    def _softmax_loss(self, y_true, y_pred):
        """Compute softmax loss.

        # Arguments
            y_true: Ground truth targets,
                tensor of shape (?, num_boxes, num_classes).
            y_pred: Predicted logits,
                tensor of shape (?, num_boxes, num_classes).

        # Returns
            softmax_loss: Softmax loss, tensor of shape (?, num_boxes).
        """
        y_pred = K.maximum(K.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = - K.sum(y_true * K.log(y_pred), axis=-1)
        return softmax_loss

    def compute_loss(self, y_true, y_pred):
        """Compute mutlibox loss.

        # Arguments
            y_true: Ground truth targets,
                tensor of shape (?, num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                y_true[:, :, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                y_true[:, :, -7:] are all 0.
            y_pred: Predicted logits,
                tensor of shape (?, num_boxes, 4 + num_classes + 8).

        # Returns
            loss: Loss for prediction, tensor of shape (?,).
        """
        batch_size = K.shape(y_true)[0]
        num_boxes = K.cast(K.shape(y_true)[1], 'float')

        # loss for all priors
        classification_loss = self._softmax_loss(y_true[:, :, 4:-8],
                                                 y_pred[:, :, 4:-8])
        localization_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                                 y_pred[:, :, :4])

        # get positives loss
        # num_positives is matrix of dimensions batch_size, num_priors
        num_positives = K.sum(y_true[:, :, -8], axis=-1)
        positive_localization_losses = localization_loss * y_true[:, :, -8]
        positive_localization_loss = K.sum(positive_localization_losses, 1)
        positive_classification_losses = classification_loss * y_true[:, :, -8]
        positive_classification_loss = K.sum(positive_classification_losses, 1)

        # TODO: Refactor -------------------------------------------------------
        # every batch contains all priors: here we take the least amount of
        # negatives which depends on the amount of positives at every batch
        # at every set of priors.
        num_negatives = self.neg_pos_ratio * num_positives
        """
        positive_num_negatives_mask = K.greater(num_negatives, 0)
        has_positive = K.cast(K.any(positive_num_negatives_mask), 'float')
        num_negatives = tf.concat(0, [num_negatives,
                            [(1 - has_positive) * self.negatives_for_hard]])
        positive_num_negatives = tf.boolean_mask(num_negatives,
                                    positive_num_negatives_mask)
        num_neg_batch = K.min(positive_num_negatives)
        num_neg_batch = K.cast(num_neg_batch, 'int32')
        """
        num_neg_batch = K.min(K.cast(num_negatives, 'int32'))
        # ----------------------------------------------------------------------

        class_start = 4 + self.background_label_id + 1
        class_end = class_start + self.num_classes - 1
        # each prior box can only have one class then we take the max at axis 2
        best_class_scores = K.max(y_pred[:, :, class_start:class_end], 2)
        y_true_negatives_mask = 1 - y_true[:, :, -8]
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
