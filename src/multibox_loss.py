import tensorflow as tf

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
    def __init__(self, num_classes, alpha=1.0, negative_positive_ratio=3.0,
                background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.negative_positive_ratio = negative_positive_ratio
        if background_label_id != 0:
            raise Exception('Id must be 0 for class background')
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard #?

    def _l1_smooth_loss(self, y_true, y_predicted):
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
        absolute_value_loss = tf.abs(y_true, y_predicted) - 0.5
        square_loss = 0.5 * (y_true - y_predicted)**2
        smooth_l1_condition = tf.less(absolute_value_loss, 1.0)
        l1_smooth_loss = tf.select(smooth_l1_condition,
                                    square_loss,
                                    absolute_value_loss)
        return tf.reduce_sum(l1_smooth_loss, -1)

    def _softmax_loss(self, y_true, y_predicted):
        """Compute softmax loss.
        # Arguments
            y_true: Ground truth targets,
                tensor of shape (?, num_boxes, num_classes).
            y_pred: Predicted logits,
                tensor of shape (?, num_boxes, num_classes).
        # Returns
            softmax_loss: Softmax loss, tensor of shape (?, num_boxes).
        """
        y_predicted = tf.maximum(tf.minimum(y_predicted, 1- 1e-15), 1e-15)
        softmax_loss = -tf.reduce_sum(y_true * tf.log(y_predicted),
                                    reduction_indices=-1)
        return softmax_loss

    def compute_multibox_loss(self, y_true, y_predicted):
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

        # Questions
            Q: What is 4 + num_classes + 8?
            A: 4 is for box coordinates, but +8 is full of zeros in the y_true
            It is probably related to the box coordinates and the... ?
        """

        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.to_float(tf.shape(y_true)[1])

        # loss for all prior boxes
        confidence_loss = self._softmax_loss(y_true[:,:,4:-8],
                                            y_predicted[:,:,4:-8])
        localization_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                                y_predicted[:, :, :4])

        # get positive loss
        num_positives = tf.reduce_sum(y_true[:, :, -8], reduction_indices=-1)

        positive_localization_loss = tf.reduce_sum(localization_loss *
                                                    y_true[:, :, -8],
                                                    reduction_indices=1)
        positive_confidence_loss = tf.reduce_sum(confidence_loss *
                                                    y_true[:, :, -8],
                                                    reduction_indices=1)

        # get negative loss, we penalize only confidence here
        negative_loss_1 = self.negative_positive_ratio * num_positives
        negative_loss_2 = num_boxes - num_positives
        num_negatives = tf.minimum(negative_loss_1, negative_loss_2)
        positive_num_negatives_mask = tf.greater(num_negatives, 0)
        has_min = tf.to_float(tf.reduce_any(positive_num_negatives_mask))
        num_negatives = tf.concat(0, [num_negatives,])










