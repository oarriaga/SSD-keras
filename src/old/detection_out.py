def detection_out(self, predictions, background_label_id=0, keep_top_k=200,
                  confidence_threshold=0.01):
    """Do non maximum suppression (nms) on prediction results.
    # Arguments
        predictions: Numpy array of predicted values.
        num_classes: Number of classes for prediction.
        background_label_id: Label of background class.
        keep_top_k: Number of total bboxes to be kept per image
            after nms step.
        confidence_threshold: Only consider detections,
            whose confidences are larger than a threshold.
    # Return
        results: List of predictions for every picture. Each prediction is:
            [label, confidence, xmin, ymin, xmax, ymax]
    """
    mbox_loc = predictions[:, :, :4]
    variances = predictions[:, :, -4:]
    mbox_priorbox = predictions[:, :, -8:-4]
    mbox_conf = predictions[:, :, 4:-8]
    results = []
    for i in range(len(mbox_loc)):
        results.append([])
        decode_bbox = self.decode_boxes(mbox_loc[i],
                                        mbox_priorbox[i], variances[i])
        for c in range(self.num_classes):
            if c == background_label_id:
                continue
            c_confs = mbox_conf[i, :, c]
            c_confs_m = c_confs > confidence_threshold
            if len(c_confs[c_confs_m]) > 0:
                boxes_to_process = decode_bbox[c_confs_m]
                confs_to_process = c_confs[c_confs_m]
                feed_dict = {self.boxes: boxes_to_process,
                             self.scores: confs_to_process}
                idx = self.sess.run(self.nms, feed_dict=feed_dict)
                good_boxes = boxes_to_process[idx]
                confs = confs_to_process[idx][:, None]
                labels = c * np.ones((len(idx), 1))
                c_pred = np.concatenate((labels, confs, good_boxes),
                                        axis=1)
                results[-1].extend(c_pred)
        if len(results[-1]) > 0:
            results[-1] = np.array(results[-1])
            argsort = np.argsort(results[-1][:, 1])[::-1]
            results[-1] = results[-1][argsort]
            results[-1] = results[-1][:keep_top_k]
    return results
