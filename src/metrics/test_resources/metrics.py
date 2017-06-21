# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for computing metrics like precision, recall, CorLoc and etc."""
from __future__ import division

import numpy as np

import tensorflow as tf

from object_detection.utils import metrics


def compute_precision_recall(scores, labels, num_gt):
  """Compute precision and recall.

  Args:
    scores: A float numpy array representing detection score
    labels: A boolean numpy array representing true/false positive labels
    num_gt: Number of ground truth instances

  Raises:
    ValueError: if the input is not of the correct format

  Returns:
    precision: Fraction of positive instances over detected ones. This value is
      None if no ground truth labels are present.
    recall: Fraction of detected positive instance over all positive instances.
      This value is None if no ground truth labels are present.

  """
  if not isinstance(
      labels, np.ndarray) or labels.dtype != np.bool or len(labels.shape) != 1:
    raise ValueError("labels must be single dimension bool numpy array")

  if not isinstance(
      scores, np.ndarray) or len(scores.shape) != 1:
    raise ValueError("scores must be single dimension numpy array")

  if num_gt < np.sum(labels):
    raise ValueError("Number of true positives must be smaller than num_gt.")

  if len(scores) != len(labels):
    raise ValueError("scores and labels must be of the same size.")

  if num_gt == 0:
    return None, None

  sorted_indices = np.argsort(scores)
  sorted_indices = sorted_indices[::-1]
  labels = labels.astype(int)
  true_positive_labels = labels[sorted_indices]
  false_positive_labels = 1 - true_positive_labels
  cum_true_positives = np.cumsum(true_positive_labels)
  cum_false_positives = np.cumsum(false_positive_labels)
  precision = cum_true_positives.astype(float) / (
                cum_true_positives + cum_false_positives)
  recall = cum_true_positives.astype(float) / num_gt
  return precision, recall


def compute_average_precision(precision, recall):
  """Compute Average Precision according to the definition in VOCdevkit.

  Precision is modified to ensure that it does not decrease as recall
  decrease.

  Args:
    precision: A float [N, 1] numpy array of precisions
    recall: A float [N, 1] numpy array of recalls

  Raises:
    ValueError: if the input is not of the correct format

  Returns:
    average_precison: The area under the precision recall curve. NaN if
      precision and recall are None.

  """
  if precision is None:
    if recall is not None:
      raise ValueError("If precision is None, recall must also be None")
    return np.NAN

  if not isinstance(precision, np.ndarray) or not isinstance(recall,
                                                             np.ndarray):
    raise ValueError("precision and recall must be numpy array")
  if precision.dtype != np.float or recall.dtype != np.float:
    raise ValueError("input must be float numpy array.")
  if len(precision) != len(recall):
    raise ValueError("precision and recall must be of the same size.")
  if not precision.size:
    return 0.0
  if np.amin(precision) < 0 or np.amax(precision) > 1:
    raise ValueError("Precision must be in the range of [0, 1].")
  if np.amin(recall) < 0 or np.amax(recall) > 1:
    raise ValueError("recall must be in the range of [0, 1].")
  if not all(recall[i] <= recall[i + 1] for i in xrange(len(recall) - 1)):
    raise ValueError("recall must be a non-decreasing array")

  recall = np.concatenate([[0], recall, [1]])
  precision = np.concatenate([[0], precision, [0]])

  # Preprocess precision to be a non-decreasing array
  for i in range(len(precision) - 2, -1, -1):
    precision[i] = np.maximum(precision[i], precision[i + 1])

  indices = np.where(recall[1:] != recall[:-1])[0] + 1
  average_precision = np.sum(
      (recall[indices] - recall[indices - 1]) * precision[indices])
  return average_precision


def compute_cor_loc(num_gt_imgs_per_class,
                    num_images_correctly_detected_per_class):
  """Compute CorLoc according to the definition in the following paper.

  https://www.robots.ox.ac.uk/~vgg/rg/papers/deselaers-eccv10.pdf

  Returns nans if there are no ground truth images for a class.

  Args:
    num_gt_imgs_per_class: 1D array, representing number of images containing
        at least one object instance of a particular class
    num_images_correctly_detected_per_class: 1D array, representing number of
        images that are correctly detected at least one object instance of a
        particular class

  Returns:
    corloc_per_class: A float numpy array represents the corloc score of each
      class
  """
  return np.where(
      num_gt_imgs_per_class == 0,
      np.nan,
      num_images_correctly_detected_per_class / num_gt_imgs_per_class)


# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for object_detection.metrics."""


class MetricsTest(tf.test.TestCase):

  def test_compute_cor_loc(self):
    num_gt_imgs_per_class = np.array([100, 1, 5, 1, 1], dtype=int)
    num_images_correctly_detected_per_class = np.array([10, 0, 1, 0, 0],
                                                       dtype=int)
    corloc = metrics.compute_cor_loc(num_gt_imgs_per_class,
                                     num_images_correctly_detected_per_class)
    expected_corloc = np.array([0.1, 0, 0.2, 0, 0], dtype=float)
    self.assertTrue(np.allclose(corloc, expected_corloc))

  def test_compute_cor_loc_nans(self):
    num_gt_imgs_per_class = np.array([100, 0, 0, 1, 1], dtype=int)
    num_images_correctly_detected_per_class = np.array([10, 0, 1, 0, 0],
                                                       dtype=int)
    corloc = metrics.compute_cor_loc(num_gt_imgs_per_class,
                                     num_images_correctly_detected_per_class)
    expected_corloc = np.array([0.1, np.nan, np.nan, 0, 0], dtype=float)
    self.assertAllClose(corloc, expected_corloc)

  def test_compute_precision_recall(self):
    num_gt = 10
    scores = np.array([0.4, 0.3, 0.6, 0.2, 0.7, 0.1], dtype=float)
    labels = np.array([0, 1, 1, 0, 0, 1], dtype=bool)
    accumulated_tp_count = np.array([0, 1, 1, 2, 2, 3], dtype=float)
    expected_precision = accumulated_tp_count / np.array([1, 2, 3, 4, 5, 6])
    expected_recall = accumulated_tp_count / num_gt
    precision, recall = metrics.compute_precision_recall(scores, labels, num_gt)
    self.assertAllClose(precision, expected_precision)
    self.assertAllClose(recall, expected_recall)

  def test_compute_average_precision(self):
    precision = np.array([0.8, 0.76, 0.9, 0.65, 0.7, 0.5, 0.55, 0], dtype=float)
    recall = np.array([0.3, 0.3, 0.4, 0.4, 0.45, 0.45, 0.5, 0.5], dtype=float)
    processed_precision = np.array([0.9, 0.9, 0.9, 0.7, 0.7, 0.55, 0.55, 0],
                                   dtype=float)
    recall_interval = np.array([0.3, 0, 0.1, 0, 0.05, 0, 0.05, 0], dtype=float)
    expected_mean_ap = np.sum(recall_interval * processed_precision)
    mean_ap = metrics.compute_average_precision(precision, recall)
    self.assertAlmostEqual(expected_mean_ap, mean_ap)

  def test_compute_precision_recall_and_ap_no_groundtruth(self):
    num_gt = 0
    scores = np.array([0.4, 0.3, 0.6, 0.2, 0.7, 0.1], dtype=float)
    labels = np.array([0, 0, 0, 0, 0, 0], dtype=bool)
    expected_precision = None
    expected_recall = None
    precision, recall = metrics.compute_precision_recall(scores, labels, num_gt)
    self.assertEquals(precision, expected_precision)
    self.assertEquals(recall, expected_recall)
    ap = metrics.compute_average_precision(precision, recall)
    self.assertTrue(np.isnan(ap))


if __name__ == '__main__':
  tf.test.main()
