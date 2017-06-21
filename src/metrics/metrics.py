import numpy as np

def compute_precision_and_recall(scores, labels, num_gt):
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


scores = np.random.rand(10)
print(scores)
labels = np.random.randint(0, 2, (10)).astype(bool)
print(labels)
num_gt = 10

precision, recall = compute_precision_recall(scores, labels, num_gt)





