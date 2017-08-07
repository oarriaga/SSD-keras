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

"""object_detection_evaluation module.
ObjectDetectionEvaluation is a class which manages ground truth information
of a object detection dataset, and computes frequently used detection metrics
such as Precision, Recall, CorLoc of the provided detection results.
It supports the following operations:
1) Add ground truth information of images sequentially.
2) Add detection result of images sequentially.
3) Evaluate detection metrics on already inserted detection results.
4) Write evaluation result into a pickle file for future processing or
   visualization.
Note: This module operates on numpy boxes and box lists.
"""

import copy
import logging
import numpy as np
from object_detection_evaluation import ObjectDetectionEvaluation


def evaluate_detection_results_pascal_voc(result_lists,
                                          categories,
                                          label_id_offset=0,
                                          iou_thres=0.5,
                                          corloc_summary=False):
    """Computes Pascal VOC detection metrics given groundtruth and detections.
    This function computes Pascal VOC metrics. This function by default
    takes detections and groundtruth boxes encoded in result_lists and writes
    evaluation results to tf summaries which can be viewed on tensorboard.
    Args:
    result_lists: a dictionary holding lists of groundtruth and detection
      data corresponding to each image being evaluated.  The following keys
      are required:
        'image_id': a list of string ids
        'detection_boxes': a list of float32 numpy arrays of shape [N, 4]
        'detection_scores': a list of float32 numpy arrays of shape [N]
        'detection_classes': a list of int32 numpy arrays of shape [N]
        'groundtruth_boxes': a list of float32 numpy arrays of shape [M, 4]
        'groundtruth_classes': a list of int32 numpy arrays of shape [M]
      and the remaining fields below are optional:
        'difficult': a list of boolean arrays of shape [M] indicating the
        difficulty of groundtruth boxes. Some datasets like PASCAL VOC provide
        this information and it is used to remove difficult examples from eval
          in order to not penalize the models on them.
      Note that it is okay to have additional fields in result_lists --- they
      are simply ignored.
    categories: a list of dictionaries representing all possible categories.
      Each dict in this list has the following keys:
          'id': (required) an integer id uniquely identifying this category
          'name': (required) string representing category name
            e.g., 'cat', 'dog', 'pizza'
    label_id_offset: an integer offset for the label space.
    iou_thres: float determining the IoU threshold at which a box is considered
        correct. Defaults to the standard 0.5.
    corloc_summary: boolean. If True, also outputs CorLoc metrics.
    Returns:
    A dictionary of metric names to scalar values.
    Raises:
    ValueError: if the set of keys in result_lists is not a superset of the
      expected list of keys.  Unexpected keys are ignored.
    ValueError: if the lists in result_lists have inconsistent sizes.
    """
    # check for expected keys in result_lists
    expected_keys = [
        'detection_boxes', 'detection_scores', 'detection_classes', 'image_id'
        ]
    expected_keys += ['groundtruth_boxes', 'groundtruth_classes']
    if not set(expected_keys).issubset(set(result_lists.keys())):
        raise ValueError('result_lists does not have expected key set.')
    num_results = len(result_lists[expected_keys[0]])
    for key in expected_keys:
        if len(result_lists[key]) != num_results:
            raise ValueError('Inconsistent list sizes in result_lists')

    # Pascal VOC evaluator assumes foreground index starts from zero.
    categories = copy.deepcopy(categories)
    for idx in range(len(categories)):
        categories[idx]['id'] -= label_id_offset

    # num_classes (maybe encoded as categories)
    num_classes = max([cat['id'] for cat in categories]) + 1
    logging.info('Computing Pascal VOC metrics on results.')
    if all(image_id.isdigit() for image_id in result_lists['image_id']):
        image_ids = [int(image_id) for image_id in result_lists['image_id']]
    else:
        image_ids = range(num_results)
    """
    evaluator = object_detection_evaluation.ObjectDetectionEvaluation(
        num_classes, matching_iou_threshold=iou_thres)
    """
    evaluator = ObjectDetectionEvaluation(
                    num_classes, matching_iou_threshold=iou_thres)

    difficult_lists = None
    if 'difficult' in result_lists and result_lists['difficult']:
        difficult_lists = result_lists['difficult']
    for idx, image_id in enumerate(image_ids):
        difficult = None
        if difficult_lists is not None and difficult_lists[idx].size:
            difficult = difficult_lists[idx].astype(np.bool)
        evaluator.add_single_ground_truth_image_info(
            image_id, result_lists['groundtruth_boxes'][idx],
            result_lists['groundtruth_classes'][idx] - label_id_offset,
            difficult)
        evaluator.add_single_detected_image_info(
            image_id, result_lists['detection_boxes'][idx],
            result_lists['detection_scores'][idx],
            result_lists['detection_classes'][idx] - label_id_offset)

    per_class_ap, mean_ap, _, _, per_class_corloc, mean_corloc = (
        evaluator.evaluate())

    metrics = {'Precision/mAP@{}IOU'.format(iou_thres): mean_ap}
    # category_index = label_map_util.create_category_index(categories)
    category_index = create_category_index(categories)
    for idx in range(per_class_ap.size):
        if idx in category_index:
            display_name = ('PerformanceByCategory/mAP@{}IOU/{}'
                            .format(iou_thres, category_index[idx]['name']))
            metrics[display_name] = per_class_ap[idx]

    if corloc_summary:
        metrics['CorLoc/CorLoc@{}IOU'.format(iou_thres)] = mean_corloc
        for idx in range(per_class_corloc.size):
            if idx in category_index:
                display_name = (
                    'PerformanceByCategory/CorLoc@{}IOU/{}'.format(
                        iou_thres, category_index[idx]['name']))
                metrics[display_name] = per_class_corloc[idx]
    return metrics


def create_category_index(categories):
    """Creates dictionary of COCO compatible categories keyed by category id.
    Args:
    categories: a list of dicts, each of which has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.
    Returns:
    category_index: a dict containing the same entries as categories, but keyed
      by the 'id' field of each category.
    """
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index
