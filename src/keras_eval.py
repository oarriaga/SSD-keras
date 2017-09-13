"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torchvision.transforms as transforms
# from torch.autograd import Variable
# from data import VOCroot
from pytorch_tests.pytorch_parameters import VOC_CLASSES as labelmap
from utils.boxes import unregress_boxes
from utils.boxes import create_prior_boxes
from utils.boxes import apply_non_max_suppression
# from data import VOC_CLASSES as labelmap
# import torch.utils.data as data

# from data import AnnotationTransform, VOCDetection, BaseTransform
from pytorch_tests.pytorch_datasets import VOCDetection
from pytorch_tests.pytorch_datasets import AnnotationTransform
from pytorch_tests.pytorch_datasets import BaseTransform
#
# from ssd import build_ssd

from models import SSD300

import sys
import os
import time
import argparse
import numpy as np
import pickle
# import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


voc_root = '../datasets/VOCdevkit'
prior_boxes = create_prior_boxes()


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=voc_root,
                    help='Location of VOC root directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
"""
if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
"""
devkit_path = '../datasets/VOCdevkit/VOC2007/'
annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007',
                          'ImageSets', 'Main', '{:s}.txt')


# annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
# imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
# imgsetpath = os.path.join(args.voc_root,
# 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
YEAR = '2007'
# devkit_path = VOCroot + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format(set_type), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')
    detect = Detect(21, 0, 200, 0.01, .45)
    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        keras_image = np.squeeze(im)

        # keras_image = substract_mean(x)
        keras_image_input = np.expand_dims(keras_image, axis=0)
        keras_output = net.predict(keras_image_input)
        detections = detect.forward(keras_output, prior_boxes)

        # x = Variable(im.unsqueeze(0))
        """
        if args.cuda:
            x = x.cuda()
        """
        _t['im_detect'].tic()
        # detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        detection_size = 21
        # for j in range(1, detections.size(1)):
        for j in range(1, detection_size):
            dets = detections[0, j, :]
            mask = np.squeeze(dets[:, 0] > 0.01)
            dets = dets[mask]
            if len(dets) == 0:
                continue
            """
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            """

            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            # scores = dets[:, 0].cpu().numpy()
            scores = dets[:, 0]
            """
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            """
            cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
                        np.float32, copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = np.zeros(shape=len(scores))
    if boxes is None or len(boxes) == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    idx = np.argsort(scores)
    idx = idx[-top_k:]

    count = 0
    while len(idx) > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if len(idx) == 1:
            break
        idx = idx[:-1]
        xx1 = x1[idx]
        yy1 = y1[idx]
        xx2 = x2[idx]
        yy2 = y2[idx]

        xx1 = np.maximum(xx1, x1[i])
        yy1 = np.maximum(yy1, y1[i])
        xx2 = np.minimum(xx2, x2[i])
        yy2 = np.minimum(yy2, y2[i])

        w = xx2 - xx1
        h = yy2 - yy1

        w = np.maximum(w, 0.0)
        h = np.maximum(h, 0.0)

        inter = w*h
        rem_areas = area[idx]
        union = (rem_areas - inter) + area[i]
        IoU = inter/union
        print(IoU)
        # print(IoU)
        iou_mask = IoU <= overlap
        idx = idx[iou_mask]
        # print('numpy:', len(idx))
    return keep.astype(int), count


class Detect():
    def __init__(self, num_classes=21, bkg_label=0, top_k=200,
                 conf_thresh=0.099, nms_thresh=.45):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = [.1, .1, .2, .2]
        # self.output = np.zeros((1, self.num_classes, self.top_k, 5))

    def forward(self, box_data, prior_boxes):
        box_data = np.squeeze(box_data)
        regressed_boxes = box_data[:, :4]
        class_predictions = box_data[:, 4:]
        decoded_boxes = unregress_boxes(regressed_boxes, prior_boxes,
                                        self.variance)
        output = np.zeros((1, self.num_classes, self.top_k, 5))
        class_selections = []
        for class_arg in range(1, self.num_classes):
            conf_mask = class_predictions[:, class_arg] >= (self.conf_thresh)
            scores = class_predictions[:, class_arg][conf_mask]
            if len(scores) == 0:
                continue
            boxes = decoded_boxes[conf_mask]
            # pickle.dump(boxes, open('numpy_boxes.pkl', 'wb'))
            # pickle.dump(scores, open('numpy_scores.pkl', 'wb'))

            # indices, count = nms(boxes, scores, self.nms_thresh, self.top_k)
            indices, count = apply_non_max_suppression(boxes, scores,
                                                       self.nms_thresh,
                                                       self.top_k)
            # print('indices:', indices)
            # print('indices_shape:', indices.shape)
            scores = np.expand_dims(scores, -1)
            # print('scores_shape:', scores[indices].shape)
            # print('boxes_shape:', boxes[indices].shape)
            selections = np.concatenate((scores[indices[:count]],
                                        boxes[indices[:count]]), axis=1)

            class_selections.append(selections)
            # pickle.dump(selections, open('numpy_selections.pkl', 'wb'))
            # print('selections_shape', selections.shape)
            # print('count:', count)
            # self.output[0, class_arg, :count] = selections
            # print('numpy_selections:', selections)
            # print('numpy class_arg:', class_arg)
            # print('numpy count', count)
            # self.output[0, class_arg, :count, :] = selections
            output[0, class_arg, :count, :] = selections
        # pickle.dump(class_selections, open('numpy_selections.pkl', 'wb'))
        return output


if __name__ == '__main__':
    # load net
    # net = build_ssd('test', 300, 21)    # initialize SSD
    # net.load_state_dict(torch.load(args.trained_model))
    # net.eval()
    weights_path = '../trained_models/SSD300_weights.hdf5'
    net = SSD300(weights_path=weights_path)
    print('Finished loading model!')
    # load data
    """
    dataset = VOCDetection(args.voc_root, [('2007', set_type)],
                           BaseTransform(300, dataset_mean),
                           AnnotationTransform())
    """
    R_MEAN = 123
    G_MEAN = 117
    B_MEAN = 104

    dataset = VOCDetection(args.voc_root, [('2007', set_type)],
                           BaseTransform(300, (R_MEAN, G_MEAN, B_MEAN)),
                           AnnotationTransform())
    """
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    evaluation
    """
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(300, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)
