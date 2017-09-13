import numpy as np
import pickle
import torch


def nms_numpy(boxes, scores, overlap=0.5, top_k=200):
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
        # print(IoU)
        # break
        iou_mask = IoU <= overlap
        idx = idx[iou_mask]
        # print('numpy:', len(idx))
    return keep.astype(int), count


def nms_pytorch(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # print(w)
        # print(h)

        # print('torch_w', w)
        # print('torch_h', h)
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # print(IoU)
        # break
        # keep only elements with an IoU <= overlap
        # print(IoU.le(overlap))
        idx = idx[IoU.le(overlap)]
        # print('torch:', idx.size(0))
    return keep, count


torch_boxes = pickle.load(open('torch_boxes.pkl', 'rb'))
# print('torch_boxes:', torch_boxes)
torch_scores = pickle.load(open('torch_scores.pkl', 'rb'))
# print('torch_scores:', torch_scores.numpy())

numpy_boxes = pickle.load(open('numpy_boxes.pkl', 'rb'))
# print('numpy_boxes:', numpy_boxes)
numpy_scores = pickle.load(open('numpy_scores.pkl', 'rb'))
# print('numpy_scores:', numpy_scores)

numpy_idx, numpy_count = nms_numpy(numpy_boxes, numpy_scores, overlap=.5)
# numpy_idx, numpy_count = nms_numpy(torch_boxes.numpy(), torch_scores.numpy(), overlap=.5)
torch_idx, torch_count = nms_pytorch(torch_boxes, torch_scores, overlap=.5)
print('numpy_idx:', numpy_idx)
print('numpy_count:', numpy_count)
print('torch_idx:', torch_idx)
print('torch_count:', torch_count)

"""
numpy_selections = pickle.load(open('numpy_selections.pkl', 'rb'))
torch_selections = pickle.load(open('torch_selections.pkl', 'rb'))
torch_selections = torch_selections.numpy()
print('numpy', numpy_selections)
print('torch', torch_selections)
"""
