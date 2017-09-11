import numpy as np


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
    while len(idx > 0):
        i = idx[-1]
        keep[count] = i
        count += 1
        if len(idx) == 1:
            break
        idx = idx[:--1]
        xx1 = x1[idx]
        yy1 = y2[idx]
        xx2 = x2[idx]
        yy2 = y2[idx]

        xx1 = np.clip(xx1, min=x1[i])
        yy1 = np.clip(yy1, min=y1[i])
        xx2 = np.clip(xx2, max=x2[i])
        yy2 = np.clip(yy2, max=y2[i])

        w = xx2 - xx1
        h = yy2 - yy1

        w = np.clip(w, min=0.0)
        h = np.clip(h, min=0.0)
        inter = w*h
        rem_areas = area[idx]
        union = (rem_areas - inter) + area[i]
        IoU = inter/union
        iou_mask = IoU <= overlap
        idx = idx[iou_mask]
    return keep, count

