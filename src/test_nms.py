from utils.boxes import apply_non_max_suppression
import numpy as np

boxes = np.random.rand(100, 4)
scores = np.random.rand(100)


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
        # print('old', i)
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
        # print(IoU)
        iou_mask = IoU <= overlap
        idx = idx[iou_mask]
        # print('numpy:', len(idx))
    return keep.astype(int), count


a = apply_non_max_suppression(boxes, scores, .45, 200)
b = nms(boxes, scores, 0.45, 200)
print(a == b)
