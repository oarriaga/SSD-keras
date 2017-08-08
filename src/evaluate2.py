from utils.evaluation import record_detections
from models.ssd import SSD300
import pickle

dataset_name = 'VOC2007'
data_prefix = '../datasets/VOCdevkit/VOC2007/Annotations/'
image_prefix = '../datasets/VOCdevkit/VOC2007/JPEGImages/'
weights_path = '../trained_models/SSD300_weights.hdf5'
iou_threshold = .5
iou_nms_threshold = .45

model = SSD300(weights_path=weights_path)
data_records = record_detections(model, dataset_name, data_prefix,
                                 image_prefix, iou_threshold,
                                 iou_nms_threshold)
pickle.dump(data_records, open('data_records.pkl', 'rb'))
