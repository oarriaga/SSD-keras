from utils.evaluation import record_detections
from utils.datasets import get_class_names
from models.ssd import SSD300
# from tf_eval import evaluate_detection_results_pascal_voc
import pickle


dataset_name = 'VOC2007'
data_prefix = '../datasets/VOCdevkit/VOC2007/Annotations/'
image_prefix = '../datasets/VOCdevkit/VOC2007/JPEGImages/'
weights_path = '../trained_models/SSD300_weights.hdf5'
iou_threshold = .5
iou_nms_threshold = .45
# class_names = ['background', 'aeroplane', 'bird']
class_names = get_class_names()
model = SSD300(weights_path=weights_path)
data_records = record_detections(model, dataset_name, data_prefix,
                                 image_prefix, class_names, iou_threshold,
                                 iou_nms_threshold)
pickle.dump(data_records, open('data_records.pkl', 'wb'))

"""
data_records = pickle.load(open('data_records.pkl', 'rb'))
class_ids = list(range(len(class_names)))
class_ids = [int(class_id) for class_id in class_ids]
categories = []
for class_id, class_name in zip(class_ids, class_names):
    category = dict = {}
    category['id'] = class_id
    category['name'] = class_name
    categories.append(category)
# categories = dict(zip(class_ids, class_names))
# categories = dict(zip(list(range(len(class_names))), class_names))
evaluate_detection_results_pascal_voc(data_records, categories)
"""
