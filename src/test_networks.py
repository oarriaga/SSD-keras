from pytorch_tests.pytorch_networks import build_ssd
import pickle
from torch.autograd import Variable
import torch
import tensorflow as tf

from models.ssd import SSD300
from preprocessing import substract_mean
# from utils.inference import predict
from utils.boxes import create_prior_boxes
import numpy as np
from scipy.misc import imread, imresize
from datasets import DataManager
from datasets import get_class_names
from tqdm import tqdm


from utils.boxes import unregress_boxes
# from utils.tf_boxes import apply_non_max_suppression


# functions
def preprocess_pytorch_input(image):
    x = image.astype(np.float32)
    x -= (123.68, 116.779, 103.939)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    """
    if torch.cuda.is_available():
        xx = xx.cuda()
    """
    return xx


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
        # print(IoU)
        iou_mask = IoU <= overlap
        idx = idx[iou_mask]
        # print('numpy:', len(idx))
    return keep.astype(int), count


def softmax(x, axis=1):
    x = x.astype(float)
    num_samples, num_classes = x.shape
    new_x = np.zeros_like(x)
    for sample_arg in range(num_samples):
        x_sample = x[sample_arg]
        new_x[sample_arg] = np.exp(x_sample) / np.sum(np.exp(x_sample), axis=0)
    return new_x


def load_image(image_path, target_size=None):
    image_array = imread(image_path)
    height, width = image_array.shape[:2]
    if target_size is not None:
        image_array = imresize(image_array, target_size)
    return image_array, (height, width)


class Detect():
    def __init__(self, num_classes=21, bkg_label=0, top_k=200,
                 conf_thresh=0.099, nms_thresh=.45):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = [.1, .1, .2, .2]
        self.output = np.zeros((1, self.num_classes, self.top_k, 5))

    def forward(self, box_data, prior_boxes):
        box_data = np.squeeze(box_data)
        regressed_boxes = box_data[:, :4]
        class_predictions = box_data[:, 4:]
        decoded_boxes = unregress_boxes(regressed_boxes, prior_boxes,
                                        self.variance)
        for class_arg in range(1, self.num_classes):
            conf_mask = class_predictions[:, class_arg] >= (self.conf_thresh)
            scores = class_predictions[:, class_arg][conf_mask]
            if len(scores) == 0:
                continue
            boxes = decoded_boxes[conf_mask]
            """
            indices = apply_non_max_suppression(boxes, scores,
                                                self.nms_thresh,
                                                self.top_k)
            count = len(indices)
            """
            pickle.dump(boxes, open('numpy_boxes.pkl', 'wb'))
            pickle.dump(scores, open('numpy_scores.pkl', 'wb'))

            indices, count = nms(boxes, scores, self.nms_thresh, self.top_k)
            # print('indices:', indices)
            # print('indices_shape:', indices.shape)
            scores = np.expand_dims(scores, -1)
            # print('scores_shape:', scores[indices].shape)
            # print('boxes_shape:', boxes[indices].shape)
            selections = np.concatenate((scores[indices[:count]],
                                        boxes[indices[:count]]), axis=1)
            pickle.dump(selections, open('numpy_selections.pkl', 'wb'))
            # print('selections_shape', selections.shape)
            # print('count:', count)
            # self.output[0, class_arg, :count] = selections
            self.output[0, class_arg, :count] = selections
        return self.output


# parameters
trained_weights_path = '../trained_models/ssd_300_VOC0712.pth'
input_size = 300
num_classes = 21
iou_threshold = .45
lower_probability_threshold = .1
background_index = 0
dataset_name = 'VOC2007'
data_prefix = '../datasets/VOCdevkit/VOC2007/Annotations/'
image_prefix = '../datasets/VOCdevkit/VOC2007/JPEGImages/'
selected_classes = get_class_names(dataset_name)
prior_boxes = create_prior_boxes()

# loading pytorch model
pytorch_ssd = build_ssd('test', input_size, num_classes)
pytorch_ssd.load_weights(trained_weights_path)

# loading keras model
weights_path = '../trained_models/SSD300_weights.hdf5'
with tf.device('/cpu:0'):
    model = SSD300(weights_path=weights_path)

split = 'train'
data_manager = DataManager(dataset_name, split, selected_classes)

ground_truth_data = data_manager.load_data()
detect = Detect()
# image_names = sorted(list(ground_truth_data.keys()))
po = None
ko = None
a = 0
image_names = list(ground_truth_data.keys())
for image_name in tqdm(image_names):
    a = a + 1
    ground_truth_sample = ground_truth_data[image_name]
    image_path = image_prefix + image_name

    rgb_image, image_size = load_image(image_path, target_size=(300, 300))
    pytorch_image = preprocess_pytorch_input(rgb_image)
    pytorch_output = pytorch_ssd(pytorch_image)
    po = pytorch_output.data.numpy()

    # keras_image = preprocess_images(rgb_image)
    keras_image = substract_mean(rgb_image)
    keras_image_input = np.expand_dims(keras_image, axis=0)
    keras_output = model.predict(keras_image_input)
    ko = detect.forward(keras_output, prior_boxes)

    numpy_selections = pickle.load(open('numpy_selections.pkl', 'rb'))
    torch_selections = pickle.load(open('torch_selections.pkl', 'rb'))
    torch_selections = torch_selections.numpy()
    print('numpy', numpy_selections)
    print('torch', torch_selections)
    print(np.sum(np.abs(torch_selections - numpy_selections) > .1))


    if a > 10:
        break
