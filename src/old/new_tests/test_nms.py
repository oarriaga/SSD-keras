from pytorch_tests.pytorch_networks import build_ssd
from torch.autograd import Variable
import torch
import tensorflow as tf

from models.ssd import SSD300
from utils.inference import infer_from_path
from utils.boxes import create_prior_boxes
from datasets import get_class_names
import numpy as np
from scipy.misc import imread, imresize
from visualizer import draw_image_boxes


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


def load_image(image_path, target_size=None):
    image_array = imread(image_path)
    height, width = image_array.shape[:2]
    if target_size is not None:
        image_array = imresize(image_array, target_size)
    return image_array, (height, width)


# parameters
trained_weights_path = '../trained_models/ssd_300_VOC0712.pth'
input_size = 300
num_classes = 21
iou_threshold = .5
lower_probability_threshold = .01
background_index = 0
dataset_name = 'VOC2007'
data_prefix = '../datasets/VOCdevkit/VOC2007/Annotations/'
image_prefix = '../datasets/VOCdevkit/VOC2007/JPEGImages/'
selected_classes = get_class_names(dataset_name)
prior_boxes = create_prior_boxes()


# loading keras model
image_path = '../images/boys.jpg'
weights_path = '../trained_models/SSD300_weights.hdf5'
with tf.device('/cpu:0'):
    model = SSD300(weights_path=weights_path)

# infer with keras
keras_detections = infer_from_path(image_path, model)

# loading pytorch model
pytorch_ssd = build_ssd('test', input_size, num_classes)
pytorch_ssd.load_weights(trained_weights_path)

original_image_array = load_image(image_path)[0]
rgb_image, image_size = load_image(image_path, target_size=(300, 300))
pytorch_image = preprocess_pytorch_input(rgb_image)
pytorch_output = pytorch_ssd(pytorch_image)
# pytorch_detection = pytorch_output.data.numpy()
pytorch_detections = pytorch_output.data
scale = torch.Tensor([rgb_image.shape[1::-1], rgb_image.shape[1::-1]])
print(scale)
pytorch_labels = selected_classes[1:]
pts = []

h, w, channels = rgb_image.shape
# skip j = 0, because it's the background class
for j in range(1, pytorch_detections.size(1)):
    i = 0
    dets = pytorch_detections[0, j, :]
    value = 0.
    mask = dets[:, 0].gt(value).expand(5, dets.size(0)).t()
    dets = torch.masked_select(dets, mask).view(-1, 5)
    if dets.dim() == 0:
        continue
    boxes = dets[:, 1:].cpu().numpy()
    boxes[:, 0] *= w
    boxes[:, 2] *= w
    boxes[:, 1] *= h
    boxes[:, 3] *= h
    scores = dets[:, 0].cpu().numpy()
    cls_dets = np.hstack((boxes,
                         scores[:, np.newaxis])).astype(np.float32,
                                                        copy=False)
    pts.append(cls_dets)
pytorch_detections = np.concatenate(pts, axis=0)
# pytorch_detections = np.asarray(pts)
draw_image_boxes(pytorch_detections[:, :-1], original_image_array)
