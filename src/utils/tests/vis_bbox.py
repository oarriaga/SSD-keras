import pickle
import matplotlib.pyplot as plt

"""
from datasets import DataManager
from utils.generator import ImageGenerator
from utils.boxes import to_point_form
from utils.boxes import create_prior_boxes
from tqdm import tqdm

datasets = ['VOC2007', 'VOC2012']
splits = ['trainval', 'trainval']
class_names = 'all'
difficult_boxes = True
batch_size = 32
box_scale_factors = [.1, .1, .2, .2]

dataset_manager = DataManager(datasets, splits, class_names, difficult_boxes)
train_data = dataset_manager.load_data()
val_data = test_data = DataManager('VOC2007', 'test').load_data()
class_names = dataset_manager.class_names
num_classes = len(class_names)

# generator
prior_boxes = to_point_form(create_prior_boxes())
generator = ImageGenerator(train_data, val_data, prior_boxes, batch_size,
                           box_scale_factors, num_classes)

steps_per_epoch = int(len(train_data) / batch_size)
train_generator = generator.flow('train')
data = []
for step_arg in tqdm(range(steps_per_epoch)):
    batch = next(train_generator)
    sample = batch[-1]['predictions']
    positive_mask = sample[:, :, 4] != 1
    positive_samples = sample[positive_mask]
    data.append(positive_samples)
"""

encoded_positive_boxes = pickle.load(open('encoded_positive_boxes.pkl', 'rb'))

encoded_cx = encoded_positive_boxes[:, 0]
plt.hist(encoded_cx, bins=25)
plt.title('encoded center x')
plt.show()

encoded_cy = encoded_positive_boxes[:, 1]
plt.hist(encoded_cy, bins=25)
plt.title('encoded center y')
plt.show()

encoded_w = encoded_positive_boxes[:, 2]
plt.hist(encoded_w, bins=50)
plt.title('encoded widths')
plt.show()

encoded_h = encoded_positive_boxes[:, 3]
plt.hist(encoded_h, bins=50)
plt.title('encoded heights')
plt.show()
