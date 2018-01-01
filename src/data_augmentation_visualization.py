from utils.data_augmentation import ConvertFromInts
from utils.data_augmentation import ToAbsoluteCoords
from utils.data_augmentation import PhotometricDistort2
from utils.data_augmentation import Expand
from utils.data_augmentation import RandomSampleCrop
from utils.data_augmentation import HorizontalFlip
from utils.data_augmentation import ToPercentCoords
from utils.data_augmentation import Resize
from utils.data_augmentation import SubtractMeans
from utils.preprocessing import load_image
from utils.preprocessing import B_MEAN, G_MEAN, R_MEAN
from datasets import DataManager
import matplotlib.pyplot as plt

data_manager = DataManager('VOC2007', 'train')
class_names = data_manager.class_names
train_data = data_manager.load_data()

image_path = sorted(list(train_data.keys()))[7]
box_data = train_data[image_path]
image_array = load_image(image_path, RGB=False).copy()
data = (image_array, box_data[:, :4], box_data[:, 4:])

convert_from_ints = ConvertFromInts()
to_absolute_coords = ToAbsoluteCoords()
photometric_distort = PhotometricDistort2()
expand = Expand((B_MEAN, G_MEAN, R_MEAN))
random_sample_crop = RandomSampleCrop()
horizontal_flip = HorizontalFlip()
to_percent_coords = ToPercentCoords()
resize = Resize(300)
subtract_means = SubtractMeans((B_MEAN, G_MEAN, R_MEAN))

plt.imshow(image_array[..., ::-1])
plt.show()
data = convert_from_ints(*data)
data = to_absolute_coords(*data)
data = photometric_distort(*data)
plt.imshow(data[0][..., ::-1].astype('uint8'))
plt.show()
data = expand(*data)
plt.imshow(data[0][..., ::-1].astype('uint8'))
plt.show()
data = random_sample_crop(*data)
plt.imshow(data[0][..., ::-1].astype('uint8'))
plt.show()
data = horizontal_flip(*data)
data = to_percent_coords(*data)
data = resize(*data)
data = subtract_means(*data)
