import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from utils import load_image

data_path_prefix = '../../datasets/COCO/'
annotations_path = data_path_prefix + 'annotations/instances_train2014.json'
image_path_prefix = data_path_prefix + 'images/train2014/'
class_names = ['person', 'dog', 'skateboard']

coco = COCO(annotations_path)

class_ids = coco.getCatIds(catNms=class_names)
image_ids = coco.getImgIds(catIds=class_ids)
image_data = coco.loadImgs(image_ids[np.random.randint(0,len(image_ids))])[0]
image_path = image_path_prefix + image_data['file_name']
image_array = load_image(image_path)
plt.imshow(image_array.astype('uint8'))
plt.show()


annotation_ids = coco.getAnnIds(imgIds=image_data['id'], catIds=class_ids)
annotations = coco.loadAnns(annotation_ids)
num_objects_in_image = len(annotations)
labels = []
box_coordinates = []
for object_arg in range(num_objects_in_image):
    labels.append(annotations[object_arg]['category_id'])
    coco_coordinates = annotations[object_arg]['bbox']
    x_min = coco_coordinates[0]
    y_min = coco_coordinates[1]
    x_max = x_min + coco_coordinates[2]
    y_max = x_max + coco_coordinates[3]
    box_coordinates.append([x_min, y_min, x_max, y_max])
box_coordinates = np.asarray(box_coordinates)
