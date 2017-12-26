import numpy as np
try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None


class COCOParser(object):
    def __init__(self, annotations_path, class_names='all'):
        self.coco = COCO(annotations_path)
        self.class_names = class_names
        self.data = dict()
        if self.class_names == 'all':
            class_data = self.coco.loadCats(self.coco.getCatIds())
            self.class_names = [class_['name'] for class_ in class_data]
            coco_ids = [class_['id'] for class_ in class_data]
            one_hot_ids = list(range(1, len(coco_ids) + 1))
            self.coco_id_to_class_arg = dict(zip(coco_ids, one_hot_ids))
            self.class_names = ['background'] + self.class_names
            self.num_classes = len(self.class_names)
            self.image_ids = self.coco.getImgIds()
        else:
            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
            catIds = self.coco.getCatIds(catNms=self.class_names)
            self.image_ids = self.coco.getImgIds(catIds=catIds)
            # imgIds = self.coco.getImgIds(imgIds=[324158])

    def load_data(self):
        # image_ids = self.coco.getImgIds()
        for image_id in self.image_ids:
            image_data = self.coco.loadImgs(image_id)[0]
            image_file_name = image_data['file_name']
            width = float(image_data['width'])
            height = float(image_data['height'])
            annotation_ids = self.coco.getAnnIds(imgIds=image_data['id'])
            annotations = self.coco.loadAnns(annotation_ids)
            num_objects_in_image = len(annotations)
            if num_objects_in_image == 0:
                continue
            image_ground_truth = []
            for object_arg in range(num_objects_in_image):
                coco_id = annotations[object_arg]['category_id']
                class_arg = self.coco_id_to_class_arg[coco_id]
                one_hot_class = self._to_one_hot(class_arg)
                coco_coordinates = annotations[object_arg]['bbox']

                x_min = coco_coordinates[0] / width
                y_min = coco_coordinates[1] / height
                x_max = x_min + coco_coordinates[2] / width
                y_max = y_min + coco_coordinates[3] / height

                ground_truth = [x_min, y_min, x_max, y_max] + one_hot_class
                image_ground_truth.append(ground_truth)
            image_ground_truth = np.asarray(image_ground_truth)
            if len(image_ground_truth.shape) == 1:
                image_ground_truth = np.expand_dims(image_ground_truth, 0)
            self.data[image_file_name] = image_ground_truth
        return self.data

    def _to_one_hot(self, class_arg):
        one_hot_vector = [0] * self.num_classes
        one_hot_vector[class_arg] = 1
        return one_hot_vector
