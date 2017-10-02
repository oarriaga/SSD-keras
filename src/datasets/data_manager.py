from .voc_parser import VOCParser
from .data_utils import get_class_names


class DataManager(object):
    def __init__(self, dataset_name='VOC2007', split='train',
                 class_names='all', with_difficult_objects=False,
                 dataset_path='../datasets/VOCdevkit/'):

        self.dataset_name = dataset_name
        self.split = split
        self.with_difficult_objects = with_difficult_objects
        self.class_names = class_names
        if class_names == 'all':
            self.class_names = get_class_names(self.dataset_name)
        self.dataset_path = dataset_path
        self.images_path = None
        self.arg_to_class = None
        self.ground_truth_data = None

    def load_data(self):
        if self.dataset_name == 'VOC2007':
            self._load_VOC2007()
        return self.ground_truth_data

    def _load_VOC2007(self):
        self.parser = VOCParser(self.dataset_name,
                                self.split,
                                self.class_names,
                                self.with_difficult_objects,
                                self.dataset_path)
        self.images_path = self.parser.images_path
        self.arg_to_class = self.parser.arg_to_class
        self.ground_truth_data = self.parser.load_data()
