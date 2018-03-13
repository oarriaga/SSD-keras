from .voc_parser import VOCParser
from .data_utils import get_class_names
from .data_utils import merge_two_dictionaries


class DataManager(object):
    def __init__(self, dataset_name='VOC2007', split='train',
                 class_names='all', with_difficult_objects=True,
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

    def load_data(self):
        if self.dataset_name == 'VOC2007':
            ground_truth_data = self._load_VOC(self.dataset_name, self.split)
        if self.dataset_name == 'VOC2012':
            ground_truth_data = self._load_VOC(self.dataset_name, self.split)
        if isinstance(self.dataset_name, list):
            if not isinstance(self.split, list):
                raise Exception("'split' should also be a list")
            if set(self.dataset_name).issubset(['VOC2007', 'VOC2012']):
                data_0 = self._load_VOC(self.dataset_name[0], self.split[0])
                data_1 = self._load_VOC(self.dataset_name[1], self.split[1])
                ground_truth_data = merge_two_dictionaries(data_0, data_1)

        return ground_truth_data

    def _load_VOC(self, dataset_name, split):
        self.parser = VOCParser(dataset_name,
                                split,
                                self.class_names,
                                self.with_difficult_objects,
                                self.dataset_path)
        self.images_path = self.parser.images_path
        self.arg_to_class = self.parser.arg_to_class
        ground_truth_data = self.parser.load_data()
        return ground_truth_data
