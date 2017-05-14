from utils.prior_box_creator import PriorBoxCreator
from models import SSD300
from utils.XML_parser import XMLParser
import pickle


class Tester(object):
    def __init__(self,
            XML_files_path='../datasets/VOCdevkit/VOC2007/Annotations/'):
        self.model = SSD300()
        self.XML_files_path = XML_files_path


    def test_prior_boxes(self,
           original_prior_boxes_file='prior_boxes_ssd300.pkl'):
        prior_box_creator = PriorBoxCreator(self.model)
        self.prior_boxes = prior_box_creator.create_boxes()
        original_prior_boxes = pickle.load(
                        open(original_prior_boxes_file, 'rb'))
        prior_box_difference = self.prior_boxes - original_prior_boxes[:, :4]
        print('shape {}, max value {}, min value {}'.format(
                                        prior_box_difference.shape,
                                        prior_box_difference.max(),
                                        prior_box_difference.min()))

    def test_XML_parser(self, class_names=['bottle', 'sofa', 'chair']):
        class_names = None
        XML_parser = XMLParser(self.XML_files_path, class_names)
        ground_truths = XML_parser.get_data()
        print('Number of images:', len(ground_truths.keys()))


if __name__ == "__main__":
    tester = Tester()
    tester.test_prior_boxes()
    tester.test_XML_parser()
