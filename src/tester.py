from utils.prior_box_creator import PriorBoxCreator
from utils.prior_box_manager import PriorBoxManager
from utils.utils import imread
from utils.utils import imresize
from keras.models import load_model
#from models import SSD300
from utils.XML_parser import XMLParser
from utils.utils import list_files_in_directory
from utils.utils import preprocess_input
from utils.utils import get_class_names
import pickle
import numpy as np
import matplotlib.pyplot as plt
from layers import Normalize
from layers import PriorBox


class Tester(object):
    def __init__(self, model,
            XML_files_path='../datasets/VOCdevkit/VOC2007/Annotations/'):
        self.model = model
        self.XML_files_path = XML_files_path
        # you should load everything before performing the tests
        # if not the order will make them fail.
        self.prior_boxes = None
        self.class_names = None


    def test_prior_boxes(self,
           original_prior_boxes_file='test_resources/prior_boxes_ssd300.pkl'):
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
        self.class_names = class_names
        XML_parser = XMLParser(self.XML_files_path, class_names)
        ground_truths = XML_parser.get_data()
        print('Number of images:', len(ground_truths.keys()))

    def test_model(self):
        inputs = []
        images = []
        file_paths = list_files_in_directory('test_resources/*.jpg')
        for file_path in file_paths:
            image_array = imread(file_path)
            image_array = imresize(image_array, (300, 300))
            images.append(image_array)
            inputs.append(image_array.copy())
        inputs = np.asarray(image_array)
        inputs = preprocess_input(inputs)
        predictions = self.model.predict(inputs, batch_size=1, verbose=1)
        prior_box_manager = PriorBoxManager(self.prior_boxes)
        results = prior_box_manager.detection_out(predictions)
        for image_arg, image_array in enumerate(images):
            # Parse the outputs.
            det_label = results[image_arg][:, 0]
            det_conf = results[image_arg][:, 1]
            det_xmin = results[image_arg][:, 2]
            det_ymin = results[image_arg][:, 3]
            det_xmax = results[image_arg][:, 4]
            det_ymax = results[image_arg][:, 5]

            # Get detections with confidence higher than 0.6.
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

            plt.imshow(image_array / 255.)
            currentAxis = plt.gca()

            for i in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[image_arg] * image_array.shape[1]))
                ymin = int(round(top_ymin[image_arg] * image_array.shape[0]))
                xmax = int(round(top_xmax[image_arg] * image_array.shape[1]))
                ymax = int(round(top_ymax[image_arg] * image_array.shape[0]))
                score = top_conf[i]
                label = int(top_label_indices[i])
                #label_name = self.class_names[label - 1]
                label_name = self.class_names[label]
                display_txt = '{:0.2f}, {}'.format(score, label_name)
                coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
                color = colors[label]
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False,
                                            edgecolor=color, linewidth=2))
                currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
            plt.show()


if __name__ == "__main__":
    model_path = '../trained_models/SSD300.11-1.96.hdf5'
    custom_layers = dict()
    custom_layers['Normalize'] = Normalize
    custom_layers['PriorBox'] = PriorBox
    model = load_model(model_path, custom_layers)
    tester = Tester(model)
    tester.test_prior_boxes()
    tester.test_XML_parser()
    tester.test_model()
