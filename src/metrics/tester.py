import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.misc import imread

from keras.preprocessing import image as image_processor
from models import SSD300
from utils.prior_box_creator import PriorBoxCreator
from utils.prior_box_manager import PriorBoxManager
from utils.XML_parser import XMLParser
from utils.utils import list_files_in_directory
from utils.utils import preprocess_input
from utils.utils import get_class_names
from utils.utils import load_image


class Tester(object):
    def __init__(self, model,
            XML_files_path='../datasets/VOCdevkit/VOC2007/Annotations/',
            class_names=None):
        self.model = model
        self.XML_files_path = XML_files_path
        # you should load everything before performing the tests
        # if not the order will make them fail.
        self.prior_boxes = None
        self.class_names = class_names
        if self.class_names == None:
            self.class_names = get_class_names('VOC2007')
            #self.class_names = pickle.load(open('utils/coco_classes.p', 'rb'))


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

    def test_XML_parser(self):
        #class_names = None
        #self.class_names = class_names
        XML_parser = XMLParser(self.XML_files_path, self.class_names)
        ground_truths = XML_parser.get_data()
        print('Number of images:', len(ground_truths.keys()))

    def test_image_loading(self):
        test_image_path = 'test_resources/fish-bike.jpg'
        keras_image = image_processor.load_img(test_image_path)
        keras_image = image_processor.img_to_array(keras_image)
        scipy_image = imread(test_image_path)
        scipy_image = scipy_image.astype('float32')
        value = np.all(keras_image == scipy_image)
        print('Image test:', value)

    def test_model(self):
        inputs = []
        images = []
        file_paths = list_files_in_directory('test_resources/*.jpg')
        for file_path in file_paths:
            image_array = load_image(file_path, False, target_size=(300, 300))
            images.append(imread(file_path))
            inputs.append(image_array)
        inputs = np.asarray(inputs, dtype='float32')
        inputs = preprocess_input(inputs)
        predictions = self.model.predict(inputs, batch_size=1, verbose=1)
        prior_box_manager = PriorBoxManager(self.prior_boxes,
                            box_scale_factors=[.1, .1, .2, .2])
        results = prior_box_manager.detection_out(predictions)
        for i, img in enumerate(images):
            # Parse the outputs.
            det_label = results[i][:, 0]
            det_conf = results[i][:, 1]
            det_xmin = results[i][:, 2]
            det_ymin = results[i][:, 3]
            det_xmax = results[i][:, 4]
            det_ymax = results[i][:, 5]

            # Get detections with confidence higher than 0.6.
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.8]

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

            plt.imshow(img / 255.)
            currentAxis = plt.gca()

            for i in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * img.shape[1]))
                ymin = int(round(top_ymin[i] * img.shape[0]))
                xmax = int(round(top_xmax[i] * img.shape[1]))
                ymax = int(round(top_ymax[i] * img.shape[0]))
                score = top_conf[i]
                label = int(top_label_indices[i])
                #label_name = voc_classes[label - 1]
                #label_name = self.class_names[label - 1]
                label_name = self.class_names[label]
                display_txt = '{:0.2f}, {}'.format(score, label_name)
                coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
                color = colors[label]
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
            plt.show()




if __name__ == "__main__":
    #weights_path = '../trained_models/ssd300_weights.34-1.54.hdf5'
    weights_path = '../trained_models/ssd300_weights.17-1.51.hdf5'
    #weights_path = '../trained_models/SSD300_COCO_weights.08-0.35.hdf5'
    model = SSD300(num_classes=21)
    model.load_weights(weights_path)
    tester = Tester(model)
    tester.test_prior_boxes()
    tester.test_XML_parser()
    tester.test_image_loading()
    tester.test_model()
