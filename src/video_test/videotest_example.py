from videotest import VideoTest

import sys
sys.path.append("..")
from mini_models import mini_SSD300

input_shape = (300,300,3)

# Change this if you run with other classes than VOC
class_names=['chair', 'bottle', 'sofa', 'tvmonitor', 'diningtable']

NUM_CLASSES = len(class_names)

num_classes = len(class_names) + 1

model = mini_SSD300(input_shape, num_classes)

# Change this path if you want to use your own trained weights
model.load_weights('../../trained_models/model_checkpoints/weights.02-2.83.hdf5')
vid_test = VideoTest(class_names, model, input_shape)

# To test on webcam 0, remove the parameter (or change it to another number
# to test on that webcam)
#vid_test.run('path/to/your/video.mkv')
vid_test.run(0)
