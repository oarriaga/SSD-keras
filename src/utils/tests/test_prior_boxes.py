from datasets import DataManager
from models import SSD300, make_prior_boxes
from utils.boxes import create_prior_boxes, to_point_form
import numpy as np

model = SSD300()
prior_boxes_1 = make_prior_boxes(model)
prior_boxes_2 = to_point_form(create_prior_boxes())
np.allclose(prior_boxes_1, prior_boxes_2)
