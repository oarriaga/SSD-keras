import pickle
from utils.prior_box_creator import PriorBoxCreator
from ssd import SSD300
from utils.utils import flatten_prior_boxes
model = SSD300((300,300,3))
prior_box_creator = PriorBoxCreator(model)
prior_boxes = prior_box_creator.create_boxes()
prior_boxes = flatten_prior_boxes(prior_boxes)
previous_prior_boxes = pickle.load(open('prior_boxes_ssd300.pkl','rb'))

