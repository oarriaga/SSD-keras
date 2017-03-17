import numpy as np

def non_max_suppression_fast(boxes, overlap_threshold):
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	picked_indices = []

	# grab the coordinates of the bounding boxes
	x_min = boxes[:, 0]
	y_min = boxes[:, 1]
	x_max = boxes[:, 2]
	y_max = boxes[:, 3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	areas = (x_max - x_min + 1) * (y_max - y_min + 1)
	indices = np.argsort(y_max)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(indices) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(indices) - 1
		index = indices[last]
		picked_indices.append(index)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx_min = np.maximum(x_min[index], x_min[indices[:last]])
		yy_min = np.maximum(y_min[index], y_min[indices[:last]])
		xx_max = np.minimum(x_max[index], x_max[indices[:last]])
		yy_max = np.minimum(y_max[index], y_max[indices[:last]])

		# compute the width and height of the bounding box
		width = np.maximum(0, xx_max - xx_min + 1)
		height = np.maximum(0, yy_max - yy_min + 1)

		# compute the ratio of overlap
		overlap = (width * height) / areas[indices[:last]]

		# delete all indexes from the index list that have
		indices = np.delete(indices, np.concatenate(([last],
			np.where(overlap > overlap_threshold)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[picked_indices].astype("int")
