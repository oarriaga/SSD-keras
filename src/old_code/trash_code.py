
# tests
boxes = ground_truth_data['000009.jpg']
encode_box = bounding_box_utils.encode_box
#calculate_iou = bounding_box_utils.calculate_intersection_over_union
# returns a value for each box for every prior (num_boxes, num_priors)
#iou = np.apply_along_axis(calculate_iou, 1, boxes[:, :4])
#assign_mask = iou[0, :] > .5
num_priors = len(priors)
#encoded_box = np.zeros((num_priors, 4 + 1))
encoded_boxes = np.apply_along_axis(encode_box, 1, boxes[:, :4])
encoded_boxes = encoded_boxes.reshape(-1, num_priors, 5)
best_iou = encoded_boxes[:, :, -1].max(axis=0) #? shouldn't it be axis = 1
best_iou_indices = encoded_boxes[:, :, -1].argmax(axis=0) # ? same here
best_iou_mask = best_iou > 0
best_iou_indices2 = best_iou_indices[best_iou_mask]
num_assigned_boxes = len(best_iou_indices2) # ?
encoded_boxes2 = encoded_boxes[:, best_iou_mask, :]

num_classes = 21
assignment = np.zeros(shape=(num_priors,  4 + num_classes + 8))
assignment[:, 4] = 1.0 # is this the background?

assignment[:, 4][best_iou_mask] = 0
assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_indices2, 4:]
assignment[:, -8][best_iou_mask] = 1



def test_saturation():
    image_name = 'image.jpg'
    generator = ImageGenerator()
    image_array = generator._imread(image_name)
    transformed_image_array = generator.saturation(image_array)
    transformed_image_array = transformed_image_array.astype('uint8')
    plt.imshow(transformed_image_array)
    plt.show()

def test_brightness(brightness_var=.5):
    image_name = 'image.jpg'
    generator = ImageGenerator(brightness_var=brightness_var)
    image_array = generator._imread(image_name)
    transformed_image_array = generator.brightness(image_array)
    transformed_image_array = transformed_image_array.astype('uint8')
    plt.imshow(transformed_image_array)
    plt.show()

def test_contrast():
    image_name = 'image.jpg'
    generator = ImageGenerator()
    image_array = generator._imread(image_name)
    transformed_image_array = generator.contrast(image_array)
    transformed_image_array = transformed_image_array.astype('uint8')
    plt.imshow(transformed_image_array)
    plt.show()

def test_lighting(lighting_std=.5):
    image_name = 'image.jpg'
    generator = ImageGenerator(lighting_std=lighting_std)
    image_array = generator._imread(image_name)
    transformed_image_array = generator.contrast(image_array)
    transformed_image_array = transformed_image_array.astype('uint8')
    plt.imshow(transformed_image_array)
    plt.show()

def test_horizontal_flip():
    image_name = 'image.jpg'
    generator = ImageGenerator()
    image_array = generator._imread(image_name)
    transformed_image_array = generator.horizontal_flip(image_array)
    transformed_image_array = transformed_image_array.astype('uint8')
    plt.imshow(transformed_image_array)
    plt.show()

def test_vertical_flip():
    image_name = 'image.jpg'
    generator = ImageGenerator()
    image_array = generator._imread(image_name)
    transformed_image_array = generator.vertical_flip(image_array)
    transformed_image_array = transformed_image_array.astype('uint8')
    plt.imshow(transformed_image_array)
    plt.show()





